import math
import typing

import einops as eo
import torch


class DiTConfig(typing.NamedTuple):
    vocab_size: int = 16
    hidden_dim: int = 16
    num_heads: int = 4
    num_layers: int = 2
    eps: float = 1e-8
    sequence_length: int = 64
    ada_dim: int = 512
    time_dim: int = 256


def nearest_multiple_of_128(n: int) -> int:
    return int(128 * max(1, n // 128))


def zero_init(layer: torch.nn.Module) -> torch.nn.Module:
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

    return layer


def xavier_init(layer: torch.nn.Module) -> torch.nn.Module:
    torch.nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

    return layer


def normal_init(layer: torch.nn.Module) -> torch.nn.Module:
    torch.nn.init.normal_(layer.weight, std=0.02)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

    return layer


def build_rope_cache(seq_len: int, n_elem: int) -> torch.Tensor:
    freqs = torch.logspace(
        start=math.log10(0.5 * math.pi),
        end=math.log10(0.5 * math.pi * seq_len),
        steps=n_elem // 2,
    )
    t = torch.arange(seq_len, device=freqs.device) / seq_len
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    freqs = freqs[: x.size(2)]
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs = freqs.view(1, 1, xshaped.size(2), xshaped.size(3), 2)
    x_out = torch.stack(
        [
            xshaped[..., 0] * freqs[..., 0] - xshaped[..., 1] * freqs[..., 1],
            xshaped[..., 1] * freqs[..., 0] + xshaped[..., 0] * freqs[..., 1],
        ],
        -1,
    )
    return x_out.flatten(3).type_as(x)


def checkpoint(
    enabled: bool,
    function: typing.Callable[..., torch.Tensor],
    x: torch.Tensor,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> torch.Tensor:
    if x.requires_grad and enabled:
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, x, *args, **kwargs)  # type: ignore
    return function(x, *args, **kwargs)


class FourierFeatures(torch.nn.Module):
    def __init__(self, time_dim: int, out_dim: int, std: float = 1.0) -> None:
        super().__init__()

        assert time_dim % 2 == 0
        self.register_buffer("scales", torch.randn([time_dim // 2, 1]) * std)
        self.to_out = normal_init(torch.nn.Linear(time_dim, out_dim, bias=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * input @ self.scales.T
        fouriered = torch.cat([f.cos(), f.sin()], dim=-1)
        return self.to_out(fouriered)


class Modulation(torch.nn.Module):
    def __init__(self, dim: int, ada_dim: int, use_gate: bool = True) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.proj = torch.nn.Sequential(
            torch.nn.SiLU(),
            zero_init(
                torch.nn.Linear(ada_dim, dim * 3 if use_gate else dim * 2, bias=True)
            ),
        )

    def forward(
        self, x: torch.Tensor, ada_cond: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mod = self.proj(ada_cond.unsqueeze(1)).chunk(3 if self.use_gate else 2, dim=-1)
        x = x * (mod[0] + 1) + mod[1]
        return (x, mod[2]) if self.use_gate else x


class Attention(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ada_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = xavier_init(
            torch.nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        )
        self.o_proj = xavier_init(torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.norm = torch.nn.LayerNorm(hidden_dim, bias=False)
        self.modulation = Modulation(hidden_dim, ada_dim=ada_dim, use_gate=True)

        self.q_norm = torch.nn.LayerNorm(self.head_dim, bias=False)
        self.k_norm = torch.nn.LayerNorm(self.head_dim, bias=False)

    def forward(
        self, x: torch.Tensor, ada_cond: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm(x)
        x, gate = self.modulation(x, ada_cond)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [
            eo.rearrange(
                t, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=self.head_dim
            )
            for t in (q, k, v)
        ]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = eo.rearrange(x, "b nh s hd -> b s (nh hd)")
        return self.o_proj(x) * gate


class Mlp(torch.nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, ada_dim: int) -> None:
        super().__init__()

        self.fc1 = xavier_init(
            torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        )
        self.fc2 = xavier_init(
            torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        )
        self.fc3 = xavier_init(
            torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)
        )
        self.norm = torch.nn.LayerNorm(hidden_dim, bias=False)
        self.modulation = Modulation(hidden_dim, ada_dim=ada_dim, use_gate=True)

    def forward(self, x: torch.Tensor, ada_cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x, gate = self.modulation(x, ada_cond=ada_cond)
        x = torch.nn.functional.silu(self.fc1(x)) * self.fc2(x)
        return self.fc3(x) * gate


class DiTLayer(torch.nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()

        intermediate_dim = int(128 * max(1, (int(config.hidden_dim * 8 / 3)) // 128))

        self.attn = Attention(config.hidden_dim, config.num_heads, config.ada_dim)
        self.mlp = Mlp(config.hidden_dim, intermediate_dim, config.ada_dim)
        self.use_checkpointing = False

    def enable_checkpointing(self) -> None:
        self.use_checkpointing = True

    def forward(
        self, x: torch.Tensor, ada_cond: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        x = x + checkpoint(
            self.use_checkpointing and self.training, self.attn, x, ada_cond, freqs
        )
        return x + checkpoint(
            self.use_checkpointing and self.training, self.mlp, x, ada_cond
        )


class DiT(torch.nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config

        self.layers = torch.nn.ModuleList(
            [DiTLayer(config) for _ in range(config.num_layers)]
        )

        self.norm = torch.nn.LayerNorm(config.hidden_dim, bias=False)
        self.modulation = Modulation(
            config.hidden_dim, ada_dim=config.ada_dim, use_gate=False
        )

        self.time_cond_net = FourierFeatures(config.time_dim, config.ada_dim)

        self.word_embedding = torch.nn.Embedding(
            config.vocab_size, embedding_dim=config.hidden_dim
        )

        self.head = zero_init(
            torch.nn.Linear(config.hidden_dim, config.vocab_size),
        )

        self.register_buffer(
            "freqs",
            build_rope_cache(
                seq_len=config.sequence_length,
                n_elem=config.hidden_dim // config.num_heads,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = self.word_embedding(x)
        ada_cond = self.time_cond_net(time)
        for layer in self.layers:
            x = layer(x, ada_cond=ada_cond, freqs=self.freqs)
        x = self.norm(x)
        x = self.modulation(x, ada_cond=ada_cond)
        return self.head(x)
