import typing
from collections import Counter

import datasets
import pydantic_settings as pyds
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from zipf_diffusion.dit import DiT, DiTConfig


class TrainConfig(pyds.BaseSettings):
    lr: float
    max_steps: int
    batch_size: int
    dit_config: DiTConfig
    dataset_name: str = "tiny_shakespeare"
    num_workers: int
    tokenizer_name: str = "gpt2"
    grad_norm: float = 1.0
    test_every: int
    chunk_size: int
    zipf_lower_bound: float
    blank_is_noise: bool
    generate_kwargs: dict
    num_warmup_steps: int


class ZipfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        chunk_size: int,
        split: str,
    ) -> None:
        super().__init__()
        dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)[split]  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_function(examples: dict[str, typing.Any]) -> dict[str, list[int]]:
            return self.tokenizer(examples["text"])

        dataset = dataset.map(tokenize_function, batched=True)  # type: ignore

        all_tokens = []
        for tokens in dataset["input_ids"]:
            for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
                token_chunk = tokens[i : i + chunk_size]
                if len(token_chunk) == chunk_size:
                    all_tokens.append(token_chunk)

        self.tokenized_dataset = datasets.Dataset.from_dict({"input_ids": all_tokens})

    def __len__(self) -> int:
        return len(self.tokenized_dataset)

    def __getitem__(self, index: int) -> dict:
        out = self.tokenized_dataset[index]
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        return out


class TrainState(typing.NamedTuple):
    model: DiT
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    dataset: ZipfDataset


def init_train_state(config: TrainConfig, device: torch.device) -> TrainState:
    model = DiT(config.dit_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Create scheduler for linear warmup and linear decay
    def lr_lambda(current_step: int):
        if current_step < config.num_warmup_steps:
            return float(current_step) / float(max(1, config.num_warmup_steps))
        # linear decay
        return max(
            0.0,
            float(config.max_steps - current_step)
            / float(max(1, config.max_steps - config.num_warmup_steps)),
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_steps,
    )

    dataset = ZipfDataset(
        dataset_name=config.dataset_name,
        tokenizer_name=config.tokenizer_name,
        chunk_size=config.chunk_size,
        split="train",
    )
    return TrainState(
        model=model, optimizer=optimizer, scheduler=scheduler, dataset=dataset
    )


def create_batched_cross_reference_mask(
    reference_idxs: list[list[int]], target_tensor: torch.Tensor
) -> torch.Tensor:
    device = target_tensor.device
    batch_size = target_tensor.shape[0]
    mask = torch.zeros_like(target_tensor, dtype=torch.bool)
    for i in range(batch_size):
        ref = torch.tensor(reference_idxs[i], device=device)
        mask[i] = torch.isin(target_tensor[i], ref)
    return mask


def calculate_zipf_distribution(dataset: ZipfDataset) -> list[int]:
    """Calculate Zipf distribution from dataset."""
    all_ids = [
        token_id for example in dataset for token_id in example["input_ids"].tolist()
    ]
    counts = Counter(all_ids)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sorted_token_idxs = [item[0] for item in sorted_counts]
    return sorted_token_idxs


def log_transform(
    noise: torch.Tensor, num_tokens: int, lower_bound: float
) -> torch.Tensor:
    """Transform noise value to token index based on Zipf distribution."""
    device = noise.device
    eps = 1e-10
    lower_bound_t = torch.tensor(lower_bound, device=device, dtype=noise.dtype)
    normalized = (
        torch.log10(torch.clamp(noise, min=eps)) - torch.log10(lower_bound_t)
    ) / (-torch.log10(lower_bound_t))
    idx = (normalized * (num_tokens - 1)).long()
    return torch.clamp(idx, min=0, max=num_tokens - 1)


def add_noise(
    x: torch.Tensor,
    time: torch.Tensor,
    vocab_size: int,
    sorted_token_idxs: list[int],
    zipf_lower_bound: float,
    blank_is_noise: bool,
) -> torch.Tensor:
    """Apply noise to input tensor based on the Zipf distribution and time."""
    device = x.device
    # Convert time to indices in the Zipf distribution
    zipf_indices = log_transform(
        time, num_tokens=len(sorted_token_idxs), lower_bound=zipf_lower_bound
    )

    # get the noise tokens for each timestep
    noise_idxs = [sorted_token_idxs[-i:] for i in zipf_indices]
    mask = create_batched_cross_reference_mask(noise_idxs, target_tensor=x)

    noised_input = x.clone()
    if blank_is_noise:
        noise = torch.full(
            noised_input.shape,
            fill_value=vocab_size - 1,
            dtype=torch.long,
            device=device,
        )
    else:
        noise = torch.randint(
            low=0,
            high=vocab_size,
            size=noised_input.shape,
            device=device,
            dtype=torch.long,
        )

    noised_input[mask] = noise[mask]
    return noised_input


def compute_loss(
    model: DiT,
    config: TrainConfig,
    input_ids: torch.Tensor,
    sorted_token_idxs: list[int],
) -> torch.Tensor:
    device = input_ids.device
    batch_size = input_ids.size(0)
    time = torch.rand(batch_size, device=device, dtype=torch.float32)

    noised_input = add_noise(
        x=input_ids,
        time=time.unsqueeze(1),  # make time shape (B,1) for model input
        vocab_size=model.config.vocab_size,
        sorted_token_idxs=sorted_token_idxs,
        zipf_lower_bound=config.zipf_lower_bound,
        blank_is_noise=config.blank_is_noise,
    )

    pred_logits = model(noised_input, time=time.unsqueeze(1))
    loss = torch.nn.functional.cross_entropy(
        pred_logits.view(-1, pred_logits.size(-1)).float(),
        input_ids.view(-1),
        reduction="mean",
    )
    return loss


def train_step(
    model: DiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainConfig,
    input_ids: torch.Tensor,
    sorted_token_idxs: list[int],
) -> dict:
    loss = compute_loss(
        model=model,
        config=config,
        input_ids=input_ids,
        sorted_token_idxs=sorted_token_idxs,
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
    optimizer.step()
    scheduler.step()

    return dict(loss=loss.item(), grad_norm=grad_norm.item())


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    scaled_logits = logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
        logits.size(0), -1
    )


def generate_text(
    model: DiT,
    batch_size: int,
    num_steps: int,
    tokenizer: AutoTokenizer,
    sorted_token_idxs: list[int],
    zipf_lower_bound: float,
    blank_is_noise: bool,
    temperature: float = 1.0,
) -> list[str]:
    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )
    sequence_length = model.config.sequence_length

    shape = (batch_size, sequence_length)
    noise = add_noise(
        x=torch.ones(shape, device=device, dtype=torch.long),
        time=torch.ones(batch_size, 1, device=device, dtype=torch.float32),
        vocab_size=model.config.vocab_size,
        sorted_token_idxs=sorted_token_idxs,
        zipf_lower_bound=zipf_lower_bound,
        blank_is_noise=blank_is_noise,
    )

    ts = torch.linspace(1.0, 0.0, steps=num_steps, device=device, dtype=torch.float32)
    ts = ts.view(-1, 1, 1).repeat(1, batch_size, 1)
    x0 = None

    for i in tqdm.tqdm(range(len(ts) - 1), desc="Generating text", colour="yellow"):
        logits = model(noise, time=ts[i])
        x0 = sample_from_logits(logits, temperature)
        noise = add_noise(
            x=x0,
            time=ts[i + 1],
            vocab_size=model.config.vocab_size,
            sorted_token_idxs=sorted_token_idxs,
            zipf_lower_bound=zipf_lower_bound,
            blank_is_noise=blank_is_noise,
        )

    return tokenizer.batch_decode(x0) if x0 is not None else [""]  # type: ignore


def train(config: TrainConfig):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize training state (model, optimizer, scheduler, dataset)
    state = init_train_state(config, device=device)
    sorted_token_idxs = calculate_zipf_distribution(state.dataset)

    dl = DataLoader(
        state.dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=config.num_workers,
    )

    step = 0
    pbar = tqdm.tqdm(
        total=config.max_steps, desc="Training", dynamic_ncols=True, colour="blue"
    )

    while step < config.max_steps:
        for batch in dl:
            if step >= config.max_steps:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            info = train_step(
                model=state.model,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                config=config,
                input_ids=input_ids,
                sorted_token_idxs=sorted_token_idxs,
            )

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=info["loss"], grad_norm=info["grad_norm"])

            # Test generation at intervals
            if step % config.test_every == 0:
                generated = generate_text(
                    model=state.model,
                    batch_size=config.batch_size,
                    num_steps=config.generate_kwargs.get("num_steps", 10),
                    tokenizer=state.dataset.tokenizer,
                    sorted_token_idxs=sorted_token_idxs,
                    zipf_lower_bound=config.zipf_lower_bound,
                    blank_is_noise=config.blank_is_noise,
                    temperature=config.generate_kwargs.get("temperature", 1.0),
                )
                print("\n[Generated Samples]")
                print("\n\n".join(generated))

    pbar.close()


def save_model(model: DiT, path: str) -> None:
    """Save the trained model."""
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    chunk_size = 128
    train_config = TrainConfig(
        lr=1e-4,
        max_steps=100,
        batch_size=4,
        dit_config=DiTConfig(
            vocab_size=65536,
            hidden_dim=16,
            num_heads=8,
            num_layers=1,
            sequence_length=chunk_size,
        ),
        chunk_size=chunk_size,
        blank_is_noise=False,
        test_every=5,
        zipf_lower_bound=1e-11,
        generate_kwargs=dict(num_steps=10, temperature=1.0),
        num_warmup_steps=10,
        num_workers=4,
    )

    train(train_config)
