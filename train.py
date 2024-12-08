from zipf_diffusion.trainer import TrainConfig, train
from zipf_diffusion.dit import DiTConfig


if __name__ == "__main__":
    chunk_size = 512
    train_config = TrainConfig(
        lr=3e-4,
        max_steps=10_000,
        batch_size=64,
        dit_config=DiTConfig(
            vocab_size=65536,
            hidden_dim=512,
            num_heads=8,
            num_layers=8,
            sequence_length=chunk_size,
        ),
        chunk_size=chunk_size,
        blank_is_noise=False,
        test_every=250,
        zipf_lower_bound=1e-11,
        generate_kwargs=dict(num_steps=30, temperature=1.0),
        num_warmup_steps=250,
        num_workers=12,
    )

    train(train_config)
