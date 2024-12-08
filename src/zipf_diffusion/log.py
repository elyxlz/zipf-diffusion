import torch


def log_transform(
    noise: torch.Tensor, num_tokens: int, lower_bound: float
) -> torch.Tensor:
    """Transform noise value to token index based on Zipf distribution."""
    device = noise.device

    normalized = (
        torch.log10(torch.clamp(noise, min=1e-10))
        - torch.log10(torch.tensor(lower_bound))
    ) / (-torch.log10(torch.tensor(lower_bound)))
    idx = (normalized * (num_tokens - 1)).long()
    return torch.clamp(idx, min=0, max=num_tokens - 1)


#
for noise in [0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
    idx = log_transform(
        torch.tensor(noise).unsqueeze(0), num_tokens=13_000, lower_bound=1e-7
    ).item()
    percentage = (idx / 13_000) * 100
    print(f"noise: {noise:.3f} -> idx: {idx:5d} ({percentage:.1f}%)")
