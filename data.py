"""Deterministic offline image/action trajectories; no downloads or simulator."""
import torch
from torch.utils.data import Dataset


def render(positions, size=28, dot_sigma=0.08):
    """Normalized xy [...,2] -> grayscale Gaussian dots [...,1,size,size]."""
    if dot_sigma <= 0:
        raise ValueError("dot_sigma must be positive")
    axis = torch.linspace(0, 1, size, device=positions.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = ((xx - positions[..., 0, None, None]).square()
                + (yy - positions[..., 1, None, None]).square())
    return torch.exp(-distance / (2 * dot_sigma ** 2)).unsqueeze(-3)


class DotTrajectories(Dataset):
    def __init__(self, count=2048, horizon=10, num_skills=2, seed=0, dot_sigma=0.08):
        if dot_sigma <= 0:
            raise ValueError("dot_sigma must be positive")
        self.dot_sigma = dot_sigma
        generator = torch.Generator().manual_seed(seed)
        steps = horizon * num_skills
        # Each H-step block has a distinct roughly constant velocity.
        commands = torch.rand(count, num_skills, 2, generator=generator) * 2 - 1
        commands = commands.repeat_interleave(horizon, dim=1)
        positions = torch.empty(count, steps + 1, 2)
        positions[:, 0] = 0.2 + 0.6 * torch.rand(count, 2, generator=generator)
        for t in range(steps):
            positions[:, t + 1] = (positions[:, t] + 0.025 * commands[:, t]).clamp(0.08, 0.92)
        # Effective commands account for clipping and exactly explain transitions.
        self.actions = (positions[:, 1:] - positions[:, :-1]) / 0.025
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        return {"images": render(self.positions[index], dot_sigma=self.dot_sigma), "actions": self.actions[index]}
