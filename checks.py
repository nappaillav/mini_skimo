"""Small mathematical checks; incomplete exercises should fail explicitly."""
import argparse
import torch
from data import DotTrajectories
from losses import diagonal_kl
from model import sample_skill, SkiMo


def check_data():
    dataset = DotTrajectories(count=3)
    sample = dataset[0]
    assert sample["images"].shape == (21,1,28,28)
    assert sample["actions"].shape == (20,2)
    torch.testing.assert_close(dataset.positions[:,1:] - dataset.positions[:,:-1],
                               0.025 * dataset.actions)
    assert sample["images"].min() >= 0 and sample["images"].max() <= 1
    print("PASS: image/action alignment and transition dynamics")


def check_gaussian():
    mu = torch.zeros(2,4, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    torch.testing.assert_close(diagonal_kl(mu, logvar, mu, logvar), torch.zeros(2))
    torch.testing.assert_close(diagonal_kl(mu+1, logvar, mu, logvar), torch.full((2,), 2.))
    torch.testing.assert_close(sample_skill(mu+1, logvar, deterministic=True), torch.tanh(mu+1))
    z = sample_skill(mu, logvar)
    assert z.shape == mu.shape and (z.abs() <= 1).all()
    z.sum().backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    print("PASS: Gaussian KL and reparameterized sampling")


def check_target():
    model = SkiMo()
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    for online, target in zip(model.encoder.parameters(), model.target_encoder.parameters()):
        torch.testing.assert_close(online, target)
    before = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1)
    model.update_target(tau=0.2)
    for old, target in zip(before, model.target_encoder.parameters()):
        torch.testing.assert_close(target, old+0.2)
    print("PASS: target initialization, freezing, and EMA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exercise", choices=["data", "gaussian", "target", "all"])
    args = parser.parse_args()
    torch.manual_seed(0)
    checks = {"data": check_data, "gaussian": check_gaussian, "target": check_target}
    for check in checks.values() if args.exercise == "all" else [checks[args.exercise]]:
        check()
