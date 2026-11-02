import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from data import DotTrajectories
from model import SkiMo
from losses import pretrain_loss


def train_step(model, batch, optimizer):
    """Student TODO: forward, loss, zero_grad, backward, step, target EMA.

    Return detached Python float components plus total. Batch is on the device.
    Exclude target_encoder from optimizer updates (its requires_grad is False).
    """
    # raise NotImplementedError("Exercise 5: joint optimization step")
    outputs = model(batch["images"], batch["actions"]) # skiMo.forward

    # loss
    loss, components = pretrain_loss(
        outputs,
        batch["images"],
        batch["actions"],
        horizon=model.horizon,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Slowly update the target using the newly updated online encoder.
    model.update_target()

    metrics = {
        name: value.item()
        for name, value in components.items()
    }
    metrics["total"] = loss.item()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--preview-data", action="store_true")
    parser.add_argument("--dot-sigma", type=float, default=0.08,
                        help="Gaussian dot width in normalized image coordinates (default: 0.08)")
    args = parser.parse_args()
    torch.manual_seed(0)
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    dataset = DotTrajectories(dot_sigma=args.dot_sigma)
    if args.preview_data:
        save_image(dataset[0]["images"], output / "trajectory.png", nrow=11)
        print(f"Saved {output / 'trajectory.png'}; 21 frames, two skills of 10 actions.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.overfit_one_batch:
        dataset = Subset(dataset, range(args.batch_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validation = DotTrajectories(count=8, seed=123, dot_sigma=args.dot_sigma)
    preview = next(iter(DataLoader(validation, batch_size=8)))
    if args.overfit_one_batch:
        preview = next(iter(loader))
    preview = {k: v.to(device) for k, v in preview.items()}
    model = SkiMo().to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        totals, count = {}, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = train_step(model, batch, optimizer)
            size = len(batch["images"])
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.) + value * size
            count += size
        print(f"epoch={epoch+1} " + " ".join(f"{k}={v/count:.5f}" for k,v in totals.items()))
        model.eval()
        with torch.no_grad():
            predictions = model(preview["images"], preview["actions"], deterministic=True)
            actual = preview["images"][:8, -1]
            imagined = model.decoder(predictions["predicted_states"][:8, -1])
            save_image(torch.cat([actual, imagined]), output / f"epoch_{epoch+1:03d}.png", nrow=len(actual))
    torch.save(model.state_dict(), output / "model.pt")


if __name__ == "__main__":
    main()
