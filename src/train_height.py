"""Caminho A: treina U-Net (encoder pré-treinado) para prever altura a partir de uma imagem.

Teste de overfit no notebook (deve levar o erro de treino para perto de zero):
  .venv\\Scripts\\python -m src.train_height --overfit 10 --epochs 40 --crop 256 --batch 4 --name overfit10

Treino normal (PCAD):
  python -m src.train_height --epochs 60 --crop 512 --batch 8 --name unet_r34

Checkpoints e logs em runs/<name>/. Rodar de novo com o mesmo --name continua de onde parou.
"""
import argparse
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import Track2Height, load_split

STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    print("Sinal recebido: salvando checkpoint e saindo no fim do passo atual.", flush=True)


def masked_l1(pred, target, mask):
    m = mask.float()
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device, amp, baseline):
    model.eval()
    abs_err = sq_err = base_abs = base_sq = n = 0.0
    for img, agl, mask in loader:
        img, agl, mask = img.to(device), agl.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(img).float()
        m = mask
        diff = (pred - agl)[m]
        bdiff = (baseline - agl)[m]
        abs_err += diff.abs().sum().item(); sq_err += (diff ** 2).sum().item()
        base_abs += bdiff.abs().sum().item(); base_sq += (bdiff ** 2).sum().item()
        n += m.sum().item()
    model.train()
    n = max(n, 1)
    return {"mae": abs_err / n, "rmse": math.sqrt(sq_err / n),
            "base_mae": base_abs / n, "base_rmse": math.sqrt(base_sq / n)}


def save_figure(model, ds, device, amp, path, k=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    model.eval()
    k = min(k, len(ds))
    fig, axes = plt.subplots(k, 3, figsize=(10, 3.3 * k), squeeze=False)
    with torch.no_grad():
        for r in range(k):
            img, agl, mask = ds[r]
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(img[None].to(device)).float()[0, 0].cpu().numpy()
            rgb = np.clip(img.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
            gt = np.where(mask[0].numpy(), agl[0].numpy(), np.nan)
            vmax = np.nanpercentile(gt, 99.5) if np.isfinite(gt).any() else 1
            for ax, im, t in zip(axes[r], [rgb, gt, pred], ["imagem", "altura real (m)", "altura prevista (m)"]):
                h = ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap="viridis", vmin=0, vmax=vmax)
                if im.ndim == 2:
                    plt.colorbar(h, ax=ax, fraction=0.046)
                ax.set_title(t, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--name", default="unet_r34")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true", help="ablação: encoder sem pesos ImageNet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overfit", type=int, default=0, help="usa só N pares, sem augmentation, treino = validação")
    ap.add_argument("--fig-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if hasattr(signal, "SIGUSR1"):  # PCAD: o SLURM avisa antes do limite de 24h
        signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    if args.overfit:
        pairs = load_split(args.splits, "train")[: args.overfit]
        reps = max(1, 100 // len(pairs))  # ~100 amostras por época, para ter passos suficientes
        train_ds = Track2Height(args.data, pairs * reps, crop=args.crop, train=False, augment=False)
        val_ds = Track2Height(args.data, pairs, crop=args.crop, train=False, augment=False)
    else:
        train_ds = Track2Height(args.data, load_split(args.splits, "train"), crop=args.crop, train=True)
        val_ds = Track2Height(args.data, load_split(args.splits, "val"), crop=None, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=amp, drop_last=len(train_ds) > args.batch, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1 if not args.overfit else args.batch, num_workers=args.workers)

    # baseline bobo: altura média do conjunto de treino
    s = c = 0.0
    for i in range(min(len(val_ds if args.overfit else train_ds), 50)):
        _, agl, mask = (val_ds if args.overfit else train_ds)[i]
        s += agl[mask].sum().item(); c += mask.sum().item()
    baseline = s / max(c, 1)
    print(f"Dispositivo: {device} | treino {len(train_ds)} pares, val {len(val_ds)} | baseline (média) {baseline:.2f} m")

    model = smp.Unet(args.encoder, encoder_weights=None if args.no_pretrained else "imagenet",
                     in_channels=3, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best = 0, float("inf")

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Retomando da época {start_epoch} (melhor RMSE até agora {best:.3f} m)")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "args": vars(args)}, path)

    writer = SummaryWriter(out / "tb")
    for epoch in range(start_epoch, args.epochs):
        t0, run_loss, steps = time.time(), 0.0, 0
        for img, agl, mask in train_dl:
            img, agl, mask = img.to(device, non_blocking=True), agl.to(device), mask.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(img)
            loss = masked_l1(pred.float(), agl, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item(); steps += 1
            if STOP:
                break
        sched.step()
        metrics = evaluate(model, val_dl, device, amp, baseline)
        if metrics["rmse"] < best:
            best = metrics["rmse"]; save(out / "best.pt", epoch)
        save(last, epoch)
        writer.add_scalar("train/l1", run_loss / max(steps, 1), epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"época {epoch:3d} | treino L1 {run_loss / max(steps, 1):.3f} m | val MAE {metrics['mae']:.3f} "
              f"RMSE {metrics['rmse']:.3f} m (baseline RMSE {metrics['base_rmse']:.3f}) | {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % args.fig_every == 0 or epoch == args.epochs - 1:
            save_figure(model, val_ds, device, amp, out / f"pred_ep{epoch:03d}.png")
        if STOP:
            print("Parado por sinal; rode de novo com o mesmo --name para continuar."); break
    writer.close()
    print(f"Fim. Melhor RMSE de validação: {best:.3f} m. Arquivos em {out}")


if __name__ == "__main__":
    main()
