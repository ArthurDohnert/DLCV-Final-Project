"""Caminho B: treina PSMNet para estimar disparidade a partir de um par estéreo de satélite.

Teste de overfit no notebook:
  .venv\\Scripts\\python -m src.train_stereo --overfit 4 --epochs 40 --crop 256 384 --batch 2 --name psm_overfit4

Treino normal:
  python -m src.train_stereo --epochs 60 --crop 256 512 --batch 4 --name psmnet

Além do erro em disparidade (EPE e D1-3px, as métricas de estéreo), converte a disparidade
em altura para comparar com o Caminho A em metros. A conversão usa, para cada par, a reta
altura = a * disparidade + b ajustada na verdade de solo (papel da calibração das câmeras).
Checkpoints e logs em runs/<name>/. Rodar de novo com o mesmo --name continua de onde parou.
"""
import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import Track2Stereo, load_split

STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    print("Sinal recebido: salvando checkpoint e saindo no fim do passo atual.", flush=True)


def in_range(disp, mask, min_disp, max_disp):
    return mask & (disp >= min_disp) & (disp < min_disp + max_disp)


def height_fit(gt_disp, agl, m):
    """Reta altura = a*disp + b ajustada na verdade de solo de UM par."""
    x, y = gt_disp[m].double(), agl[m].double()
    if x.numel() < 100 or x.std() < 1e-3:
        return None
    a = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return a.item(), (y.mean() - a * x.mean()).item()


@torch.no_grad()
def evaluate(model, loader, device, amp, min_disp, max_disp, mean_agl):
    model.eval()
    s = dict(epe=0.0, d1=0.0, n=0.0, base_epe=0.0, h_abs=0.0, h_sq=0.0, hb_sq=0.0, hn=0.0)
    for left, right, disp, dmask, agl, amask in loader:
        left, right = left.to(device), right.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(left, right).float().cpu()
        for i in range(pred.shape[0]):
            m = in_range(disp[i], dmask[i], min_disp, max_disp)
            err = (pred[i] - disp[i])[m].abs()
            s["epe"] += err.sum().item(); s["d1"] += (err > 3).sum().item(); s["n"] += m.sum().item()
            s["base_epe"] += disp[i][m].abs().sum().item()  # baseline: disparidade zero
            both = dmask[i] & amask[i]
            fit = height_fit(disp[i], agl[i], both)
            if fit is not None:
                h_pred = fit[0] * pred[i] + fit[1]
                hm = amask[i] & dmask[i]
                dh = (h_pred - agl[i])[hm]
                s["h_abs"] += dh.abs().sum().item(); s["h_sq"] += (dh ** 2).sum().item()
                s["hb_sq"] += ((mean_agl - agl[i][hm]) ** 2).sum().item(); s["hn"] += hm.sum().item()
    model.train()
    n, hn = max(s["n"], 1), max(s["hn"], 1)
    return {"epe": s["epe"] / n, "d1": s["d1"] / n, "base_epe": s["base_epe"] / n,
            "h_mae": s["h_abs"] / hn, "h_rmse": math.sqrt(s["h_sq"] / hn), "base_h_rmse": math.sqrt(s["hb_sq"] / hn)}


def save_figure(model, ds, device, amp, path, k=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    model.eval()
    k = min(k, len(ds))
    fig, axes = plt.subplots(k, 3, figsize=(12, 3.0 * k), squeeze=False)
    with torch.no_grad():
        for r in range(k):
            left, right, disp, dmask, _, _ = ds[r]
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(left[None].to(device), right[None].to(device)).float()[0, 0].cpu().numpy()
            rgb = np.clip(left.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
            gt = np.where(dmask[0].numpy(), disp[0].numpy(), np.nan)
            lo, hi = (np.nanpercentile(gt, [1, 99]) if np.isfinite(gt).any() else (-1, 1))
            for ax, im, t in zip(axes[r], [rgb, gt, pred], ["esquerda", "disparidade real (px)", "disparidade prevista (px)"]):
                h = ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap="viridis", vmin=lo, vmax=hi)
                if im.ndim == 2:
                    plt.colorbar(h, ax=ax, fraction=0.046)
                ax.set_title(t, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    model.train()


def main():
    # No Windows, cada worker do DataLoader reimporta este módulo: nada pesado no topo.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--name", default="psmnet")
    ap.add_argument("--max-disp", type=int, default=128, help="quantidade de disparidades candidatas")
    ap.add_argument("--min-disp", type=int, default=-64, help="menor disparidade candidata")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 512], metavar=("ALTURA", "LARGURA"))
    ap.add_argument("--val-crop", type=int, default=512, help="lado do recorte central na validação; 0 = imagem inteira")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overfit", type=int, default=0, help="usa só N pares, sem augmentation, treino = validação")
    ap.add_argument("--fig-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    crop = tuple(args.crop)
    if args.overfit:
        pairs = load_split(args.splits, "train")[: args.overfit]
        reps = max(1, 100 // len(pairs))
        train_ds = Track2Stereo(args.data, pairs * reps, crop=crop, train=False, augment=False)
        val_ds = Track2Stereo(args.data, pairs, crop=crop, train=False, augment=False)
    else:
        train_ds = Track2Stereo(args.data, load_split(args.splits, "train"), crop=crop, train=True)
        vc = (args.val_crop, args.val_crop) if args.val_crop else None
        val_ds = Track2Stereo(args.data, load_split(args.splits, "val"), crop=vc, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=amp, drop_last=len(train_ds) > args.batch, persistent_workers=args.workers > 0)
    val_workers = 0 if os.name == "nt" else args.workers
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=val_workers)

    s = c = 0.0
    for i in range(min(len(val_ds) if args.overfit else len(train_ds), 50)):
        _, _, _, _, agl, amask = (val_ds if args.overfit else train_ds)[i]
        s += agl[amask].sum().item(); c += amask.sum().item()
    mean_agl = s / max(c, 1)
    print(f"Dispositivo: {device} | treino {len(train_ds)} pares, val {len(val_ds)} | "
          f"disparidades candidatas [{args.min_disp}, {args.min_disp + args.max_disp})")

    from src.models.psmnet import PSMNetBasic
    model = PSMNetBasic(args.max_disp, args.min_disp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best = 0, float("inf")

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Retomando da época {start_epoch} (melhor EPE até agora {best:.3f} px)")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "args": vars(args)}, path)

    writer = SummaryWriter(out / "tb")
    for epoch in range(start_epoch, args.epochs):
        t0, run_loss, steps = time.time(), 0.0, 0
        for left, right, disp, dmask, _, _ in train_dl:
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            disp, dmask = disp.to(device), dmask.to(device)
            m = in_range(disp, dmask, args.min_disp, args.max_disp)
            if m.sum() == 0:
                continue
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(left, right)
            loss = F.smooth_l1_loss(pred.float()[m], disp[m])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item(); steps += 1
            if STOP:
                break
        sched.step()
        mt = evaluate(model, val_dl, device, amp, args.min_disp, args.max_disp, mean_agl)
        if mt["epe"] < best:
            best = mt["epe"]; save(out / "best.pt", epoch)
        save(last, epoch)
        writer.add_scalar("train/smooth_l1", run_loss / max(steps, 1), epoch)
        for k, v in mt.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"época {epoch:3d} | treino {run_loss / max(steps, 1):.3f} | val EPE {mt['epe']:.2f} px "
              f"(baseline {mt['base_epe']:.2f}) D1 {mt['d1']:.1%} | altura MAE {mt['h_mae']:.2f} "
              f"RMSE {mt['h_rmse']:.2f} m (baseline {mt['base_h_rmse']:.2f}) | {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % args.fig_every == 0 or epoch == args.epochs - 1:
            save_figure(model, val_ds, device, amp, out / f"pred_ep{epoch:03d}.png")
        if STOP:
            print("Parado por sinal; rode de novo com o mesmo --name para continuar."); break
    writer.close()
    print(f"Fim. Melhor EPE de validação: {best:.3f} px. Arquivos em {out}")


if __name__ == "__main__":
    main()
