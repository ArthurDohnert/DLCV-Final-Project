"""Caminho B com RAFT-Stereo pré-treinada (SceneFlow), ajustada ao US3D.

Antes, uma vez no PCAD:  bash scripts/pcad/get_raft.sh
Treino (PCAD):           sbatch scripts/slurm/train_raft.sbatch
Teste de overfit:        python -m src.train_raft --overfit 4 --epochs 20 --crop 256 384 --batch 2 --name raft_overfit4

Mesma avaliação do PSMNet (src/train_stereo.py): EPE, D1 e altura em metros, nos pixels
com disparidade entre -64 e +64 px, para os números serem comparáveis.
Checkpoints em runs/<name>/; rodar de novo com o mesmo --name continua de onde parou.
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import Track2Stereo, load_split
from src.train_stereo import evaluate

STOP = False
METRIC_MIN_DISP, METRIC_MAX_DISP = -64, 128   # mesmos pixels avaliados na PSMNet


def request_stop(*_):
    global STOP
    STOP = True
    print("Sinal recebido: salvando checkpoint e saindo no fim do passo atual.", flush=True)


def sequence_loss(preds, gt, mask, gamma=0.9):
    """L1 em todas as iterações, com peso maior nas últimas (como no RAFT-Stereo original)."""
    n = len(preds)
    g = gamma ** (15 / max(n - 1, 1))
    loss = 0.0
    for i, p in enumerate(preds):
        loss = loss + g ** (n - i - 1) * (p.float() - gt).abs()[mask].mean()
    return loss


def save_figure(model, ds, device, path, k=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    k = min(k, len(ds))
    fig, axes = plt.subplots(k, 3, figsize=(12, 3.0 * k), squeeze=False)
    with torch.no_grad():
        for r in range(k):
            left, right, disp, dmask, _, _ = ds[r]
            pred = model(left[None].to(device), right[None].to(device)).float()[0, 0].cpu().numpy()
            gt = np.where(dmask[0].numpy(), disp[0].numpy(), np.nan)
            lo, hi = (np.nanpercentile(gt, [1, 99]) if np.isfinite(gt).any() else (-1, 1))
            imgs = [left.numpy().transpose(1, 2, 0) / 255.0, gt, pred]
            for ax, im, t in zip(axes[r], imgs, ["esquerda", "disparidade real (px)", "disparidade prevista (px)"]):
                h = ax.imshow(im.clip(0, 1)) if im.ndim == 3 else ax.imshow(im, cmap="viridis", vmin=lo, vmax=hi)
                if im.ndim == 2:
                    plt.colorbar(h, ax=ax, fraction=0.046)
                ax.set_title(t, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--name", default="raft")
    ap.add_argument("--raft-dir", default=None, help="pasta do repositório oficial (padrão: third_party/RAFT-Stereo)")
    ap.add_argument("--weights", default="third_party/RAFT-Stereo/models/raftstereo-sceneflow.pth",
                    help="pesos pré-treinados; 'none' treina do zero")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 512], metavar=("ALTURA", "LARGURA"))
    ap.add_argument("--val-crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-iters", type=int, default=16)
    ap.add_argument("--valid-iters", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--fig-every", type=int, default=5)
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
    meta = dict(vars(args), model="raft", min_disp=METRIC_MIN_DISP, max_disp=METRIC_MAX_DISP)
    (out / "args.json").write_text(json.dumps(meta, indent=1))

    crop = tuple(args.crop)
    if args.overfit:
        pairs = load_split(args.splits, "train")[: args.overfit]
        train_ds = Track2Stereo(args.data, pairs * max(1, 100 // len(pairs)), crop=crop, train=False,
                                augment=False, normalize=False)
        val_ds = Track2Stereo(args.data, pairs, crop=crop, train=False, augment=False, normalize=False)
    else:
        train_ds = Track2Stereo(args.data, load_split(args.splits, "train"), crop=crop, train=True, normalize=False)
        vc = (args.val_crop, args.val_crop) if args.val_crop else None
        val_ds = Track2Stereo(args.data, load_split(args.splits, "val"), crop=vc, train=False, normalize=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=amp, drop_last=len(train_ds) > args.batch, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0 if os.name == "nt" else args.workers)

    s = c = 0.0
    ref = val_ds if args.overfit else train_ds
    for i in range(min(len(ref), 50)):
        _, _, _, _, agl, amask = ref[i]
        s += agl[amask].sum().item(); c += amask.sum().item()
    mean_agl = s / max(c, 1)

    from src.models.raft import RaftDisparity
    model = RaftDisparity(args.raft_dir, mixed_precision=amp, train_iters=args.train_iters,
                          valid_iters=args.valid_iters).to(device)
    if args.weights and args.weights.lower() != "none":
        model.load_pretrained(args.weights)
        print(f"Pesos pré-treinados: {args.weights}")
    model.train()
    model.net.freeze_bn()  # como no fine-tune original: BatchNorm congelada (lotes pequenos)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-8)
    total = args.epochs * max(len(train_dl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total + 100, pct_start=0.01,
                                                cycle_momentum=False, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best = 0, float("inf")
    print(f"Dispositivo: {device} | treino {len(train_ds)} pares, val {len(val_ds)} | {total} passos")

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Retomando da época {start_epoch} (melhor EPE até agora {best:.3f} px)")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "args": meta}, path)

    if start_epoch == 0 and not args.overfit:
        mt = evaluate(model, val_dl, device, amp, METRIC_MIN_DISP, METRIC_MAX_DISP, mean_agl)
        model.train(); model.net.freeze_bn()
        print(f"antes do ajuste (zero-shot) | val EPE {mt['epe']:.2f} px D1 {mt['d1']:.1%} | "
              f"altura MAE {mt['h_mae']:.2f} RMSE {mt['h_rmse']:.2f} m", flush=True)

    writer = SummaryWriter(out / "tb")
    for epoch in range(start_epoch, args.epochs):
        t0, run_loss, steps = time.time(), 0.0, 0
        for left, right, disp, dmask, _, _ in train_dl:
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            disp, dmask = disp.to(device), dmask.to(device)
            m = dmask & (disp.abs() < 512)
            if m.sum() == 0:
                continue
            preds = model(left, right)
            loss = sequence_loss(preds, disp, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item(); steps += 1
            if STOP:
                break
        mt = evaluate(model, val_dl, device, amp, METRIC_MIN_DISP, METRIC_MAX_DISP, mean_agl)
        model.train(); model.net.freeze_bn()
        if mt["epe"] < best:
            best = mt["epe"]; save(out / "best.pt", epoch)
        save(last, epoch)
        writer.add_scalar("train/seq_l1", run_loss / max(steps, 1), epoch)
        for k, v in mt.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"época {epoch:3d} | treino {run_loss / max(steps, 1):.3f} | val EPE {mt['epe']:.2f} px "
              f"(baseline {mt['base_epe']:.2f}) D1 {mt['d1']:.1%} | altura MAE {mt['h_mae']:.2f} "
              f"RMSE {mt['h_rmse']:.2f} m (baseline {mt['base_h_rmse']:.2f}) | {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % args.fig_every == 0 or epoch == args.epochs - 1:
            save_figure(model, val_ds, device, out / f"pred_ep{epoch:03d}.png")
            model.net.freeze_bn()
        if STOP:
            print("Parado por sinal; rode de novo com o mesmo --name para continuar."); break
    writer.close()
    print(f"Fim. Melhor EPE de validação: {best:.3f} px. Arquivos em {out}")


if __name__ == "__main__":
    main()
