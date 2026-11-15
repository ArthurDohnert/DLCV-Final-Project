"""Compara o Caminho A (1 imagem) e o Caminho B (2 imagens) de forma justa.

Os dois são avaliados nas MESMAS imagens inteiras e nos MESMOS pixels: onde há verdade
de altura e de disparidade (dentro do intervalo de disparidade do modelo B).

Conversão de B para metros: reta altura = a*disparidade + b ajustada em cada par na verdade
de solo (substitui a calibração das câmeras, que não vem nos arquivos da Track 2). Para não
favorecer B, A também é reportado com uma reta ajustada por par sobre a própria previsão
("A + ajuste"), um benefício maior do que o que B recebe.

Uso (notebook):
  .venv\\Scripts\\python -m src.evaluate --height runs\\unet_r34_local\\best.pt --stereo runs\\psmnet_local\\best.pt
Saídas em outputs/eval/: metrics.json, por_par.csv, figuras e previsões de alguns pares (.npz).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import Track2Stereo, load_split


def predict_disp(model, left, right, amp, strip=256, overlap=32):
    """PSMNet em faixas horizontais de largura total (as linhas epipolares são horizontais)."""
    h = left.shape[-2]
    out = torch.zeros(1, 1, h, left.shape[-1])
    y = 0
    while y < h:
        y0, y1 = max(0, y - overlap), min(h, y + strip + overlap)
        with torch.autocast(left.device.type, dtype=torch.float16, enabled=amp):
            d = model(left[..., y0:y1, :], right[..., y0:y1, :]).float().cpu()
        n = min(strip, h - y)
        out[..., y:y + n, :] = d[..., y - y0:y - y0 + n, :]
        y += strip
    return out[0, 0]


def fit_line(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    if x.size < 100 or x.std() < 1e-6:
        return None
    a = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return a, y.mean() - a * x.mean()


class Acc:
    def __init__(self):
        self.abs = self.sq = self.ok1 = self.n = 0.0

    def add(self, err):
        self.abs += np.abs(err).sum(); self.sq += (err ** 2).sum()
        self.ok1 += (np.abs(err) < 1.0).sum(); self.n += err.size

    def result(self):
        n = max(self.n, 1)
        return {"mae_m": float(self.abs / n), "rmse_m": float(math.sqrt(self.sq / n)), "erro_menor_1m": float(self.ok1 / n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", required=True, help="checkpoint do Caminho A (best.pt)")
    ap.add_argument("--stereo", required=True, help="checkpoint do Caminho B (best.pt)")
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--save-preds", type=int, default=5, help="quantos pares salvar em .npz e nas figuras")
    args = ap.parse_args()

    import segmentation_models_pytorch as smp
    from src.models.psmnet import PSMNetBasic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(args.out); (out / "preds").mkdir(parents=True, exist_ok=True)

    ck_a = torch.load(args.height, map_location=device, weights_only=False)
    model_a = smp.Unet(ck_a["args"]["encoder"], encoder_weights=None, in_channels=3, classes=1).to(device)
    model_a.load_state_dict(ck_a["model"]); model_a.eval()
    ck_b = torch.load(args.stereo, map_location=device, weights_only=False)
    min_d, max_d = ck_b["args"]["min_disp"], ck_b["args"]["max_disp"]
    model_b = PSMNetBasic(max_d, min_d).to(device)
    model_b.load_state_dict(ck_b["model"]); model_b.eval()
    print(f"A: {args.height} (época {ck_a['epoch']}) | B: {args.stereo} (época {ck_b['epoch']})")

    pairs = load_split(args.splits, args.split)
    ds = Track2Stereo(args.data, pairs, crop=None, train=False, augment=False)
    # baseline: altura média do conjunto de TREINO (não do conjunto avaliado)
    tr = Track2Stereo(args.data, load_split(args.splits, "train")[:30], crop=None, train=False, augment=False)
    tot = cnt = 0.0
    for j in range(len(tr)):
        _, _, _, _, a_, m_ = tr[j]
        tot += a_[m_].sum().item(); cnt += m_.sum().item()
    mean_agl = tot / max(cnt, 1)
    acc = {k: Acc() for k in ["media", "A", "A_ajuste", "B"]}
    epe = d1 = nd = 0.0
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            left, right, disp, dmask, agl, amask = ds[i]
            disp, agl = disp[0].numpy(), agl[0].numpy()
            m = (dmask[0] & amask[0]).numpy() & (disp >= min_d) & (disp < min_d + max_d)
            if m.sum() < 100:
                continue
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                h_a = model_a(left[None].to(device)).float()[0, 0].cpu().numpy()
            d_b = predict_disp(model_b, left[None].to(device), right[None].to(device), amp).numpy()
            cal = fit_line(disp[m], agl[m])          # geometria do par (verdade disp -> altura)
            fa = fit_line(h_a[m], agl[m])            # benefício extra dado a A
            if cal is None or fa is None:
                continue
            h_b = cal[0] * d_b + cal[1]
            h_a2 = fa[0] * h_a + fa[1]
            gt = agl[m]
            row = {"par": pairs[i]}
            for k, pred in [("media", np.full_like(gt, mean_agl)), ("A", h_a[m]), ("A_ajuste", h_a2[m]), ("B", h_b[m])]:
                err = pred - gt
                acc[k].add(err)
                row[f"{k}_rmse"] = float(np.sqrt((err ** 2).mean()))
            e = np.abs(d_b[m] - disp[m]); epe += e.sum(); d1 += (e > 3).sum(); nd += e.size
            row["B_epe_px"] = float(e.mean()); row["m_por_px"] = float(abs(cal[0]))
            rows.append(row)
            if i < args.save_preds:
                np.savez_compressed(out / "preds" / f"{pairs[i]}.npz", gt=np.where(amask[0].numpy(), agl, np.nan).astype(np.float16),
                                    a=h_a.astype(np.float16), b=h_b.astype(np.float16), mask=m)
            print(f"{i + 1}/{len(ds)} {pairs[i]} | RMSE A {row['A_rmse']:.2f} m, B {row['B_rmse']:.2f} m", flush=True)

    res = {k: v.result() for k, v in acc.items()}
    res["B_disparidade"] = {"epe_px": float(epe / max(nd, 1)), "d1_3px": float(d1 / max(nd, 1))}
    res["pares_avaliados"] = len(rows)
    res["pixels_avaliados"] = int(acc["A"].n)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(out / "por_par.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\nResultado no split", args.split, f"({len(rows)} pares, mesmos pixels para todos):")
    names = {"media": "Baseline (altura média)", "A": "A: 1 imagem (U-Net)", "A_ajuste": "A + ajuste por par",
             "B": "B: 2 imagens (PSMNet)"}
    print(f"{'método':28s} {'MAE (m)':>8s} {'RMSE (m)':>9s} {'erro<1m':>8s}")
    for k, name in names.items():
        r = res[k]; print(f"{name:28s} {r['mae_m']:8.2f} {r['rmse_m']:9.2f} {r['erro_menor_1m']:8.1%}")
    print(f"B em disparidade: EPE {res['B_disparidade']['epe_px']:.2f} px, D1 {res['B_disparidade']['d1_3px']:.1%}")

    # Geometria do par: quantos metros de altura cada pixel de disparidade representa.
    # Par com ângulo pequeno entre as duas vistas = muitos metros por pixel = estéreo pouco preciso.
    print("\nPor geometria do par (média dos RMSE por par):")
    res["por_geometria"] = {}
    for lo, hi, nome in [(0, 1, "< 1 m/px (boa)"), (1, 2, "1-2 m/px"), (2, 1e9, "> 2 m/px (ruim)")]:
        sel = [r for r in rows if lo <= r["m_por_px"] < hi]
        if not sel:
            continue
        ra = float(np.mean([r["A_rmse"] for r in sel])); rb = float(np.mean([r["B_rmse"] for r in sel]))
        vence = sum(r["B_rmse"] < r["A_rmse"] for r in sel)
        res["por_geometria"][nome] = {"pares": len(sel), "A_rmse": ra, "B_rmse": rb, "B_vence": vence}
        print(f"  {nome:16s} {len(sel):3d} pares | A {ra:.2f} m | B {rb:.2f} m | B vence em {vence}")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))

    save_figure(out, pairs[: args.save_preds], ds)


def save_figure(out, pairs, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    files = [out / "preds" / f"{p}.npz" for p in pairs if (out / "preds" / f"{p}.npz").exists()][:3]
    if not files:
        return
    fig, axes = plt.subplots(len(files), 4, figsize=(15, 3.6 * len(files)), squeeze=False)
    for row, f in zip(axes, files):
        z = np.load(f); i = ds.pairs.index(f.stem)
        rgb = np.clip(ds[i][0].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        gt = z["gt"].astype(np.float32)
        vmax = np.nanpercentile(gt, 99.5)
        for ax, im, t in zip(row, [rgb, gt, z["a"].astype(np.float32), z["b"].astype(np.float32)],
                             ["imagem esquerda", "altura real (LiDAR)", "A: 1 imagem", "B: 2 imagens"]):
            h = ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap="viridis", vmin=0, vmax=vmax)
            if im.ndim == 2:
                plt.colorbar(h, ax=ax, fraction=0.046)
            ax.set_title(f"{t}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "comparacao_A_B.png", dpi=90); plt.close(fig)
    print(f"Figura: {out / 'comparacao_A_B.png'}")


if __name__ == "__main__":
    main()
