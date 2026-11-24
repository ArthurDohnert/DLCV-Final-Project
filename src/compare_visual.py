"""Comparação visual A x B x verdade em pares do teste escolhidos por regra fixa (sem escolher a dedo).

Regra de escolha, a partir do por_par.csv da avaliação com a RAFT (outputs/test_raft).
Só entram cenas com relevo (desvio-padrão da altura real >= 2 m): em cenas planas a reta
de calibração por par torna B trivialmente perfeito, o que não mostra nada sobre o estéreo.
  tipico   - par de geometria boa com o RMSE da fusão final mais próximo da mediana
  ganho    - par de geometria boa em que a fusão mais reduz o RMSE em relação a A
  perda    - par de geometria boa em que a fusão mais aumenta o RMSE em relação a A
  ruim     - par de geometria ruim (> 2 m/px) com o RMSE de B mais próximo da mediana
Para cada par: imagem, verdade, A, SGM, RAFT e fusão; mapas de erro; e um perfil de altura
ao longo da linha que passa pelo ponto mais alto da cena (corte transversal dos prédios).

Uso (PCAD):
  python -m src.compare_visual --height runs/unet_r34/best.pt --stereo runs/raft/best.pt \
      --csv outputs/test_raft/por_par.csv --data $HOME/data/dfc2019/track2 --out outputs/visual
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, Track2Stereo
from src.evaluate import GEOMETRIAS, combine, fit_line

TXT = {
    "pt": {"rgb": "imagem esquerda", "gt": "altura real (LiDAR)", "A": "A: 1 imagem (U-Net)",
           "sgm": "B: SGM clássico", "raft": "B: RAFT-Stereo", "ab": "A+B: fusão final",
           "err": "erro absoluto", "perfil": "perfil de altura na linha marcada", "h": "altura (m)",
           "x": "coluna (px, ~0,3 m/px)", "tipico": "par típico", "ganho": "fusão ajuda mais",
           "perda": "fusão atrapalha mais", "ruim": "geometria ruim", "rgb_r": "imagem direita", "mask": "pixels avaliados (branco)"},
    "en": {"rgb": "left image", "gt": "ground truth (LiDAR)", "A": "A: single image (U-Net)",
           "sgm": "B: classical SGM", "raft": "B: RAFT-Stereo", "ab": "A+B: final fusion",
           "err": "absolute error", "perfil": "height profile along the marked row", "h": "height (m)",
           "x": "column (px, ~0.3 m/px)", "tipico": "typical pair", "ganho": "fusion helps most",
           "perda": "fusion hurts most", "ruim": "poor geometry", "rgb_r": "right image", "mask": "evaluated pixels (white)"},
}


def _im(ax, img, **kw):
    """imshow com reamostragem depois da conversão em cores: ao reduzir 1024 px para ~250 px,
    a reamostragem dos dados espalhava os NaN esparsos (pixels sem verdade) pelo painel todo."""
    return ax.imshow(img, interpolation="antialiased", interpolation_stage="rgba", **kw)


def relevo_std(data_dir, par):
    import tifffile
    a = tifffile.imread(Path(data_dir) / f"{par}LEFT_AGL.tif").astype(np.float32)
    a = a[np.isfinite(a) & (a > -998)]
    return float(a.std()) if a.size else 0.0


def primeiro_com_relevo(cands, data_dir, min_std):
    for r in cands:
        sd = relevo_std(data_dir, r["par"])
        if sd >= min_std:
            r["altura_std"] = sd
            return r
    return None


def escolher(csv_path, data_dir, min_std=2.0):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        for k, v in r.items():
            if k.endswith("_rmse") or k in ("m_por_px", "B_epe_px"):
                r[k] = float(v)
    boa = [r for r in rows if r["geometria"] == GEOMETRIAS[0][2]]
    ruim = [r for r in rows if r["geometria"] == GEOMETRIAS[2][2]]
    sel = {}
    if boa:
        med = np.median([r["AB_relevo_geo_rmse"] for r in boa])
        ganho = lambda r: r["A_rmse"] - r["AB_relevo_geo_rmse"]
        sel["tipico"] = primeiro_com_relevo(sorted(boa, key=lambda r: abs(r["AB_relevo_geo_rmse"] - med)), data_dir, min_std)
        sel["ganho"] = primeiro_com_relevo(sorted(boa, key=ganho, reverse=True), data_dir, min_std)
        sel["perda"] = primeiro_com_relevo(sorted(boa, key=ganho), data_dir, min_std)
    if ruim:
        med = np.median([r["B_rmse"] for r in ruim])
        sel["ruim"] = primeiro_com_relevo(sorted(ruim, key=lambda r: abs(r["B_rmse"] - med)), data_dir, min_std)
    return {k: v for k, v in sel.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", required=True, help="best.pt do Caminho A")
    ap.add_argument("--stereo", required=True, help="best.pt da RAFT-Stereo ajustada")
    ap.add_argument("--raft-dir", default=None)
    ap.add_argument("--csv", default="outputs/test_raft/por_par.csv")
    ap.add_argument("--pairs", nargs="*", default=None, help="pares específicos (ignora a regra de escolha)")
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--out", default="outputs/visual")
    ap.add_argument("--min-std", type=float, default=2.0, help="desvio-padrão mínimo da altura real (m)")
    ap.add_argument("--lang", default="pt", choices=["pt", "en"])
    args = ap.parse_args()
    T = TXT[args.lang]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import segmentation_models_pytorch as smp
    from src.models.raft import RaftDisparity
    from src.models.sgm import sgm_disparity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ck_a = torch.load(args.height, map_location=device, weights_only=False)
    model_a = smp.Unet(ck_a["args"]["encoder"], encoder_weights=None, in_channels=3, classes=1).to(device)
    model_a.load_state_dict(ck_a["model"]); model_a.eval()
    ck_b = torch.load(args.stereo, map_location=device, weights_only=False)
    iters = ck_b.get("args", {}).get("valid_iters", 32)
    model_b = RaftDisparity(args.raft_dir, mixed_precision=amp, valid_iters=iters).to(device)
    model_b.load_state_dict(ck_b["model"]); model_b.eval()
    min_d, max_d = -64, 128

    if args.pairs:
        sel = {f"par{i + 1}": {"par": p} for i, p in enumerate(args.pairs)}
    else:
        sel = escolher(args.csv, args.data, args.min_std)
    names = [s["par"] for s in sel.values()]
    ds = Track2Stereo(args.data, names, crop=None, train=False, augment=False)
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    resumo = {}

    for i, (rotulo, info) in enumerate(sel.items()):
        par = info["par"]
        left, right, disp, dmask, agl, amask = ds[i]
        disp, agl = disp[0].numpy(), agl[0].numpy()
        av = amask[0].numpy()
        m = (dmask[0].numpy() & av) & (disp >= min_d) & (disp < min_d + max_d)
        raw_l = (left[None] * std_t + mean_t) * 255
        raw_r = (right[None] * std_t + mean_t) * 255
        with torch.no_grad():
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                h_a = model_a(left[None].to(device)).float()[0, 0].cpu().numpy()
            d_r = model_b(raw_l.to(device), raw_r.to(device)).float()[0, 0].cpu().numpy()
        u8 = lambda x: x[0].clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
        d_s, _ = sgm_disparity(u8(raw_l), u8(raw_r), min_d, max_d)
        cal = fit_line(disp[m], agl[m])
        if cal is None:
            print(f"{par}: poucos pixels válidos, pulando"); continue
        m_px = float(abs(cal[0]))
        dv = dmask[0].numpy() & av
        fora = float(((disp < min_d) | (disp >= min_d + max_d))[dv].mean()) if dv.any() else 0.0
        dmax = float(np.abs(disp[dv]).max()) if dv.any() else 0.0
        h_r = cal[0] * d_r + cal[1]
        h_s = cal[0] * d_s + cal[1]
        h_ab = combine(h_a, h_r, m_px)["AB_relevo_geo"]

        rgb = np.clip(left.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        gt = np.where(av, agl, np.nan)
        vmax = float(np.nanpercentile(gt, 99.5))
        # linha do perfil: a que passa pelo pixel mais alto da verdade (ignorando 0,5% extremo)
        alto = np.where(np.isfinite(gt), gt, -np.inf)
        alto = np.where(alto <= np.nanpercentile(gt, 99.9), alto, -np.inf)
        lin = int(np.unravel_index(np.argmax(alto), alto.shape)[0])

        preds = [("A", h_a), ("sgm", h_s), ("raft", h_r), ("ab", h_ab)]
        rmse = {k: float(np.sqrt(((p[m] - agl[m]) ** 2).mean())) for k, p in preds}
        mae = {k: float(np.abs(p[m] - agl[m]).mean()) for k, p in preds}
        resumo[rotulo] = {"par": par, "m_por_px": m_px, "fora_intervalo": fora, "disp_max_abs_px": dmax, "linha_perfil": lin, "rmse_m": rmse, "mae_m": mae}

        fig = plt.figure(figsize=(20, 11))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.75], hspace=0.28, wspace=0.08)
        cmap_h = plt.get_cmap("viridis").copy(); cmap_h.set_bad("0.6")
        cmap_e = plt.get_cmap("magma").copy(); cmap_e.set_bad("0.6")
        ax = fig.add_subplot(gs[0, 0]); _im(ax, rgb); ax.axhline(lin, color="w", lw=1, ls="--")
        ax.set_title(T["rgb"], fontsize=10); ax.axis("off")
        ax = fig.add_subplot(gs[0, 1]); hh = _im(ax, gt, cmap=cmap_h, vmin=0, vmax=vmax)
        ax.axhline(lin, color="w", lw=1, ls="--"); ax.set_title(T["gt"], fontsize=10); ax.axis("off")
        for j, (k, p) in enumerate(preds):
            ax = fig.add_subplot(gs[0, 2 + j]); _im(ax, p, cmap=cmap_h, vmin=0, vmax=vmax)
            ax.set_title(f"{T[k]}\nRMSE {rmse[k]:.2f} m | MAE {mae[k]:.2f} m", fontsize=10); ax.axis("off")
        fig.colorbar(hh, ax=fig.axes[1:6], fraction=0.012, pad=0.01, label=T["h"])
        rgb_r = np.clip(right.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        ax = fig.add_subplot(gs[1, 0]); _im(ax, rgb_r)
        ax.set_title(T["rgb_r"], fontsize=10); ax.axis("off")
        ax = fig.add_subplot(gs[1, 1]); _im(ax, np.where(m, 1.0, 0.35), cmap="gray", vmin=0, vmax=1)
        ax.set_title(T["mask"], fontsize=10); ax.axis("off")
        eaxes = []
        for j, (k, p) in enumerate(preds):
            ax = fig.add_subplot(gs[1, 2 + j]); eaxes.append(ax)
            he = _im(ax, np.where(m, np.abs(p - agl), np.nan), cmap=cmap_e, vmin=0, vmax=5)
            ax.set_title(f"{T['err']}\n{T[k]}", fontsize=9); ax.axis("off")
        fig.colorbar(he, ax=eaxes, fraction=0.012, pad=0.01, label=f"{T['err']} (m)")
        ax = fig.add_subplot(gs[2, :])
        xs = np.arange(gt.shape[1])
        ax.plot(xs, gt[lin], color="k", lw=2.2, label=T["gt"])
        for (k, p), c in zip(preds, ["tab:blue", "tab:gray", "tab:orange", "tab:green"]):
            ax.plot(xs, p[lin], color=c, lw=1.2, alpha=0.9, label=T[k], ls=":" if k == "sgm" else "-")
        ax.set_xlim(0, xs[-1]); ax.set_ylim(min(-2, np.nanmin(gt[lin]) - 1), vmax * 1.25 + 1)
        ax.set_xlabel(T["x"]); ax.set_ylabel(T["h"]); ax.set_title(T["perfil"], fontsize=10)
        ax.legend(ncol=5, fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
        extra = (f"{fora:.1%} da verdade fora de [{min_d}, {min_d + max_d}) px; |disp| máx {dmax:.0f} px" if args.lang == "pt"
                 else f"{fora:.1%} of ground truth outside [{min_d}, {min_d + max_d}) px; max |disp| {dmax:.0f} px")
        fig.suptitle(f"{T[rotulo] if rotulo in T else rotulo}: {par}  ({m_px:.2f} m/px; {extra})", fontsize=13)
        f = out / f"{rotulo}_{par.rstrip('_')}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"{rotulo:7s} {par} | RMSE A {rmse['A']:.2f} SGM {rmse['sgm']:.2f} RAFT {rmse['raft']:.2f} "
              f"A+B {rmse['ab']:.2f} m -> {f}", flush=True)

    (out / "selecionados.json").write_text(json.dumps(resumo, indent=1))
    print(f"Resumo: {out / 'selecionados.json'}")


if __name__ == "__main__":
    main()
