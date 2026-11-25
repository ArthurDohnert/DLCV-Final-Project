"""Compara o Caminho A (1 imagem) e o Caminho B (2 imagens) de forma justa.

Os dois são avaliados nas MESMAS imagens inteiras e nos MESMOS pixels: onde há verdade
de altura e de disparidade (dentro do intervalo de disparidade do modelo B).

Conversão de B para metros: reta altura = a*disparidade + b ajustada em cada par na verdade
de solo (substitui a calibração das câmeras, que não vem nos arquivos da Track 2). Para não
favorecer B, A também é reportado com uma reta ajustada por par sobre a própria previsão
("A + ajuste"), um benefício maior do que o que B recebe.

Uso (notebook):
  .venv\\Scripts\\python -m src.evaluate --height runs\\unet_r34_local\\best.pt --stereo runs\\psmnet_local\\best.pt
  --stereo sgm usa o baseline clássico (Semi-Global Matching, CPU, sem rede neural).
  --solo acrescenta a análise post hoc "B relativo ao chão de A": a U-Net marca o chão (A < 1 m),
  uma superfície suave do terreno é ajustada à disparidade prevista nesses pixels e subtraída
  antes da conversão para metros (o estéreo mede terreno + objetos; o alvo é altura acima do solo).
Saídas em outputs/eval/: metrics.json, por_par.csv, figuras e previsões de alguns pares (.npz).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, Track2Stereo, load_split


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


GEOMETRIAS = [(0, 1, "< 1 m/px (boa)"), (1, 2, "1-2 m/px"), (2, 1e9, "> 2 m/px (ruim)")]


def combine(h_a, h_b, m_por_px):
    """Combinações de A e B, com regras fixadas antes de ver os resultados.

    media:      (A + B) / 2
    relevo:     no chão usa A, no que é alto usa B. Peso de B = 0 quando max(A, B) < 1 m,
                1 quando > 3 m, e cresce linearmente entre 1 e 3 m.
    relevo_geo: como relevo, mas usa só A quando a geometria do par é ruim (>= 2 m por pixel
                de disparidade); na prática essa geometria vem dos ângulos nos metadados.
    """
    w = np.clip((np.maximum(h_a, h_b) - 1.0) / 2.0, 0.0, 1.0)
    relevo = (1 - w) * h_a + w * h_b
    return {"AB_media": 0.5 * (h_a + h_b), "AB_relevo": relevo,
            "AB_relevo_geo": h_a if m_por_px >= 2 else relevo}


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
    ap.add_argument("--stereo", required=True,
                    help="Caminho B: best.pt da PSMNet ou da RAFT-Stereo, pesos .pth da RAFT-Stereo sem ajuste, "
                         "ou 'sgm' (baseline clássico, sem aprendizado)")
    ap.add_argument("--raft-dir", default=None, help="repositório oficial da RAFT-Stereo (padrão: third_party/RAFT-Stereo)")
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--solo", action="store_true",
                    help="análise post hoc: converte B relativo ao terreno estimado nos pixels de chão de A")
    ap.add_argument("--solo-cobertura", type=float, default=0.25,
                    help="fração mínima de blocos 32x32 com chão; abaixo disso B fica com a reta original")
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
    is_sgm = args.stereo.lower() == "sgm"
    ck_b = {} if is_sgm else torch.load(args.stereo, map_location=device, weights_only=False)
    is_raft = not is_sgm and ("model" not in ck_b or ck_b.get("args", {}).get("model") == "raft")
    if is_sgm:
        from src.models.sgm import sgm_disparity
        model_b = None
        min_d, max_d = -64, 128                     # mesmos pixels avaliados nas redes
        b_name, epoca_b = "B: 2 imagens (SGM clássico)", "sem treino"
    elif is_raft:
        from src.models.raft import RaftDisparity
        iters = ck_b.get("args", {}).get("valid_iters", 32)   # mesmas iterações usadas na validação do treino
        model_b = RaftDisparity(args.raft_dir, mixed_precision=amp, valid_iters=iters).to(device)
        if "model" in ck_b:
            model_b.load_state_dict(ck_b["model"])
        else:
            model_b.load_pretrained(args.stereo)   # pesos oficiais, sem ajuste ao US3D
        min_d, max_d = -64, 128                     # mesmos pixels avaliados na PSMNet
        b_name, epoca_b = "B: 2 imagens (RAFT-Stereo)", ck_b.get("epoch", "sem ajuste")
    else:
        min_d, max_d = ck_b["args"]["min_disp"], ck_b["args"]["max_disp"]
        model_b = PSMNetBasic(max_d, min_d).to(device)
        model_b.load_state_dict(ck_b["model"])
        b_name, epoca_b = "B: 2 imagens (PSMNet)", ck_b["epoch"]
    if model_b is not None:
        model_b.eval()
    print(f"A: {args.height} (época {ck_a['epoch']}) | B: {args.stereo} (época {epoca_b})")
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    pairs = load_split(args.splits, args.split)
    ds = Track2Stereo(args.data, pairs, crop=None, train=False, augment=False)
    # baseline: altura média do conjunto de TREINO (não do conjunto avaliado)
    tr = Track2Stereo(args.data, load_split(args.splits, "train")[:30], crop=None, train=False, augment=False)
    tot = cnt = 0.0
    for j in range(len(tr)):
        _, _, _, _, a_, m_ = tr[j]
        tot += a_[m_].sum().item(); cnt += m_.sum().item()
    mean_agl = tot / max(cnt, 1)
    metodos = ["media", "A", "A_ajuste", "B", "AB_media", "AB_relevo", "AB_relevo_geo"]
    if args.solo:
        from src.terrain import superficie_solo
        metodos += ["B_solo", "AB_relevo_solo", "AB_relevo_geo_solo"]
    solo_usado = solo_recuo = 0
    acc = {k: Acc() for k in metodos}
    acc_geo = {nome: {k: Acc() for k in metodos} for _, _, nome in GEOMETRIAS}
    epe = d1 = nd = 0.0
    cobertura = []   # SGM: fração de pixels com resposta antes do preenchimento
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
            if is_sgm:  # imagens de volta para uint8; roda na CPU
                to_u8 = lambda x: ((x[None] * std_t + mean_t)[0] * 255).clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
                d_b, cov = sgm_disparity(to_u8(left), to_u8(right), min_d, max_d)
                cobertura.append(cov)
            elif is_raft:  # a RAFT-Stereo recebe as imagens em 0-255 e processa a imagem inteira de uma vez
                raw_l = ((left[None] * std_t + mean_t) * 255).to(device)
                raw_r = ((right[None] * std_t + mean_t) * 255).to(device)
                d_b = model_b(raw_l, raw_r).float()[0, 0].cpu().numpy()
            else:
                d_b = predict_disp(model_b, left[None].to(device), right[None].to(device), amp).numpy()
            cal = fit_line(disp[m], agl[m])          # geometria do par (verdade disp -> altura)
            fa = fit_line(h_a[m], agl[m])            # benefício extra dado a A
            if cal is None or fa is None:
                continue
            h_b = cal[0] * d_b + cal[1]
            h_a2 = fa[0] * h_a + fa[1]
            gt = agl[m]
            m_px = float(abs(cal[0]))
            geo = next(nome for lo, hi, nome in GEOMETRIAS if lo <= m_px < hi)
            comb = combine(h_a, h_b, m_px)
            preds = {"media": np.full_like(gt, mean_agl), "A": h_a[m], "A_ajuste": h_a2[m], "B": h_b[m]}
            preds.update({k: v[m] for k, v in comb.items()})
            if args.solo:
                # chão segundo A; superfície do terreno na disparidade PREVISTA (nenhuma verdade aqui)
                chao = h_a < 1.0
                sup_b = superficie_solo(d_b.astype(np.float64), chao, min_cobertura=args.solo_cobertura)
                # calibração por par, como na avaliação oficial: verdade de disparidade e de altura,
                # com o terreno da verdade estimado da mesma forma (mesma máscara de chão de A)
                sup_g = superficie_solo(np.where(m, disp, 0.0), chao & m, min_cobertura=args.solo_cobertura)
                cal_s = None if sup_b is None or sup_g is None else fit_line(disp[m] - sup_g[m], agl[m])
                if cal_s is None:
                    h_bs = h_b; solo_recuo += 1           # pouco chão visível: mantém a reta original
                else:
                    h_bs = cal_s[0] * (d_b - sup_b) + cal_s[1]; solo_usado += 1
                comb_s = combine(h_a, h_bs, m_px)
                preds.update({"B_solo": h_bs[m], "AB_relevo_solo": comb_s["AB_relevo"][m],
                              "AB_relevo_geo_solo": comb_s["AB_relevo_geo"][m]})
            row = {"par": pairs[i], "geometria": geo}
            for k, pred in preds.items():
                err = pred - gt
                acc[k].add(err); acc_geo[geo][k].add(err)
                row[f"{k}_rmse"] = float(np.sqrt((err ** 2).mean()))
            e = np.abs(d_b[m] - disp[m]); epe += e.sum(); d1 += (e > 3).sum(); nd += e.size
            row["B_epe_px"] = float(e.mean()); row["m_por_px"] = m_px
            rows.append(row)
            if i < args.save_preds:
                np.savez_compressed(out / "preds" / f"{pairs[i]}.npz", gt=np.where(amask[0].numpy(), agl, np.nan).astype(np.float16),
                                    a=h_a.astype(np.float16), b=h_b.astype(np.float16),
                                    ab=comb["AB_relevo_geo"].astype(np.float16), mask=m)
            print(f"{i + 1}/{len(ds)} {pairs[i]} | RMSE A {row['A_rmse']:.2f} m, B {row['B_rmse']:.2f} m", flush=True)

    res = {k: v.result() for k, v in acc.items()}
    res["B_disparidade"] = {"epe_px": float(epe / max(nd, 1)), "d1_3px": float(d1 / max(nd, 1))}
    if cobertura:
        res["B_disparidade"]["cobertura_sgm"] = float(np.mean(cobertura))
    res["pares_avaliados"] = len(rows)
    if args.solo:
        res["solo"] = {"pares_corrigidos": solo_usado, "pares_com_reta_original": solo_recuo,
                       "cobertura_minima": args.solo_cobertura}
    res["pixels_avaliados"] = int(acc["A"].n)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(out / "por_par.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\nResultado no split", args.split, f"({len(rows)} pares, mesmos pixels para todos):")
    names = {"media": "Baseline (altura média)", "A": "A: 1 imagem (U-Net)", "A_ajuste": "A + ajuste por par",
             "B": b_name, "AB_media": "A+B: média",
             "AB_relevo": "A+B: por relevo", "AB_relevo_geo": "A+B: relevo + geometria"}
    if args.solo:
        names.update({"B_solo": "B rel. ao chão de A*", "AB_relevo_solo": "A+B: relevo, chão de A*",
                      "AB_relevo_geo_solo": "A+B: rel.+geo, chão de A*"})
    print(f"{'método':28s} {'MAE (m)':>8s} {'RMSE (m)':>9s} {'erro<1m':>8s}")
    for k, name in names.items():
        r = res[k]; print(f"{name:28s} {r['mae_m']:8.2f} {r['rmse_m']:9.2f} {r['erro_menor_1m']:8.1%}")
    print(f"B em disparidade: EPE {res['B_disparidade']['epe_px']:.2f} px, D1 {res['B_disparidade']['d1_3px']:.1%}")
    if args.solo:
        print(f"* análise post hoc: {solo_usado} pares corrigidos pelo terreno, "
              f"{solo_recuo} com pouco chão visível (mantida a reta original)")
    if cobertura:
        print(f"SGM: {np.mean(cobertura):.1%} dos pixels com resposta antes do preenchimento")

    # Geometria do par: quantos metros de altura cada pixel de disparidade representa.
    # Par com ângulo pequeno entre as duas vistas = muitos metros por pixel = estéreo pouco preciso.
    print("\nPor geometria do par (por pixel; 'vence A' = pares em que o método erra menos que A):")
    res["por_geometria"] = {}
    for _, _, nome in GEOMETRIAS:
        sel = [r for r in rows if r["geometria"] == nome]
        if not sel:
            continue
        res["por_geometria"][nome] = {"pares": len(sel)}
        print(f"  {nome} ({len(sel)} pares)")
        for k in ["A", "B", "AB_media", "AB_relevo", "AB_relevo_geo"] + (
                ["B_solo", "AB_relevo_solo", "AB_relevo_geo_solo"] if args.solo else []):
            r = acc_geo[nome][k].result()
            vence = sum(x[f"{k}_rmse"] < x["A_rmse"] for x in sel)
            res["por_geometria"][nome][k] = dict(r, vence_A=vence)
            extra = "" if k == "A" else f" | vence A em {vence}"
            print(f"    {names[k]:26s} MAE {r['mae_m']:5.2f} | RMSE {r['rmse_m']:5.2f} | erro<1m {r['erro_menor_1m']:6.1%}{extra}")
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
