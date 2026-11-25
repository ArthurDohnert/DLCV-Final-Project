"""Piso de erro da conversão disparidade -> altura (sem rede neural nenhuma).

Aplica a calibração por par à disparidade VERDADEIRA e mede o erro contra a altura real.
É o erro que sobraria em B mesmo com um estéreo perfeito.
  reta:  altura = a*disp + b                  (calibração usada na avaliação oficial)
  plano: altura = a*disp + b + c*x + e*y      (reta + inclinação do terreno; análise post hoc)
A disparidade mede a altura absoluta (terreno + objetos); a altura do dataset é acima do solo.
Em terreno inclinado, a reta não separa uma coisa da outra.

Mesmos pixels da avaliação: verdade de altura e de disparidade válidas, disparidade em [-64, 64).
Uso (PCAD, só CPU):
  python -m src.conversion_floor --data $HOME/data/dfc2019/track2 --split test --out outputs/piso_test
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import tifffile

from src.data.dataset import load_split
from src.evaluate import GEOMETRIAS


def valido(a):
    return np.isfinite(a) & (a > -998)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--splits", default="outputs/inspect/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/piso_test")
    ap.add_argument("--csv-raft", default=None, help="por_par.csv da RAFT, para comparar o piso com o erro real de B")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    b_real = {}
    if args.csv_raft:
        for r in csv.DictReader(open(args.csv_raft, encoding="utf-8")):
            b_real[r["par"]] = float(r["B_rmse"])

    tot = {k: [0.0, 0.0, 0] for k in ("reta", "plano")}           # soma |e|, soma e², n
    geo = {g: {k: [0.0, 0.0, 0] for k in ("reta", "plano")} for _, _, g in GEOMETRIAS}
    rows = []
    for i, par in enumerate(load_split(args.splits, args.split)):
        d = tifffile.imread(data / f"{par}LEFT_DSP.tif").astype(np.float64)
        h = tifffile.imread(data / f"{par}LEFT_AGL.tif").astype(np.float64)
        m = valido(d) & valido(h) & (d >= -64) & (d < 64)
        if m.sum() < 100:
            continue
        yy, xx = np.nonzero(m)
        dv, hv = d[m], h[m]
        if dv.std() < 1e-6:
            continue
        n = 1024.0
        X1 = np.stack([dv, np.ones_like(dv)], 1)
        X2 = np.stack([dv, np.ones_like(dv), xx / n, yy / n], 1)
        c1, *_ = np.linalg.lstsq(X1, hv, rcond=None)
        c2, *_ = np.linalg.lstsq(X2, hv, rcond=None)
        e1, e2 = X1 @ c1 - hv, X2 @ c2 - hv
        m_px = abs(c1[0])
        g = next(nome for lo, hi, nome in GEOMETRIAS if lo <= m_px < hi)
        for k, e in (("reta", e1), ("plano", e2)):
            for acc in (tot[k], geo[g][k]):
                acc[0] += np.abs(e).sum(); acc[1] += (e ** 2).sum(); acc[2] += e.size
        row = {"par": par, "geometria": g, "m_por_px": m_px, "m_por_px_plano": abs(c2[0]),
               "inclinacao_m_por_tile": float(math.hypot(c2[2], c2[3])),
               "piso_reta_rmse": float(np.sqrt((e1 ** 2).mean())), "piso_plano_rmse": float(np.sqrt((e2 ** 2).mean()))}
        if par in b_real:
            row["B_raft_rmse"] = b_real[par]
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} pares", flush=True)

    res = lambda a: {"mae_m": a[0] / max(a[2], 1), "rmse_m": math.sqrt(a[1] / max(a[2], 1))}
    resumo = {"pares": len(rows), "geral": {k: res(v) for k, v in tot.items()},
              "por_geometria": {g: dict({k: res(v) for k, v in d.items()}, pares=sum(r["geometria"] == g for r in rows))
                                for g, d in geo.items()}}
    (out / "piso.json").write_text(json.dumps(resumo, indent=1))
    with open(out / "piso_por_par.csv", "w", newline="") as f:
        campos = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader(); w.writerows(rows)

    print(f"\nPiso da conversão com a disparidade VERDADEIRA ({len(rows)} pares, split {args.split}):")
    print(f"{'':22s} {'MAE (m)':>8s} {'RMSE (m)':>9s}")
    for k in ("reta", "plano"):
        r = resumo["geral"][k]; print(f"{k:22s} {r['mae_m']:8.2f} {r['rmse_m']:9.2f}")
    for g, d in resumo["por_geometria"].items():
        print(f"  {g} ({d['pares']} pares): reta RMSE {d['reta']['rmse_m']:.2f} | plano RMSE {d['plano']['rmse_m']:.2f}")
    pior = sorted(rows, key=lambda r: -r["piso_reta_rmse"])[:8]
    print("\nPares com maior piso (reta):")
    for r in pior:
        extra = f" | B real {r['B_raft_rmse']:.2f}" if "B_raft_rmse" in r else ""
        print(f"  {r['par']} reta {r['piso_reta_rmse']:.2f} m, plano {r['piso_plano_rmse']:.2f} m, "
              f"inclinação {r['inclinacao_m_por_tile']:.1f} m/tile{extra}")
    for p in ("OMA276_023_025_", "OMA172_027_019_", "JAX_260_006_011_"):
        r = next((x for x in rows if x["par"] == p), None)
        if r:
            extra = f" | B real {r['B_raft_rmse']:.2f}" if "B_raft_rmse" in r else ""
            print(f"  [{p}] reta {r['piso_reta_rmse']:.2f} m, plano {r['piso_plano_rmse']:.2f} m{extra}")


if __name__ == "__main__":
    main()
