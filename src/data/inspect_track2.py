"""Passo 9 do guia: inspeciona a Track 2 do DFC2019 (US3D) e cria o split por tile.

Responde às perguntas da semana 1 sem abrir arquivo por arquivo:
  - quantos pares e quantos tiles geográficos existem
  - se cada LEFT_RGB tem RIGHT_RGB, LEFT_DSP (disparidade) e LEFT_AGL (altura)
  - intervalo de disparidade e de altura (para configurar as redes)
  - split treino/val/teste POR TILE, salvo em splits.json
  - figura com exemplos (esquerda, direita, disparidade, altura)
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NO_DATA = -999.0  # valor usado pelo DFC2019 para pixels sem verdade de solo


def tile_id(base):
    """Tile geográfico a partir do nome. O dataset usa JAX_156_013_008_ e OMA276_004_003_."""
    m = re.match(r"([A-Z]+)_?(\d+)_", base)
    return f"{m.group(1)}_{m.group(2)}" if m else base.split("_")[0]


def valid(a):
    return np.isfinite(a) & (a > NO_DATA + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta com os .tif da Track 2")
    ap.add_argument("--out", default="outputs/inspect")
    ap.add_argument("--sample", type=int, default=300, help="pares usados nas estatísticas")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lefts = sorted(data.glob("*_LEFT_RGB.tif"))
    print(f"Pares encontrados (LEFT_RGB): {len(lefts)}")
    print("Exemplos de nome:", [p.name for p in lefts[:3]])

    pairs, missing = [], defaultdict(int)
    for left in lefts:
        base = left.name[: -len("LEFT_RGB.tif")]
        files = {k: data / f"{base}{k}.tif" for k in ["LEFT_RGB", "RIGHT_RGB", "LEFT_DSP", "LEFT_AGL", "LEFT_CLS"]}
        for k, f in files.items():
            if not f.exists():
                missing[k] += 1
        if all(files[k].exists() for k in ["LEFT_RGB", "RIGHT_RGB", "LEFT_DSP", "LEFT_AGL"]):
            pairs.append(base)
    print(f"Pares completos (RGB esq+dir, DSP, AGL): {len(pairs)}")
    if missing:
        print("Arquivos faltando por tipo:", dict(missing))
    if not pairs:
        raise SystemExit("Nenhum par completo. Faltou descompactar o zip de Reference (Truth)?")

    # tile geográfico = cidade + número do tile, ex. JAX_004 (os demais campos são as vistas)
    by_tile = defaultdict(list)
    for b in pairs:
        by_tile[tile_id(b)].append(b)
    cities = defaultdict(int)
    for t in by_tile:
        cities[t.split("_")[0]] += 1
    print(f"Tiles geográficos: {len(by_tile)} | por cidade: {dict(cities)}")
    counts = sorted(len(v) for v in by_tile.values())
    print(f"Pares por tile: min {counts[0]}, mediana {counts[len(counts)//2]}, max {counts[-1]}")

    # split por tile (nunca o mesmo lugar em treino e teste), estratificado por cidade
    rng = random.Random(0)
    split = {"train": [], "val": [], "test": []}
    for city in cities:
        tiles = sorted(t for t in by_tile if t.startswith(city))
        rng.shuffle(tiles)
        n = len(tiles); n_test = max(1, round(0.15 * n)); n_val = max(1, round(0.15 * n))
        split["test"] += tiles[:n_test]
        split["val"] += tiles[n_test:n_test + n_val]
        split["train"] += tiles[n_test + n_val:]
    result = {k: {"tiles": sorted(v), "pairs": sorted(p for t in v for p in by_tile[t])} for k, v in split.items()}
    (out / "splits.json").write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        print(f"split {k}: {len(v['tiles'])} tiles, {len(v['pairs'])} pares")

    # estatísticas numa amostra
    sample = rng.sample(pairs, min(args.sample, len(pairs)))
    d_vals, a_vals, nod, shapes, dtypes = [], [], [], set(), set()
    for b in sample:
        rgb = tifffile.imread(data / f"{b}LEFT_RGB.tif")
        dsp = tifffile.imread(data / f"{b}LEFT_DSP.tif").astype(np.float32)
        agl = tifffile.imread(data / f"{b}LEFT_AGL.tif").astype(np.float32)
        shapes.add((rgb.shape, dsp.shape, agl.shape)); dtypes.add((str(rgb.dtype), str(dsp.dtype), str(agl.dtype)))
        nod.append(1 - valid(dsp).mean())
        d_vals.append(dsp[valid(dsp)][::50]); a_vals.append(agl[valid(agl)][::50])
    d, a = np.concatenate(d_vals), np.concatenate(a_vals)
    pct = [0.1, 1, 50, 99, 99.9]
    print("Formatos (rgb, dsp, agl):", shapes)
    print("Tipos (rgb, dsp, agl):", dtypes)
    print(f"Disparidade: min {d.min():.1f}, max {d.max():.1f}, percentis {pct}: {np.round(np.percentile(d, pct), 1).tolist()}")
    print(f"Fração média de NO_DATA na disparidade: {np.mean(nod):.1%}")
    print(f"Altura AGL (m): min {a.min():.1f}, max {a.max():.1f}, percentis {pct}: {np.round(np.percentile(a, pct), 1).tolist()}")
    stats = {"disp_min": float(d.min()), "disp_max": float(d.max()),
             "disp_p01": float(np.percentile(d, 0.1)), "disp_p999": float(np.percentile(d, 99.9)),
             "agl_mean": float(a.mean()), "agl_p999": float(np.percentile(a, 99.9))}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))

    # figura: 3 pares de tiles diferentes
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, t in zip(axes, rng.sample(sorted(by_tile), 3)):
        b = by_tile[t][0]
        imgs = [tifffile.imread(data / f"{b}{k}.tif") for k in ["LEFT_RGB", "RIGHT_RGB", "LEFT_DSP", "LEFT_AGL"]]
        for ax, im, title in zip(row, imgs, ["esquerda", "direita", "disparidade (px)", "altura AGL (m)"]):
            if im.ndim == 2:
                im = np.where(valid(im.astype(np.float32)), im, np.nan)
                h = ax.imshow(im, cmap="viridis"); plt.colorbar(h, ax=ax, fraction=0.046)
            else:
                ax.imshow(im[..., :3])
            ax.set_title(f"{b.rstrip('_')}: {title}", fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "exemplos_track2.png", dpi=110)
    print(f"Figura salva em {out / 'exemplos_track2.png'}")


if __name__ == "__main__":
    main()
