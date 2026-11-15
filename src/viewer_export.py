"""Exporta um par avaliado para o visualizador 3D (viewer/index.html).

Lê as previsões salvas por src/evaluate.py (.npz) e a imagem de satélite, reduz para uma
malha de 256x256, calcula a rota do rover sobre cada reconstrução e grava viewer/data.js.

Uso:
  .venv\\Scripts\\python -m src.viewer_export --pred outputs\\eval_lr3e4\\preds\\JAX_467_001_004_.npz
Depois abra viewer\\index.html no navegador.
"""
import argparse
import base64
import csv
import heapq
import io
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

GSD = 0.3          # metros por pixel da imagem WorldView-3 do US3D (aprox.)
MESH = 256         # vértices por lado da malha 3D
ROVER = 128        # células por lado da grade de planejamento
MAX_SLOPE = 0.35   # inclinação máxima que o rover sobe (~19 graus)


def block_mean(a, n):
    k = a.shape[0] // n
    return np.nanmean(a[: n * k, : n * k].reshape(n, k, n, k), axis=(1, 3))


def b64_heights(h):
    dm = np.clip(np.round(h * 10), -500, 4000).astype("<i2")  # decímetros
    return base64.b64encode(dm.tobytes()).decode()


def plan(h, cell, start, goal):
    """Dijkstra em 8 vizinhos; degraus mais íngremes que MAX_SLOPE são intransponíveis."""
    n = h.shape[0]
    dist = np.full((n, n), np.inf); prev = {}
    dist[start] = 0.0; pq = [(0.0, start)]
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if (i, j) == goal:
            break
        if d > dist[i, j]:
            continue
        for di, dj in steps:
            a, b = i + di, j + dj
            if not (0 <= a < n and 0 <= b < n):
                continue
            run = cell * math.hypot(di, dj)
            slope = abs(h[a, b] - h[i, j]) / run
            if slope > MAX_SLOPE:
                continue
            nd = d + run * (1 + 6 * slope)
            if nd < dist[a, b]:
                dist[a, b] = nd; prev[(a, b)] = (i, j); heapq.heappush(pq, (nd, (a, b)))
    if goal not in prev:
        return []
    path, p = [goal], goal
    while p != start:
        p = prev[p]; path.append(p)
    return path[::-1]


def check_on_truth(path, h_gt, cell):
    """Quantos passos da rota seriam intransponíveis no terreno real."""
    bad = 0
    for (i, j), (a, b) in zip(path, path[1:]):
        if abs(h_gt[a, b] - h_gt[i, j]) / (cell * math.hypot(a - i, b - j)) > MAX_SLOPE:
            bad += 1
    return bad


def low_point(h, i0, j0, r=10):
    win = h[i0:i0 + r, j0:j0 + r]
    di, dj = np.unravel_index(np.argmin(win), win.shape)
    return int(i0 + di), int(j0 + dj)


def reachable(h, cell, start):
    """Células alcançáveis a partir de start sem passar de MAX_SLOPE."""
    n = h.shape[0]; seen = np.zeros_like(h, bool); seen[start] = True; stack = [start]
    while stack:
        i, j = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < n and 0 <= b < n and not seen[a, b] and \
                        abs(h[a, b] - h[i, j]) / (cell * math.hypot(di, dj)) <= MAX_SLOPE:
                    seen[a, b] = True; stack.append((a, b))
    return seen


def far_goal(h, cell, start):
    """Destino no chão (< 1 m), alcançável no terreno real e o mais longe possível da origem."""
    ok = reachable(h, cell, start) & (h < 1.0)
    ii, jj = np.nonzero(ok)
    k = np.argmax((ii - start[0]) ** 2 + (jj - start[1]) ** 2)
    return int(ii[k]), int(jj[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help=".npz salvo por src.evaluate")
    ap.add_argument("--data", default="data/dfc2019/track2")
    ap.add_argument("--out", default="viewer/data.js")
    args = ap.parse_args()

    pred = Path(args.pred); pair = pred.stem
    z = np.load(pred)
    gt = z["gt"].astype(np.float32)
    gt = np.where(np.isfinite(gt), gt, np.nanmedian(gt))
    surfaces = {"real": gt, "a": z["a"].astype(np.float32), "b": z["b"].astype(np.float32)}

    metrics = {}
    csv_path = pred.parent.parent / "por_par.csv"
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            if r["par"] == pair:
                metrics = {"a_rmse": float(r["A_rmse"]), "b_rmse": float(r["B_rmse"]),
                           "m_por_px": float(r["m_por_px"]), "b_epe": float(r["B_epe_px"])}

    rgb = tifffile.imread(Path(args.data) / f"{pair}LEFT_RGB.tif")[..., :3]
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=82)
    size_m = gt.shape[0] * GSD
    cell = size_m / ROVER

    coarse = {k: block_mean(v, ROVER) for k, v in surfaces.items()}
    start = low_point(coarse["real"], 4, 4)
    goal = far_goal(coarse["real"], cell, start)
    routes = {}
    for k, h in coarse.items():
        path = plan(h, cell, start, goal)
        length = sum(cell * math.hypot(a - i, b - j) for (i, j), (a, b) in zip(path, path[1:]))
        routes[k] = {"path": path, "length_m": round(length, 1),
                     "blocked_steps_on_truth": check_on_truth(path, coarse["real"], cell) if path else None}
        print(f"rota sobre {k}: {len(path)} pontos, {length:.0f} m, passos intransponíveis no terreno real: "
              f"{routes[k]['blocked_steps_on_truth']}")

    data = {
        "pair": pair, "size_m": size_m, "mesh": MESH, "rover_grid": ROVER, "max_slope": MAX_SLOPE,
        "metrics": metrics,
        "heights": {k: b64_heights(block_mean(v, MESH)) for k, v in surfaces.items()},
        "texture": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "routes": routes,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.VIEWER_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"{out} gravado ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
