"""Superfície do terreno em unidades de disparidade.

O estéreo mede a altura da superfície (terreno + objetos); o US3D pede a altura acima do solo.
Ajustando uma superfície suave à disparidade nos pixels de chão e subtraindo-a, sobra só a
parte da disparidade que vem dos objetos, que é proporcional à altura acima do solo.
"""
import numpy as np


def superficie_solo(d, chao, bloco=32, min_amostras=20, min_cobertura=0.0):
    """Mediana da disparidade do chão em blocos, buracos preenchidos pelos vizinhos, interpolada.

    d: disparidade HxW; chao: máscara HxW dos pixels de chão.
    Retorna None se a fração de blocos com chão for menor que min_cobertura (pouco chão visível:
    a superfície teria de ser inventada a partir de blocos distantes).
    """
    import cv2
    h, w = d.shape
    gh, gw = h // bloco, w // bloco
    g = np.full((gh, gw), np.nan)
    for i in range(gh):
        for j in range(gw):
            sl = (slice(i * bloco, (i + 1) * bloco), slice(j * bloco, (j + 1) * bloco))
            v = d[sl][chao[sl]]
            if v.size >= min_amostras:
                g[i, j] = np.median(v)
    cobertura = float(np.isfinite(g).mean())
    if cobertura == 0.0 or cobertura < min_cobertura:
        return None
    while np.isnan(g).any():                       # preenche com a média dos vizinhos válidos
        p = np.pad(g, 1, constant_values=np.nan)
        viz = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = np.isfinite(viz).sum(0)
        media = np.where(cnt > 0, np.nansum(viz, 0) / np.maximum(cnt, 1), np.nan)
        g = np.where(np.isnan(g), media, g)
    g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
