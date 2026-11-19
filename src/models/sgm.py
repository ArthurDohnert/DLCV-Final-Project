"""Baseline clássico do Caminho B: Semi-Global Matching (OpenCV StereoSGBM), sem aprendizado.

Parâmetros iguais aos do baseline oficial do DFC2019 (pubgeo/dfc2019, track2/test-sgbm.py),
com duas diferenças:
  - intervalo de busca [-64, 64), o mesmo intervalo de pixels avaliado nas redes;
  - o filtro WLS do opencv-contrib (ximgproc) não está no ambiente; no lugar dele, os pixels
    sem resposta são preenchidos com o vizinho válido mais próximo na mesma linha e a
    disparidade recebe um filtro de mediana 5x5, para que a previsão seja densa como a das redes.
Convenção de disparidade igual à do US3D: pixel x da esquerda corresponde a x - d na direita.
"""
import cv2
import numpy as np


def _fill_rows(d, valid):
    """Preenche cada pixel inválido com o pixel válido mais próximo da mesma linha."""
    h, w = d.shape
    cols = np.arange(w)[None, :].repeat(h, 0)
    left = np.where(valid, cols, -1)
    np.maximum.accumulate(left, axis=1, out=left)               # último válido à esquerda
    right = np.where(valid, cols, w)
    right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]  # próximo válido à direita
    dl = np.where(left >= 0, cols - left, w + 1)
    dr = np.where(right < w, right - cols, w + 1)
    src = np.where(dl <= dr, left, right)
    ok = (src >= 0) & (src < w)
    out = d.copy()
    rr = np.arange(h)[:, None].repeat(w, 1)
    out[ok] = d[rr[ok], src[ok]]
    out[~ok] = 0.0                                              # linha inteira sem resposta
    return out


def sgm_disparity(left_rgb, right_rgb, min_disp=-64, num_disp=128, block=5):
    """left_rgb, right_rgb: arrays uint8 HxWx3. Retorna disparidade float32 HxW (densa)."""
    gl = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, disp12MaxDiff=1,
        uniquenessRatio=15, speckleWindowSize=0, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    valid = raw >= min_disp                                     # inválido = min_disp - 1
    d = _fill_rows(raw, valid)
    d = cv2.medianBlur(d, 5)
    return d, float(valid.mean())
