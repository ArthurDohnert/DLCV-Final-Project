"""Datasets da Track 2 do DFC2019 (US3D).

Track2Height: imagem esquerda -> altura acima do solo (Caminho A).
Os pares de cada split vêm de outputs/inspect/splits.json (split por tile geográfico).
"""
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

NO_DATA = -999.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split(splits_file, split):
    return json.loads(Path(splits_file).read_text())[split]["pairs"]


def valid_mask(a):
    return np.isfinite(a) & (a > NO_DATA + 1)


class Track2Height(Dataset):
    """Retorna (imagem normalizada 3xHxW, altura 1xHxW em metros, máscara 1xHxW)."""

    def __init__(self, data_dir, pairs, crop=512, train=True, augment=True):
        self.data_dir = Path(data_dir)
        self.pairs = list(pairs)
        self.crop = crop          # None = imagem inteira (1024x1024)
        self.train = train
        self.augment = augment and train

    def __len__(self):
        return len(self.pairs)

    def _read(self, base):
        rgb = tifffile.imread(self.data_dir / f"{base}LEFT_RGB.tif")[..., :3]
        agl = tifffile.imread(self.data_dir / f"{base}LEFT_AGL.tif").astype(np.float32)
        return rgb, agl

    def __getitem__(self, i):
        rgb, agl = self._read(self.pairs[i])
        h, w = agl.shape
        if self.crop and self.crop < min(h, w):
            if self.train:
                y = np.random.randint(0, h - self.crop + 1)
                x = np.random.randint(0, w - self.crop + 1)
            else:  # recorte central, determinístico
                y, x = (h - self.crop) // 2, (w - self.crop) // 2
            rgb = rgb[y:y + self.crop, x:x + self.crop]
            agl = agl[y:y + self.crop, x:x + self.crop]
        if self.augment:  # mesma transformação geométrica na imagem e na altura
            k = np.random.randint(4)
            rgb, agl = np.rot90(rgb, k), np.rot90(agl, k)
            if np.random.rand() < 0.5:
                rgb, agl = rgb[:, ::-1], agl[:, ::-1]
        mask = valid_mask(agl)
        agl = np.where(mask, agl, 0.0).astype(np.float32)
        img = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(agl[None])),
                torch.from_numpy(np.ascontiguousarray(mask[None])))


class Track2Stereo(Dataset):
    """Caminho B. Retorna (esquerda, direita, disparidade, máscara da disparidade, altura, máscara da altura).

    Os pares já vêm retificados (linhas epipolares horizontais). Por isso o único
    aumento geométrico permitido é o espelhamento vertical aplicado às duas imagens:
    girar ou espelhar na horizontal quebraria a geometria do par.
    crop é (altura, largura) ou None para a imagem inteira.
    """

    def __init__(self, data_dir, pairs, crop=(256, 512), train=True, augment=True):
        self.data_dir = Path(data_dir)
        self.pairs = list(pairs)
        self.crop = crop
        self.train = train
        self.augment = augment and train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        b = self.pairs[i]
        left = tifffile.imread(self.data_dir / f"{b}LEFT_RGB.tif")[..., :3]
        right = tifffile.imread(self.data_dir / f"{b}RIGHT_RGB.tif")[..., :3]
        disp = tifffile.imread(self.data_dir / f"{b}LEFT_DSP.tif").astype(np.float32)
        agl = tifffile.imread(self.data_dir / f"{b}LEFT_AGL.tif").astype(np.float32)
        h, w = disp.shape
        if self.crop:
            ch, cw = min(self.crop[0], h), min(self.crop[1], w)
            if self.train:
                y, x = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
            else:
                y, x = (h - ch) // 2, (w - cw) // 2
            sl = (slice(y, y + ch), slice(x, x + cw))
            left, right, disp, agl = left[sl], right[sl], disp[sl], agl[sl]
        if self.augment and np.random.rand() < 0.5:
            left, right, disp, agl = left[::-1], right[::-1], disp[::-1], agl[::-1]
        dmask, amask = valid_mask(disp), valid_mask(agl)
        disp = np.where(dmask, disp, 0.0).astype(np.float32)
        agl = np.where(amask, agl, 0.0).astype(np.float32)

        def norm(im):
            im = (im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a[None]))

        return norm(left), norm(right), t(disp), t(dmask), t(agl), t(amask)
