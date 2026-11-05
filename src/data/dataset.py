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
