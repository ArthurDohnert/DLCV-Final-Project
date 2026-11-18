"""RAFT-Stereo (Lipson et al., 3DV 2021) adaptada ao US3D.

O código oficial não fica no repositório: scripts/pcad/get_raft.sh o baixa em
third_party/RAFT-Stereo, junto com os pesos pré-treinados.

Convenções (conferidas no código oficial):
  - entrada: imagens RGB em 0-255 (a rede normaliza por dentro)
  - saída: "fluxo" horizontal = -disparidade (pixel x da esquerda ↔ x - d na direita, igual à PSMNet)
  - altura e largura múltiplas de 32: o adaptador preenche e recorta de volta
  - a correlação compara todas as posições da linha, então disparidades negativas funcionam
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "RAFT-Stereo"


def raft_args(mixed_precision=True):
    """Mesmos hiperparâmetros do modelo pré-treinado em SceneFlow (padrões do repositório oficial)."""
    return argparse.Namespace(
        hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4,
        corr_radius=4, n_downsample=2, context_norm="batch", slow_fast_gru=False, n_gru_layers=3,
        mixed_precision=mixed_precision)


class RaftDisparity(nn.Module):
    """Treino: devolve a lista de disparidades de cada iteração. Avaliação: só a última."""

    def __init__(self, raft_dir=None, mixed_precision=True, train_iters=16, valid_iters=32):
        super().__init__()
        raft_dir = Path(raft_dir) if raft_dir else DEFAULT_DIR
        if not (raft_dir / "core" / "raft_stereo.py").exists():
            raise FileNotFoundError(f"RAFT-Stereo não encontrada em {raft_dir}. Rode scripts/pcad/get_raft.sh")
        if str(raft_dir) not in sys.path:
            sys.path.insert(0, str(raft_dir))
        from core.raft_stereo import RAFTStereo  # noqa: E402 (vem do repositório oficial)
        self.net = RAFTStereo(raft_args(mixed_precision))
        self.train_iters, self.valid_iters = train_iters, valid_iters

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.net.load_state_dict(state, strict=True)

    def forward(self, left, right):
        h, w = left.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        if ph or pw:
            left, right = F.pad(left, pad, mode="replicate"), F.pad(right, pad, mode="replicate")

        def unpad(x):
            return x[..., pad[2]:pad[2] + h, pad[0]:pad[0] + w]

        if self.training:
            flows = self.net(left.float(), right.float(), iters=self.train_iters)
            return [-unpad(f) for f in flows]
        _, flow = self.net(left.float(), right.float(), iters=self.valid_iters, test_mode=True)
        return -unpad(flow)
