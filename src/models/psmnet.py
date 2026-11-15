"""PSMNet (Chang & Chen, CVPR 2018), versão "basic", adaptada ao US3D.

Diferenças em relação ao original:
  - disparidades podem ser negativas: o volume de custo cobre [min_disp, min_disp + max_disp)
  - pirâmide espacial (SPP) com pooling adaptativo, para aceitar recortes de qualquer tamanho
Convenção (conferida nos dados): o pixel x da esquerda corresponde a x - d na direita.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def convbn(cin, cout, k, stride, pad, dilation):
    pad = dilation if dilation > 1 else pad
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, pad, dilation, bias=False), nn.BatchNorm2d(cout))


def convbn3d(cin, cout, k, stride, pad):
    return nn.Sequential(nn.Conv3d(cin, cout, k, stride, pad, bias=False), nn.BatchNorm3d(cout))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(convbn(cin, cout, 3, stride, pad, dilation), nn.ReLU(inplace=True))
        self.conv2 = convbn(cout, cout, 3, 1, pad, dilation)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class FeatureExtraction(nn.Module):
    """CNN 2D + pirâmide espacial. Saída: 32 canais em 1/4 da resolução."""

    def __init__(self):
        super().__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(128, 3, 1, 1, 2)
        self.branches = nn.ModuleList(
            # sem BatchNorm aqui: o ramo mais grosso pode virar 1x1 e quebrar a BN com batch 1
            [nn.Sequential(nn.Conv2d(128, 32, 1, bias=True), nn.ReLU(inplace=True)) for _ in range(4)])
        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, 1, 1, 0, bias=False))

    def _make_layer(self, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes, 1, None, pad, dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        raw = self.layer2(x)
        x = self.layer3(raw)
        skip = self.layer4(x)
        h, w = skip.shape[-2:]
        feats = [raw, skip]
        for k, branch in zip((64, 32, 16, 8), self.branches):  # mesmas escalas do original (em pixels de 1/4)
            pooled = F.adaptive_avg_pool2d(skip, (max(1, h // k), max(1, w // k)))
            feats.append(F.interpolate(branch(pooled), (h, w), mode="bilinear", align_corners=False))
        return self.lastconv(torch.cat(feats, 1))


class PSMNetBasic(nn.Module):
    def __init__(self, max_disp=128, min_disp=-64):
        super().__init__()
        assert max_disp % 4 == 0 and min_disp % 4 == 0
        self.max_disp, self.min_disp = max_disp, min_disp
        self.feature = FeatureExtraction()
        self.dres0 = nn.Sequential(convbn3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True),
                                   convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn3d(32, 32, 3, 1, 1))
        self.dres2 = nn.Sequential(convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn3d(32, 32, 3, 1, 1))
        self.dres3 = nn.Sequential(convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn3d(32, 32, 3, 1, 1))
        self.dres4 = nn.Sequential(convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn3d(32, 32, 3, 1, 1))
        self.classify = nn.Sequential(convbn3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, 3, 1, 1, bias=False))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def cost_volume(self, left, right):
        b, c, h, w = left.shape
        n = self.max_disp // 4
        d0 = self.min_disp // 4
        cost = left.new_zeros(b, 2 * c, n, h, w)
        for i in range(n):
            d = d0 + i  # disparidade candidata (em pixels de 1/4)
            cost[:, :c, i] = left
            if d > 0:
                cost[:, c:, i, :, d:] = right[:, :, :, :-d]
            elif d < 0:
                cost[:, c:, i, :, :d] = right[:, :, :, -d:]
            else:
                cost[:, c:, i] = right
        return cost

    def forward(self, left, right):
        h, w = left.shape[-2:]
        cost = self.cost_volume(self.feature(left), self.feature(right))
        x = self.dres0(cost)
        x = self.dres1(x) + x
        x = self.dres2(x) + x
        x = self.dres3(x) + x
        x = self.dres4(x) + x
        x = self.classify(x)                                   # B x 1 x D/4 x H/4 x W/4
        x = F.interpolate(x, (self.max_disp, h, w), mode="trilinear", align_corners=False)
        prob = F.softmax(x.squeeze(1).float(), dim=1)          # B x D x H x W
        values = torch.arange(self.min_disp, self.min_disp + self.max_disp, device=prob.device,
                              dtype=prob.dtype).view(1, -1, 1, 1)
        return (prob * values).sum(1, keepdim=True)           # B x 1 x H x W (disparidade em pixels)
