"""mini-Xception 48x48 en PyTorch, equivalente a la versión Keras del TD4 ROB311 de Joab.

Arquitectura y pesos originales: Octavio Arriaga et al., MIT
(https://github.com/oarriaga/face_classification); cabeza adaptada a 48x48 por Joab.
Los pesos se convierten una vez desde Keras con convertir_pesos.py -> modelos/mini_xception_48.pt.

Detalles para que coincida con Keras:
- BatchNormalization de Keras usa epsilon 1e-3 (PyTorch, 1e-5 por defecto).
- padding "same" de Keras en MaxPool 3x3 stride 2 es asimétrico (sobra a la derecha/abajo).
"""
import torch
from torch import nn
import torch.nn.functional as F

EPS_BN = 1e-3


def bn(canales):
    return nn.BatchNorm2d(canales, eps=EPS_BN)


class SeparableConv(nn.Module):
    """SeparableConv2D de Keras: depthwise 3x3 (padding same) + pointwise 1x1, sin bias."""

    def __init__(self, entrada, salida):
        super().__init__()
        self.depthwise = nn.Conv2d(entrada, entrada, 3, padding=1, groups=entrada, bias=False)
        self.pointwise = nn.Conv2d(entrada, salida, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def maxpool_same(x, k=3, s=2):
    """MaxPooling2D(3, strides=2, padding="same") de Keras."""
    pads = []
    for n in (x.shape[3], x.shape[2]):              # F.pad: primero ancho, luego alto
        salida = -(-n // s)
        total = max((salida - 1) * s + k - n, 0)
        pads += [total // 2, total - total // 2]
    return F.max_pool2d(F.pad(x, pads, value=float("-inf")), k, s)


class Bloque(nn.Module):
    def __init__(self, entrada, filtros):
        super().__init__()
        self.residuo = nn.Conv2d(entrada, filtros, 1, stride=2, bias=False)
        self.bn_residuo = bn(filtros)
        self.sep1, self.bn1 = SeparableConv(entrada, filtros), bn(filtros)
        self.sep2, self.bn2 = SeparableConv(filtros, filtros), bn(filtros)

    def forward(self, x):
        residuo = self.bn_residuo(self.residuo(x))
        x = F.relu(self.bn1(self.sep1(x)))
        x = maxpool_same(self.bn2(self.sep2(x)))
        return x + residuo


class MiniXception(nn.Module):
    """Entrada (N, 1, 48, 48) normalizada a [-1, 1]; salida: probabilidades de 7 clases."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 8, 3, bias=False), bn(8)
        self.conv2, self.bn2 = nn.Conv2d(8, 8, 3, bias=False), bn(8)
        self.bloques = nn.ModuleList([Bloque(e, f) for e, f in ((8, 16), (16, 32), (32, 64), (64, 128))])
        self.final = nn.Conv2d(128, 7, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        for bloque in self.bloques:
            x = bloque(x)
        return F.softmax(self.final(x).mean(dim=(2, 3)), dim=1)


def cargar(ruta, dispositivo):
    modelo = MiniXception()
    modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return modelo.to(dispositivo).eval()
