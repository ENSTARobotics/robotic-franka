#!/usr/bin/env python3
"""Convierte mini-Xception de Keras (pesos del TD4 ROB311) a PyTorch y verifica que coinciden.

    python3 convertir_pesos.py [--ejemplos "<carpeta del TD4>/assets/examples"]

Necesita TensorFlow sólo para esta conversión (una vez). Escribe modelos/mini_xception_48.pt,
que es lo que usa rostro.py (sólo PyTorch).
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import cv2
import numpy as np
import torch

from mini_xception import MiniXception

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")


def modelo_keras():
    """Misma topología que build_model() de cnn.py del TD4."""
    import tensorflow as tf
    capas = tf.keras.layers
    entrada = capas.Input((48, 48, 1))
    x = entrada
    for _ in range(2):
        x = capas.Conv2D(8, 3, use_bias=False)(x)
        x = capas.BatchNormalization()(x)
        x = capas.Activation("relu")(x)
    for filtros in (16, 32, 64, 128):
        residuo = capas.Conv2D(filtros, 1, strides=2, padding="same", use_bias=False)(x)
        residuo = capas.BatchNormalization()(residuo)
        x = capas.SeparableConv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = capas.BatchNormalization()(x)
        x = capas.Activation("relu")(x)
        x = capas.SeparableConv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = capas.BatchNormalization()(x)
        x = capas.MaxPooling2D(3, strides=2, padding="same")(x)
        x = capas.Add()([x, residuo])
    x = capas.Conv2D(7, 3, padding="same")(x)
    x = capas.GlobalAveragePooling2D()(x)
    modelo = tf.keras.Model(entrada, capas.Activation("softmax")(x))
    modelo.load_weights(os.path.join(DIR_MODELOS, "mini_xception_48.weights.h5"))
    return modelo


def por_tipo(keras, tipo):
    """Capas de un tipo en orden de creación. keras.layers va en orden del grafo (la rama
    residual sale después de las separables), pero los nombres automáticos (conv2d,
    conv2d_1, ...) siguen el orden de creación de modelo_keras()."""
    capas = [c for c in keras.layers if type(c).__name__ == tipo]
    numero = lambda c: int(c.name.rsplit("_", 1)[1]) if c.name.rsplit("_", 1)[-1].isdigit() else 0
    return sorted(capas, key=numero)


def copiar(keras, torch_modelo):
    """Asigna cada capa de Keras a su módulo de PyTorch, por tipo y en orden de creación."""
    b = torch_modelo.bloques
    pares = (list(zip(por_tipo(keras, "Conv2D"), [torch_modelo.conv1, torch_modelo.conv2]
                      + [x.residuo for x in b] + [torch_modelo.final]))
             + list(zip(por_tipo(keras, "SeparableConv2D"), [s for x in b for s in (x.sep1, x.sep2)]))
             + list(zip(por_tipo(keras, "BatchNormalization"), [torch_modelo.bn1, torch_modelo.bn2]
                        + [n for x in b for n in (x.bn_residuo, x.bn1, x.bn2)])))
    assert len(pares) == len([c for c in keras.layers if c.get_weights()]) == 29, len(pares)
    origenes, destinos = zip(*pares)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    for capa, dst in zip(origenes, destinos):
        w = capa.get_weights()
        tipo = type(capa).__name__
        if tipo == "BatchNormalization":
            gamma, beta, media, var = w
            dst.weight.data, dst.bias.data = t(gamma), t(beta)
            dst.running_mean.data, dst.running_var.data = t(media), t(var)
        elif tipo == "SeparableConv2D":
            dst.depthwise.weight.data = t(w[0].transpose(2, 3, 0, 1))   # (k,k,in,1) -> (in,1,k,k)
            dst.pointwise.weight.data = t(w[1].transpose(3, 2, 0, 1))   # (1,1,in,out) -> (out,in,1,1)
        elif tipo == "Conv2D":
            dst.weight.data = t(w[0].transpose(3, 2, 0, 1))             # (k,k,in,out) -> (out,in,k,k)
            if len(w) > 1:
                dst.bias.data = t(w[1])
        else:
            raise ValueError("capa inesperada: %s" % tipo)


def main():
    p = argparse.ArgumentParser(description="Convierte mini-Xception Keras -> PyTorch")
    p.add_argument("--ejemplos", help="carpeta con caras de ejemplo (p. ej. assets/examples del TD4)")
    args = p.parse_args()

    keras = modelo_keras()
    red = MiniXception().eval()
    copiar(keras, red)

    # Verificación: mismas probabilidades con caras de ejemplo e imágenes aleatorias
    imagenes = []
    if args.ejemplos:
        for ruta in sorted(glob.glob(os.path.join(args.ejemplos, "*.png"))):
            gris = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            imagenes.append(cv2.resize(gris, (48, 48), interpolation=cv2.INTER_AREA))
    rng = np.random.default_rng(311)
    imagenes += list(rng.integers(0, 256, (16, 48, 48)).astype(np.uint8))
    lote = np.stack(imagenes).astype(np.float32)[..., None] / 127.5 - 1.0
    p_keras = keras(lote, training=False).numpy()
    with torch.no_grad():
        p_torch = red(torch.from_numpy(lote.transpose(0, 3, 1, 2).copy())).numpy()
    dif = float(np.abs(p_keras - p_torch).max())
    iguales = int((p_keras.argmax(1) == p_torch.argmax(1)).sum())
    print("Verificación con %d imágenes: diferencia máx. %.2e, misma clase en %d/%d"
          % (len(lote), dif, iguales, len(lote)))
    if dif > 1e-4 or iguales != len(lote):
        sys.exit("[error] la conversión no coincide con Keras: no se guarda")

    salida = os.path.join(DIR_MODELOS, "mini_xception_48.pt")
    torch.save(red.state_dict(), salida)
    print("Guardado %s" % salida)


if __name__ == "__main__":
    main()
