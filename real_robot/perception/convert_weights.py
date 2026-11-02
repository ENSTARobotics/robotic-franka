#!/usr/bin/env python3
"""Converts mini-Xception from Keras (weights from the ROB311 lab, TD4) to PyTorch and checks they match.

    python3 convert_weights.py [--examples "<TD4 folder>/assets/examples"]

Needs TensorFlow only for this conversion (once). Writes models/mini_xception_48.pt,
which is what face.py uses (PyTorch only).
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

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def keras_model():
    """Same topology as build_model() in TD4's cnn.py."""
    import tensorflow as tf
    layers = tf.keras.layers
    inputs = layers.Input((48, 48, 1))
    x = inputs
    for _ in range(2):
        x = layers.Conv2D(8, 3, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    for filters in (16, 32, 64, 128):
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Add()([x, residual])
    x = layers.Conv2D(7, 3, padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, layers.Activation("softmax")(x))
    model.load_weights(os.path.join(MODELS_DIR, "mini_xception_48.weights.h5"))
    return model


def by_type(keras, kind):
    """Layers of one kind in creation order. keras.layers is in graph order (the residual branch
    comes after the separable ones), but the automatic names (conv2d, conv2d_1, ...) follow the
    creation order of keras_model()."""
    layers = [c for c in keras.layers if type(c).__name__ == kind]
    number = lambda c: int(c.name.rsplit("_", 1)[1]) if c.name.rsplit("_", 1)[-1].isdigit() else 0
    return sorted(layers, key=number)


def copy(keras, torch_model):
    """Assigns each Keras layer to its PyTorch module, by kind and in creation order."""
    b = torch_model.blocks
    pairs = (list(zip(by_type(keras, "Conv2D"), [torch_model.conv1, torch_model.conv2]
                      + [x.residual for x in b] + [torch_model.final]))
             + list(zip(by_type(keras, "SeparableConv2D"), [s for x in b for s in (x.sep1, x.sep2)]))
             + list(zip(by_type(keras, "BatchNormalization"), [torch_model.bn1, torch_model.bn2]
                        + [n for x in b for n in (x.bn_residual, x.bn1, x.bn2)])))
    assert len(pairs) == len([c for c in keras.layers if c.get_weights()]) == 29, len(pairs)
    sources, destinations = zip(*pairs)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    for layer, dst in zip(sources, destinations):
        w = layer.get_weights()
        kind = type(layer).__name__
        if kind == "BatchNormalization":
            gamma, beta, mean, var = w
            dst.weight.data, dst.bias.data = t(gamma), t(beta)
            dst.running_mean.data, dst.running_var.data = t(mean), t(var)
        elif kind == "SeparableConv2D":
            dst.depthwise.weight.data = t(w[0].transpose(2, 3, 0, 1))   # (k,k,in,1) -> (in,1,k,k)
            dst.pointwise.weight.data = t(w[1].transpose(3, 2, 0, 1))   # (1,1,in,out) -> (out,in,1,1)
        elif kind == "Conv2D":
            dst.weight.data = t(w[0].transpose(3, 2, 0, 1))             # (k,k,in,out) -> (out,in,k,k)
            if len(w) > 1:
                dst.bias.data = t(w[1])
        else:
            raise ValueError("unexpected layer: %s" % kind)


def main():
    p = argparse.ArgumentParser(description="Convert mini-Xception Keras -> PyTorch")
    p.add_argument("--examples", help="folder with example faces (e.g. TD4's assets/examples)")
    args = p.parse_args()

    keras = keras_model()
    net = MiniXception().eval()
    copy(keras, net)

    # Check: same probabilities with example faces and random images
    images = []
    if args.examples:
        for path in sorted(glob.glob(os.path.join(args.examples, "*.png"))):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images.append(cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA))
    rng = np.random.default_rng(311)
    images += list(rng.integers(0, 256, (16, 48, 48)).astype(np.uint8))
    batch = np.stack(images).astype(np.float32)[..., None] / 127.5 - 1.0
    p_keras = keras(batch, training=False).numpy()
    with torch.no_grad():
        p_torch = net(torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())).numpy()
    diff = float(np.abs(p_keras - p_torch).max())
    same = int((p_keras.argmax(1) == p_torch.argmax(1)).sum())
    print("Check with %d images: max. difference %.2e, same class in %d/%d"
          % (len(batch), diff, same, len(batch)))
    if diff > 1e-4 or same != len(batch):
        sys.exit("[error] the conversion does not match Keras: not saved")

    out = os.path.join(MODELS_DIR, "mini_xception_48.pt")
    torch.save(net.state_dict(), out)
    print("Saved %s" % out)


if __name__ == "__main__":
    main()
