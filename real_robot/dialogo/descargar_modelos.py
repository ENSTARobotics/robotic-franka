#!/usr/bin/env python3
"""Descarga los modelos de voz de dialogo.py en dialogo/modelos/ (fuera de git, ~100 MB).

    python3 descargar_modelos.py               # voz lessac (femenina) + Vosk inglés
    python3 descargar_modelos.py --voz ryan    # voz masculina

Piper (habla): https://huggingface.co/rhasspy/piper-voices (MIT / licencias de cada voz)
Vosk (escucha): https://alphacephei.com/vosk/models (Apache 2.0)
"""
import argparse
import os
import urllib.request
import zipfile

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/%s/medium/en_US-%s-medium.onnx"
VOCES = ("lessac", "ryan", "amy")
VOSK = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def bajar(url, destino):
    if os.path.exists(destino):
        print("ya existe %s" % os.path.basename(destino))
        return
    print("descargando %s ..." % os.path.basename(destino))
    urllib.request.urlretrieve(url, destino + ".parcial")
    os.replace(destino + ".parcial", destino)


def main():
    p = argparse.ArgumentParser(description="Descarga los modelos de dialogo.py")
    p.add_argument("--voz", choices=VOCES, default="lessac")
    args = p.parse_args()
    os.makedirs(DIR, exist_ok=True)

    base = PIPER % (args.voz, args.voz)
    bajar(base, os.path.join(DIR, os.path.basename(base)))
    bajar(base + ".json", os.path.join(DIR, os.path.basename(base) + ".json"))

    carpeta = os.path.join(DIR, "vosk-model-small-en-us-0.15")
    if not os.path.isdir(carpeta):
        zip_ = os.path.join(DIR, "vosk.zip")
        bajar(VOSK, zip_)
        with zipfile.ZipFile(zip_) as z:
            z.extractall(DIR)
        os.remove(zip_)
    print("listo: %s" % DIR)


if __name__ == "__main__":
    main()
