#!/usr/bin/env python3
"""Downloads dialogue.py's speech models into dialogue/models/ (outside git, ~100 MB).

    python3 download_models.py                 # lessac voice (female) + English Vosk
    python3 download_models.py --voice ryan    # male voice

Piper (speaking): https://huggingface.co/rhasspy/piper-voices (MIT / each voice's license)
Vosk (listening): https://alphacephei.com/vosk/models (Apache 2.0)
"""
import argparse
import os
import urllib.request
import zipfile

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/%s/medium/en_US-%s-medium.onnx"
VOICES = ("lessac", "ryan", "amy")
VOSK = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def download(url, destination):
    if os.path.exists(destination):
        print("%s already exists" % os.path.basename(destination))
        return
    print("downloading %s ..." % os.path.basename(destination))
    urllib.request.urlretrieve(url, destination + ".partial")
    os.replace(destination + ".partial", destination)


def main():
    p = argparse.ArgumentParser(description="Downloads dialogue.py's models")
    p.add_argument("--voice", choices=VOICES, default="lessac")
    args = p.parse_args()
    os.makedirs(DIR, exist_ok=True)

    base = PIPER % (args.voice, args.voice)
    download(base, os.path.join(DIR, os.path.basename(base)))
    download(base + ".json", os.path.join(DIR, os.path.basename(base) + ".json"))

    folder = os.path.join(DIR, "vosk-model-small-en-us-0.15")
    if not os.path.isdir(folder):
        zip_path = os.path.join(DIR, "vosk.zip")
        download(VOSK, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DIR)
        os.remove(zip_path)
    print("done: %s" % DIR)


if __name__ == "__main__":
    main()
