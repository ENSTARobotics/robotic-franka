#!/usr/bin/env python3
"""Operator's stress from the voice (microphone) -> UDP to the supervisor. Runs on the host.

    python3 voice.py                      # microphone of the C930e (the hands camera)
    python3 voice.py --device C920        # another microphone, by part of its name
    python3 voice.py --device system      # Ubuntu's default one
    python3 voice.py --list               # see the available microphones
    python3 voice.py --wav test.wav       # test with a file (no microphone)

"Scratch" CNN from the voice-emotion-recognition project (Jair and Lucas, Learning for
robotics course), retrained on English RAVDESS (entrenar_ravdess.py in that repo), on the GPU.
Every 0.1 s it analyzes the last 3 s (log-mel 64 x 301, as in training) and sends to UDP 5010:
    {"t": .., "source": "audio", "stress": 0..1 or null, "confidence": 0..1,
     "label": "fearful", "probs": {..}, "voice": 0.8}
"voice" is the fraction of the window with speech. In silence it repeats the last result with
"held": true and its confidence going down to 0 in 5 s: it is still shown, but it stops counting.
The first 3 s it measures the background noise: stay silent.

Note: RAVDESS is actors reading two English sentences in a studio; with spontaneous speech,
another language and the lab's noise the accuracy will be lower. That is why the confidence is sent.
"""
import argparse
import collections
import json
import math
import os
import queue
import socket
import sys
import time

# Numerical libraries use every core by default and saturate the CPU: without an RT kernel that
# causes communication_constraints_violation on the robot. It must come before importing numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

import microphone

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IP = "127.0.0.1"
PERCEPTION_PORT = 5010
HZ = 10.0

# Same parameters as in training (EmotionDetection.ipynb / entrenar_ravdess.py)
SR, N_SAMPLES, WIN, HOP, N_FFT, N_MELS = 16000, 3 * 16000, 400, 160, 512, 64
# How much each emotion adds to the stress: the same weights as the face (FER-2013)
STRESS_WEIGHT = {"angry": 1.0, "disgust": 1.0, "fearful": 1.0, "sad": 0.6,
                 "surprised": 0.5, "happy": 0.0, "neutral": 0.0}
STARTUP_S = 1.0             # first seconds that are thrown away (zeros or a transient when opening the mic)
CALIBRATION_S = 2.0         # following seconds to measure the background noise
VOICE_MARGIN_DB = 6.0       # a frame is speech if it is this many dB above the background noise
MIN_VOICE = 0.10            # minimum fraction of speech in the window (0.3 s of 3 s) to classify
HIGH_NOISE_DB = -30.0       # a silence above this is a very noisy microphone
MIN_NOISE_DB = -100.0       # below this it is zeros: no audio is arriving
HOLD_S = 5.0                # s of silence in which the last result loses its weight
NOISE_WINDOW_S = 15.0       # the background noise is estimated again from the last seconds


def conv_block(cin, cout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


def load_net(path, device):
    """The notebook's ScratchCNN (identical architecture) with the RAVDESS checkpoint."""
    import torch
    import torch.nn as nn

    class ScratchCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                          conv_block(64, 128), conv_block(128, 256))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ScratchCNN(len(ck["classes"]))
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck["mu"], ck["sd"], ck["classes"]


def logmel(x):
    """Same as wav_to_logmel in training: peak at 1, the central 3 s, log-mel 64 x 301."""
    import librosa
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)))
    else:
        s = (len(x) - N_SAMPLES) // 2
        x = x[s:s + N_SAMPLES]
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)


def energy_db(x, frame=WIN):
    """Energy in dB of 25 ms frames."""
    n = len(x) // frame
    if n == 0:
        return np.array([-120.0])
    frames = x[:n * frame].reshape(n, frame)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


class NoiseFloor(object):
    """Background noise: 10th percentile of the recent energy. It follows the microphone if its level
    changes (it is usually higher right after opening it), and the pauses in speech are enough not to
    mistake it for speech."""

    def __init__(self):
        self.frames = collections.deque(maxlen=int(NOISE_WINDOW_S * SR / WIN))

    def add(self, x16k):
        self.frames.extend(float(e) for e in energy_db(x16k - x16k.mean()) if e > MIN_NOISE_DB)

    def value(self):
        return float(np.percentile(self.frames, 10)) if self.frames else None


def stress_and_confidence(p, classes, voice_fraction):
    stress = float(sum(STRESS_WEIGHT.get(c, 0.0) * float(pi) for c, pi in zip(classes, p)))
    entropy = -sum(float(pi) * math.log(float(pi)) for pi in p if pi > 1e-9)
    sharpness = max(0.0, 1.0 - entropy / math.log(len(classes)))
    # Less speech in the window = less reliable
    return min(1.0, max(0.0, stress)), sharpness * min(1.0, voice_fraction / 0.5)


class Classifier(object):
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = device
        self.net, self.mu, self.sd, self.classes = load_net(
            os.path.join(MODELS_DIR, "voice_ravdess_scratch.pt"), device)
        self.probabilities(np.zeros(N_SAMPLES, np.float32))      # warm up

    def probabilities(self, x16k):
        m = (logmel(x16k) - self.mu) / self.sd
        t = self.torch.from_numpy(m)[None, None].to(self.device)
        with self.torch.no_grad():
            return self.torch.softmax(self.net(t), dim=1)[0].cpu().numpy()


def analyze(x16k, noise_db, clf):
    """Message for the supervisor from the last 3 s at 16 kHz."""
    x16k = x16k - x16k.mean()       # no DC component (some mics have it when they are opened)
    fraction = float(np.mean(energy_db(x16k) > noise_db + VOICE_MARGIN_DB))
    message = {"t": time.time(), "source": "audio", "stress": None, "confidence": 0.0,
               "label": "", "probs": {}, "voice": round(fraction, 2)}
    if fraction >= MIN_VOICE:
        p = clf.probabilities(x16k)
        message["stress"], message["confidence"] = stress_and_confidence(p, clf.classes, fraction)
        message["label"] = clf.classes[int(np.argmax(p))]
        message["probs"] = {c: round(float(pi), 3) for c, pi in zip(clf.classes, p)}
    return message


def hold(message, last, t_last, now):
    """In silence, repeats the last result with the confidence going down to 0 in HOLD_S."""
    rest = max(0.0, 1.0 - (now - t_last) / HOLD_S)
    message.update(stress=last["stress"], label=last["label"], probs=last["probs"],
                   confidence=last["confidence"] * rest, held=True)
    return message


def show(message):
    if message["stress"] is None:
        sys.stdout.write("\r\033[Ksilence (voice %.0f%%)" % (100 * message["voice"]))
    elif message.get("held"):
        sys.stdout.write("\r\033[Ksilence; last: %-9s stress %.2f  confidence %.2f"
                         % (message["label"], message["stress"], message["confidence"]))
    else:
        sys.stdout.write("\r\033[K%-9s stress %.2f  confidence %.2f  (voice %.0f%%)"
                         % (message["label"], message["stress"], message["confidence"], 100 * message["voice"]))
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(description="Stress from the voice -> UDP to the supervisor")
    p.add_argument("--device", default=microphone.MICROPHONE,
                   help="microphone: part of its name, or 'system' (default %s)" % microphone.MICROPHONE)
    p.add_argument("--list", action="store_true", help="list the microphones and quit")
    p.add_argument("--wav", help="analyze a file instead of the microphone (tests)")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--cpu", action="store_true", help="use the CPU even if there is a GPU")
    args = p.parse_args()

    if args.list:
        microphone.list_microphones()
        return
    import sounddevice as sd
    import librosa
    import torch
    torch.set_num_threads(1)
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("[error] PyTorch does not see the GPU (CUDA). Check the NVIDIA driver or use --cpu.")
    clf = Classifier(device)
    print("Voice CNN (RAVDESS, %d classes) on %s"
          % (len(clf.classes), torch.cuda.get_device_name(0) if device == "cuda" else "CPU"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.wav:
        x, _ = librosa.load(args.wav, sr=SR, mono=True)
        noise = float(np.percentile(energy_db(x), 10))
        message = analyze(x, noise, clf)
        sock.sendto(json.dumps(message).encode("utf-8"), (IP, PERCEPTION_PORT))
        show(message)
        print("\nprobabilities: %s" % message["probs"])
        return

    mic = microphone.Microphone(args.device)
    sr_mic = int(sd.query_devices(mic.device, "input")["default_samplerate"])
    buffer = np.zeros(int(3 * sr_mic), np.float32)         # last 3 s at the mic's sample rate
    blocks = queue.Queue()                                 # the callback does not lose audio
    lost = []

    def receive(data, frames, timestamp, status):
        if status.input_overflow:
            lost.append(frames)
        blocks.put(data[:, 0].copy())

    print("Microphone in use: %s (%d Hz)" % (mic.description, sr_mic))
    print("Silence for %.0f s to measure the background noise..." % (STARTUP_S + CALIBRATION_S))
    t_start = time.time()
    floor, calibrated = NoiseFloor(), False
    last, t_last = None, 0.0

    with mic.open(lambda: sd.InputStream(device=mic.device, channels=1, samplerate=sr_mic,
                                         dtype="float32", callback=receive)):
        print("Sending to %s:%d at %.0f Hz. Ctrl+C to quit." % (IP, PERCEPTION_PORT, args.hz))
        try:
            while True:
                t_tick = time.time()
                new = [blocks.get_nowait() for _ in range(blocks.qsize())]
                if new:
                    new = np.concatenate(new)
                    buffer = np.concatenate([buffer, new])[-len(buffer):]
                x16k = librosa.resample(buffer, orig_sr=sr_mic, target_sr=SR)

                # The first second is thrown away: zeros or a transient arrive when a microphone is opened
                if len(new) and t_tick - t_start >= STARTUP_S:
                    floor.add(librosa.resample(new, orig_sr=sr_mic, target_sr=SR))

                if not calibrated:
                    if t_tick - t_start >= STARTUP_S + CALIBRATION_S:
                        if floor.value() is None:
                            sys.exit("[error] no audio is arriving from the microphone. Pick another one in "
                                     "Settings -> Sound (or with --device, see --list).")
                        calibrated = True
                        print("background noise: %.1f dB. You can talk now." % floor.value())
                        if floor.value() > HIGH_NOISE_DB:
                            print("[warning] very noisy microphone: speech is harder to tell apart. Better a closer one.")
                    message = {"t": time.time(), "source": "audio", "stress": None, "confidence": 0.0,
                               "label": "", "probs": {}, "voice": 0.0}
                else:
                    message = analyze(x16k, floor.value(), clf)
                    if message["stress"] is not None:
                        last, t_last = message, t_tick
                    elif last is not None:
                        message = hold(message, last, t_last, t_tick)
                    show(message)
                if lost:
                    print("\n[warning] audio was lost (%d samples): the laptop is very loaded" % sum(lost))
                    del lost[:]
                sock.sendto(json.dumps(message).encode("utf-8"), (IP, PERCEPTION_PORT))
                time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t_tick)))
        except KeyboardInterrupt:
            print("\nvoice closed")


if __name__ == "__main__":
    main()
