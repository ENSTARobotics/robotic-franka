#!/usr/bin/env python3
"""Operator's stress from the face (webcam) -> UDP to the supervisor. Runs on the host.

    python3 face.py --show                    # C920 webcam, with a preview window
    python3 face.py --camera C930e --show     # another camera, by name or by /dev/videoN number
    python3 face.py --image photo.jpg         # test with a still image (no camera)
    python3 face.py --camera 0 --share-frames # laptop camera, also lending its images to hands.py

YuNet detector (OpenCV) + mini-Xception (FER-2013, 7 expressions) from Joab's ROB311 lab (TD4,
"face recognition": 61% accuracy on PrivateTest), ported to PyTorch and run on the GPU with
CUDA (--cpu to force the CPU).
Sends to UDP 5010 about 10 times per second:
    {"t": .., "source": "face", "stress": 0..1 or null, "confidence": 0..1,
     "label": "Fear", "probs": {"Angry": .., ...}, "face": true}
With no visible face: stress null and confidence 0 (the supervisor ignores it in the fusion).
It also sends the annotated image (JPEG) to UDP 5012 for screen.py. With --share-frames it sends the plain
image to UDP 5019 too, for hands.py --frames-from-face: a camera can only be opened by one program, and
with a single camera (the laptop's, in simulation) both need it.

Note: a facial expression is not a reliable measure of the inner emotion; FER-2013 has
low-resolution photos and unbalanced classes. That is why the confidence is sent too.
"""
import argparse
import glob
import json
import math
import os
import socket
import sys
import time

# Numerical libraries use every core by default and saturate the CPU: without an RT kernel that
# causes communication_constraints_violation on the robot. It must come before importing numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import cv2
import numpy as np

cv2.setNumThreads(2)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IP = "127.0.0.1"
PERCEPTION_PORT = 5010
IMAGE_PORT = 5012           # annotated image for screen.py
FRAMES_PORT = 5019          # plain image for hands.py (--share-frames)
IMAGE_WIDTH = 640
CAMERA = "C920"             # the C930e is for the hands
HZ = 10.0

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# How much each expression adds to the stress (0 = nothing, 1 = everything). Tunable after the pilot.
STRESS_WEIGHT = {"Angry": 1.0, "Disgust": 1.0, "Fear": 1.0, "Sad": 0.6,
                 "Surprise": 0.5, "Happy": 0.0, "Neutral": 0.0}
DETECTION_THRESHOLD = 0.75  # minimum YuNet confidence (the same as in TD4)


class Detector(object):
    """YuNet; returns the largest face as (x, y, w, h) or None."""

    def __init__(self):
        self.yunet = cv2.FaceDetectorYN.create(
            os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx"), "",
            (320, 320), DETECTION_THRESHOLD, 0.3, 5000)

    def largest_face(self, bgr):
        height, width = bgr.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))
        small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
        self.yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.yunet.detect(small)
        if faces is None or not len(faces):
            return None
        x, y, w, h = max(faces[:, :4], key=lambda c: c[2] * c[3]) / scale
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 20 and y1 - y0 >= 20 else None


class Expression(object):
    """mini-Xception 48x48 in PyTorch (mini_xception.py), on the GPU if there is CUDA."""

    def __init__(self, device):
        import torch
        from mini_xception import load
        self.torch = torch
        self.device = device
        self.net = load(os.path.join(MODELS_DIR, "mini_xception_48.pt"), device)
        self.probabilities(np.zeros((48, 48, 3), np.uint8))     # warm up

    def probabilities(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = self.torch.from_numpy(gray / 127.5 - 1.0)[None, None].to(self.device)
        with self.torch.no_grad():
            return self.net(x)[0].cpu().numpy()


def stress_and_confidence(p):
    """Stress = weighted sum of negative expressions; confidence = 1 - normalized entropy."""
    stress = float(sum(STRESS_WEIGHT[e] * float(pi) for e, pi in zip(LABELS, p)))
    entropy = -sum(float(pi) * math.log(float(pi)) for pi in p if pi > 1e-9)
    return min(1.0, max(0.0, stress)), max(0.0, 1.0 - entropy / math.log(len(LABELS)))


def find_camera(camera):
    """/dev/videoN number of the camera: by number, or by part of its name (it does not change when
    it is plugged in again). Each webcam creates several nodes; the image one has index 0."""
    if camera.isdigit():
        return int(camera)
    available = {}
    for path in glob.glob("/sys/class/video4linux/video*"):
        with open(os.path.join(path, "index")) as f_index, open(os.path.join(path, "name")) as f_name:
            if f_index.read().strip() == "0":
                available[f_name.read().strip()] = int(os.path.basename(path)[5:])
    for name, number in sorted(available.items()):
        if camera.lower() in name.lower():
            return number
    sys.exit("[error] camera '%s' not found. Available: %s. Use --camera <name or number>."
             % (camera, ", ".join("%s (%d)" % d for d in sorted(available.items())) or "none"))


def annotate(bgr, box, message):
    if box is not None:
        x, y, w, h = box
        color = (0, 0, 255) if message["stress"] > 0.5 else (0, 200, 0)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(bgr, "%s  stress %.2f  conf %.2f" % (message["label"], message["stress"],
                    message["confidence"]), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(bgr, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return bgr


def send_image(sock, bgr, port=IMAGE_PORT):
    """Image for screen.py or hands.py (JPEG 640 px wide, fits in a UDP datagram)."""
    scale = IMAGE_WIDTH / float(bgr.shape[1])
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
    for quality in (70, 50, 30):
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(jpg) < 60000:
            sock.sendto(jpg.tobytes(), (IP, port))
            return


def main():
    p = argparse.ArgumentParser(description="Stress from the facial expression -> UDP to the supervisor")
    p.add_argument("--camera", default=CAMERA, help="name or number of the webcam (default %s)" % CAMERA)
    p.add_argument("--image", help="use a still image instead of the camera (tests)")
    p.add_argument("--show", action="store_true", help="window with the face and the expression")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--cpu", action="store_true", help="use the CPU even if there is a GPU")
    p.add_argument("--share-frames", action="store_true",
                   help="also send the plain image to UDP %d for hands.py --frames-from-face" % FRAMES_PORT)
    args = p.parse_args()

    import torch
    torch.set_num_threads(1)
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("[error] PyTorch does not see the GPU (CUDA). Check the NVIDIA driver or use --cpu.")
    detector = Detector()
    net = Expression(device)
    print("mini-Xception on %s" % (torch.cuda.get_device_name(0) if device == "cuda" else "CPU"))
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit("[error] cannot read %s" % args.image)
        camera = None
    else:
        number = find_camera(args.camera)
        camera = cv2.VideoCapture(number, cv2.CAP_V4L2)
        if not camera.isOpened():
            sys.exit("[error] cannot open /dev/video%d: is another program using it?" % number)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Sending to %s:%d at %.0f Hz. Ctrl+C to quit." % (IP, PERCEPTION_PORT, args.hz))

    period, t_print = 1.0 / args.hz, 0.0
    try:
        while True:
            t_tick = time.time()
            if camera is not None:
                ok, bgr = camera.read()
                if not ok:
                    sys.exit("[error] the camera stopped sending images")
            else:
                bgr = still.copy()

            if args.share_frames:
                send_image(sock, bgr, FRAMES_PORT)      # before annotating it
            box = detector.largest_face(bgr)
            message = {"t": time.time(), "source": "face", "stress": None, "confidence": 0.0,
                       "label": "", "probs": {}, "face": box is not None}
            if box is not None:
                x, y, w, h = box
                probs = net.probabilities(bgr[y:y + h, x:x + w])
                message["stress"], message["confidence"] = stress_and_confidence(probs)
                message["label"] = LABELS[int(np.argmax(probs))]
                message["probs"] = {e: round(float(pi), 3) for e, pi in zip(LABELS, probs)}
            sock.sendto(json.dumps(message).encode("utf-8"), (IP, PERCEPTION_PORT))

            if t_tick - t_print >= 0.5:
                t_print = t_tick
                if box is None:
                    sys.stdout.write("\r\033[Kno face")
                else:
                    sys.stdout.write("\r\033[K%-8s stress %.2f  confidence %.2f"
                                     % (message["label"], message["stress"], message["confidence"]))
                sys.stdout.flush()
            annotated = annotate(bgr, box, message)
            send_image(sock, annotated)
            if args.show:
                cv2.imshow("face (q to quit)", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, period - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("\nface closed")


if __name__ == "__main__":
    main()
