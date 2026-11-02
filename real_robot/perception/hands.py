#!/usr/bin/env python3
"""Are there hands on the PCB, next to the robot? Hand YOLO on the table camera.

    python3 hands.py                # tells the supervisor over UDP 5016 and dialogue.py (PCB) over 5015
    python3 hands.py --show         # the same, with a window: watched zone and detected hands
    python3 hands.py --point-radius-cm 4   # stop at 4 cm from a soldering point instead of 3
    python3 hands.py --frames-from-face --no-markers   # simulation: laptop camera via face.py, whole image

Uses the same camera as pcb.py (the one that sees the 4 ArUco markers), and a camera can only be
opened by one program: that is why this one also does the PCB check and sends the same as pcb.py.
The operator holds the PCB, so hands on its edges are fine: the watched zone is a circle of
--point-radius-cm around each soldering point (the same 2x3 grid as pcb_plan.py), placed in the image
through the markers. A hand counts if its box touches one of those circles.

Sends {"t": .., "near": true|false|null, "hands": n, "confidence": 0..1}; null = it does not see the
markers and does not know where the PCB is. "near" turns on with 2 of the last 3 images and turns off
after 1 s without hands in the zone: warn fast, release slowly.

Simulation without the table (--no-markers): any hand in the whole image counts as near, and there is no
PCB check. With --frames-from-face it does not open a camera: it takes the images face.py --share-frames sends
to UDP 5019, so both can use the laptop's only camera.

Model: models/hands_yolov8n.pt (hand YOLOv8n, see models/HANDS_YOLO.txt).
"""
import argparse
import collections
import json
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "robot"))
import pcb_plan
import time

# Ultralytics sends usage statistics by default: not here
os.environ.setdefault("YOLO_OFFLINE", "1")

from face import find_camera         # limits the threads before importing numpy  # noqa: F401
from pcb import Zone, PcbDetector, open_camera, IP, PCB_PORT

import cv2
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(DIR, "models", "hands_yolov8n.pt")
HANDS_PORT = 5016
FRAMES_PORT = 5019          # images from face.py --share-frames
FRAMES_TIMEOUT = 1.0        # s without images from face.py -> "near" unknown
CAMERA = "C930e"            # the one that sees the table and the 4 markers, like pcb.py
HZ = 10.0
PCB_HZ = 2.0
POINT_RADIUS_CM = 3.0       # around each soldering point (the PCB edges stay free to hold it)
MIN_CONF = 0.4              # minimum YOLO confidence
IMGSZ = 640
WINDOW = 3                  # "near" if there are hands in the zone in ...
MIN_DETECTIONS = 2          # ... at least these of the last WINDOW images
RELEASE_S = 1.0             # s in a row without hands in the zone to stop warning


def watched_circles(radius, sides=16):
    """A polygon of `radius` m around each soldering point, in the robot base frame (table plane)."""
    _, info = pcb_plan.load()
    E = [info["corner_%d" % i]["tip_xyz"] for i in range(1, 5)]
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return [np.array([[p[0] + radius * np.cos(a), p[1] + radius * np.sin(a)] for a in angles], np.float32)
            for p in pcb_plan.grid_points(E, 2, 3, pcb_plan.MARGIN)]


def touches_zone(box, zone_img):
    """True if the box (x1, y1, x2, y2) in pixels overlaps the zone's polygon."""
    x1, y1, x2, y2 = box
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    area, _ = cv2.intersectConvexConvex(rect, zone_img)
    return area > 0.0


class Watch(object):
    """Turns per-image detections into a stable "near"."""

    def __init__(self):
        self.latest = collections.deque(maxlen=WINDOW)
        self.near = False
        self.t_last = None          # last image with hands in the zone

    def update(self, seen, t):
        self.latest.append(seen)
        if seen:
            self.t_last = t
        if sum(self.latest) >= MIN_DETECTIONS:
            self.near = True
        elif self.near and (self.t_last is None or t - self.t_last >= RELEASE_S):
            self.near = False
        return self.near


class FaceFrames(object):
    """Latest image sent by face.py --share-frames (UDP 5019), instead of opening the camera."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((IP, FRAMES_PORT))
        self.sock.settimeout(FRAMES_TIMEOUT)

    def read(self):
        """(ok, bgr) with the newest image; ok False if none arrived in FRAMES_TIMEOUT."""
        try:
            data, _ = self.sock.recvfrom(65536)
        except socket.timeout:
            return False, None
        self.sock.setblocking(False)
        try:
            while True:                     # keep only the newest one
                data, _ = self.sock.recvfrom(65536)
        except (BlockingIOError, socket.error):
            pass
        finally:
            self.sock.settimeout(FRAMES_TIMEOUT)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return bgr is not None, bgr

    def release(self):
        self.sock.close()


def main():
    p = argparse.ArgumentParser(description="Detects hands in the PCB zone")
    p.add_argument("--show", action="store_true", help="window with the zone and the hands")
    p.add_argument("--camera", default=CAMERA, help="name or number of the webcam (default %s)" % CAMERA)
    p.add_argument("--point-radius-cm", type=float, default=POINT_RADIUS_CM,
                   help="cm watched around each soldering point (default %.0f)" % POINT_RADIUS_CM)
    p.add_argument("--conf", type=float, default=MIN_CONF, help="minimum YOLO confidence (0..1)")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLO input size")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--no-pcb", action="store_true", help="do not check the PCB (if pcb.py uses another camera)")
    p.add_argument("--no-markers", action="store_true",
                   help="simulation without the table: the whole image is the zone, no PCB check")
    p.add_argument("--frames-from-face", action="store_true",
                   help="take the images from face.py --share-frames (UDP %d) instead of a camera" % FRAMES_PORT)
    p.add_argument("--cpu", action="store_true", help="use the CPU even if there is a GPU")
    args = p.parse_args()

    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    torch.set_num_threads(1)
    device = 0 if torch.cuda.is_available() and not args.cpu else "cpu"
    model = YOLO(MODEL)
    print("Hand YOLO on %s" % (torch.cuda.get_device_name(0) if device == 0 else "CPU"))

    zone = None if args.no_markers else Zone()
    pcb = None if args.no_pcb or args.no_markers else PcbDetector(zone)
    circles = None if args.no_markers else watched_circles(args.point_radius_cm / 100.0)
    watch = Watch()
    camera = FaceFrames() if args.frames_from_face else open_camera(args.camera)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Watching %s. Hands to %s:%d%s. Images from %s. Ctrl+C to quit."
          % ("the whole image (no markers)" if args.no_markers else
             "%.0f cm around each soldering point" % args.point_radius_cm,
             IP, HANDS_PORT, "" if pcb is None else ", PCB to %d" % PCB_PORT,
             "face.py (UDP %d)" % FRAMES_PORT if args.frames_from_face else "camera " + args.camera))
    t_pcb = 0.0
    try:
        while True:
            t_tick = time.time()
            if args.frames_from_face:
                ok, bgr = camera.read()
                if not ok:              # face.py is not sending: say it is unknown and keep waiting
                    sock.sendto(json.dumps({"t": time.time(), "near": None, "hands": 0, "confidence": 0.0}).encode(
                        "utf-8"), (IP, HANDS_PORT))
                    sys.stdout.write("\r\033[Kno images from face.py (is it running with --share-frames?)")
                    sys.stdout.flush()
                    continue
            else:
                for _ in range(2):      # throw away the image left in the buffer
                    ok, bgr = camera.read()
                if not ok:
                    sys.exit("[error] the camera stopped sending images")
            seen = 0 if zone is None else zone.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

            result = model(bgr, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            boxes = [(c.xyxy[0].tolist(), float(c.conf[0])) for c in result.boxes]
            zones_img, in_zone = None, []
            if zone is None:
                h, w = bgr.shape[:2]
                zones_img = [np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)]
            elif zone.homography is not None:
                zones_img = [cv2.perspectiveTransform(c[None], zone.homography)[0] for c in circles]
            if zones_img is not None:
                in_zone = [conf for box, conf in boxes if any(touches_zone(box, z) for z in zones_img)]
                near = watch.update(bool(in_zone), t_tick)
            else:
                near = None
            sock.sendto(json.dumps({"t": time.time(), "near": near, "hands": len(in_zone),
                                    "confidence": round(max(in_zone), 3) if in_zone else 0.0}).encode("utf-8"),
                        (IP, HANDS_PORT))

            if pcb is not None and t_tick - t_pcb >= 1.0 / PCB_HZ:
                t_pcb = t_tick
                pcb.update(bgr, t_tick)
                sock.sendto(json.dumps(pcb.message()).encode("utf-8"), (IP, PCB_PORT))

            sys.stdout.write("\r\033[K%s  hands seen %d, in the zone %d%s%s"
                             % ({True: "HANDS NEAR", False: "zone clear", None: "markers not seen"}[near],
                                len(boxes), len(in_zone), "" if zone is None else "  markers %d/4" % seen,
                                "" if pcb is None else "  | " + pcb.text()))
            sys.stdout.flush()
            if args.show:
                if zones_img is not None:
                    cv2.polylines(bgr, [z.astype(np.int32) for z in zones_img], True,
                                  (0, 0, 255) if near else (0, 200, 0), 2)
                for (x1, y1, x2, y2), conf in boxes:
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                    cv2.putText(bgr, "%.2f" % conf, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 255), 2)
                cv2.imshow("hands (q to quit)", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
