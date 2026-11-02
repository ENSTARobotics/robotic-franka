#!/usr/bin/env python3
"""Are there hands on the PCB, next to the robot? Hand YOLO on the table camera.

    python3 hands.py                # tells the supervisor over UDP 5016 and dialogue.py (PCB) over 5015
    python3 hands.py --show         # the same, with a window: watched zone and detected hands
    python3 hands.py --margin-cm 8  # watch 8 cm around the PCB instead of 5

Uses the same camera as pcb.py (the one that sees the 4 ArUco markers), and a camera can only be
opened by one program: that is why this one also does the PCB check and sends the same as pcb.py.
The markers say where the PCB's recorded corners fall in the image; the watched zone is the PCB
enlarged by --margin-cm. A hand counts if its box touches that zone. The hand is above the table and
the camera sees it in perspective: the margin covers that error.

Sends {"t": .., "near": true|false|null, "hands": n, "confidence": 0..1}; null = it does not see the
markers and does not know where the PCB is. "near" turns on with 2 of the last 3 images and turns off
after 1 s without hands in the zone: warn fast, release slowly.

Model: models/hands_yolov8n.pt (hand YOLOv8n, see models/HANDS_YOLO.txt).
"""
import argparse
import collections
import json
import os
import socket
import sys
import time

# Ultralytics sends usage statistics by default: not here
os.environ.setdefault("YOLO_OFFLINE", "1")

from face import find_camera         # limits the threads before importing numpy  # noqa: F401
from pcb import Zone, PcbDetector, open_camera, load_table, IP, PCB_PORT

import cv2
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(DIR, "models", "hands_yolov8n.pt")
HANDS_PORT = 5016
CAMERA = "C930e"            # the one that sees the table and the 4 markers, like pcb.py
HZ = 10.0
PCB_HZ = 2.0
MARGIN_CM = 5.0             # around the PCB
MIN_CONF = 0.4              # minimum YOLO confidence
IMGSZ = 640
WINDOW = 3                  # "near" if there are hands in the zone in ...
MIN_DETECTIONS = 2          # ... at least these of the last WINDOW images
RELEASE_S = 1.0             # s in a row without hands in the zone to stop warning


def watched_zone(margin):
    """The PCB's corners enlarged by `margin` m, in the robot base frame (like Zone.table)."""
    _, e = load_table()
    length_u, length_v = np.linalg.norm(e[1] - e[0]), np.linalg.norm(e[3] - e[0])
    au, av = margin / length_u, margin / length_v

    def point(u, v):
        return (1 - u) * (1 - v) * e[0] + u * (1 - v) * e[1] + u * v * e[2] + (1 - u) * v * e[3]

    return np.array([point(-au, -av), point(1 + au, -av), point(1 + au, 1 + av), point(-au, 1 + av)], np.float32)


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


def main():
    p = argparse.ArgumentParser(description="Detects hands in the PCB zone")
    p.add_argument("--show", action="store_true", help="window with the zone and the hands")
    p.add_argument("--camera", default=CAMERA, help="name or number of the webcam (default %s)" % CAMERA)
    p.add_argument("--margin-cm", type=float, default=MARGIN_CM, help="cm watched around the PCB")
    p.add_argument("--conf", type=float, default=MIN_CONF, help="minimum YOLO confidence (0..1)")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLO input size")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--no-pcb", action="store_true", help="do not check the PCB (if pcb.py uses another camera)")
    p.add_argument("--cpu", action="store_true", help="use the CPU even if there is a GPU")
    args = p.parse_args()

    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    torch.set_num_threads(1)
    device = 0 if torch.cuda.is_available() and not args.cpu else "cpu"
    model = YOLO(MODEL)
    print("Hand YOLO on %s" % (torch.cuda.get_device_name(0) if device == 0 else "CPU"))

    zone = Zone()
    pcb = None if args.no_pcb else PcbDetector(zone)
    zone_table = watched_zone(args.margin_cm / 100.0)
    watch = Watch()
    camera = open_camera(args.camera)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Watching %.0f cm around the PCB. Hands to %s:%d%s. Ctrl+C to quit."
          % (args.margin_cm, IP, HANDS_PORT, "" if args.no_pcb else ", PCB to %d" % PCB_PORT))
    t_pcb = 0.0
    try:
        while True:
            t_tick = time.time()
            for _ in range(2):          # throw away the image left in the buffer
                ok, bgr = camera.read()
            if not ok:
                sys.exit("[error] the camera stopped sending images")
            seen = zone.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

            result = model(bgr, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            boxes = [(c.xyxy[0].tolist(), float(c.conf[0])) for c in result.boxes]
            zone_img, in_zone = None, []
            if zone.homography is not None:
                zone_img = cv2.perspectiveTransform(zone_table[None], zone.homography)[0]
                in_zone = [conf for box, conf in boxes if touches_zone(box, zone_img)]
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

            sys.stdout.write("\r\033[K%s  hands seen %d, in the zone %d  markers %d/4%s"
                             % ({True: "HANDS NEAR", False: "zone clear", None: "markers not seen"}[near],
                                len(boxes), len(in_zone), seen, "" if pcb is None else "  | " + pcb.text()))
            sys.stdout.flush()
            if args.show:
                if zone_img is not None:
                    cv2.polylines(bgr, [zone_img.astype(np.int32)], True, (0, 0, 255) if near else (0, 200, 0), 2)
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
