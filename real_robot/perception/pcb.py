#!/usr/bin/env python3
"""Is the PCB in its place? Looks for a rectangle of another color on the white base.

    python3 pcb.py            # tells dialogue.py over UDP 5015 twice per second
    python3 pcb.py --show     # the same, with a window that shows the zone and what it detects

The table's 4 ArUco markers (data/table.json) say where each point of the table falls in the image,
and the PCB's recorded corners (data/pose_info.json) say which zone to look at. Inside it, whatever is
not the color of the base is separated, and there is a PCB if it is a rectangular shape that covers the
zone. A hand does not count because it is not rectangular. It works with a notebook, a piece of
cardboard or a PCB, as long as they are not white.

Sends {"t": .., "present": true|false|null, "score": 0..1}; null = it does not see the markers.
"""
import argparse
import json
import os
import socket
import sys
import time

from face import find_camera         # limits the threads before importing numpy

import cv2
import numpy as np

cv2.setNumThreads(1)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
IP = "127.0.0.1"
PCB_PORT = 5015
HZ = 2.0
CAMERA = "C930e"            # the one that sees the table and the 4 markers
WIDTH, HEIGHT = 1280, 720
PX_PER_M = 1000.0           # resolution of the rectified zone
INNER_MARGIN = 0.015        # m ignored inside the edge of the PCB (the image is not exact)
OUTER_MARGIN = 0.04         # m of base around it: the color of the base is taken from there
COLOR_THRESHOLD = 30.0      # color distance (Lab) to the base color to say it is not base
MIN_COVERAGE = 0.60         # fraction of the zone the shape must cover
MIN_RECTANGLE = 0.80        # area of the shape / area of its bounding rectangle
STABLE_S = 1.5              # s a change of state must last


def load_table():
    with open(os.path.join(DATA_DIR, "table.json")) as f:
        table = json.load(f)
    with open(os.path.join(DATA_DIR, "pose_info.json")) as f:
        info = json.load(f)
    corners = np.array([info["corner_%d" % i]["tip_xyz"][:2] for i in range(1, 5)], np.float32)
    return table["arucos"], corners


class Zone(object):
    """Rectifies the PCB zone (with a ring of table around it) from the ArUco markers."""

    def __init__(self):
        self.arucos, e = load_table()
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                                cv2.aruco.DetectorParameters())
        self.homography = None
        length_u, length_v = np.linalg.norm(e[1] - e[0]), np.linalg.norm(e[3] - e[0])
        au, av = OUTER_MARGIN / length_u, OUTER_MARGIN / length_v
        # Enlarged corners: e[0..3] = u, v (0,0), (1,0), (1,1), (0,1) as in pcb_plan.bilinear

        def point(u, v):
            return (1 - u) * (1 - v) * e[0] + u * (1 - v) * e[1] + u * v * e[2] + (1 - u) * v * e[3]

        self.table = np.array([point(-au, -av), point(1 + au, -av), point(1 + au, 1 + av), point(-au, 1 + av)],
                              np.float32)
        self.width = int(round((length_u + 2 * OUTER_MARGIN) * PX_PER_M))
        self.height = int(round((length_v + 2 * OUTER_MARGIN) * PX_PER_M))
        self.inside = np.zeros((self.height, self.width), bool)
        edge = int((OUTER_MARGIN + INNER_MARGIN) * PX_PER_M)
        self.inside[edge:-edge, edge:-edge] = True
        self.ring = np.ones((self.height, self.width), bool)
        self.ring[int(OUTER_MARGIN * PX_PER_M):-int(OUTER_MARGIN * PX_PER_M),
                  int(OUTER_MARGIN * PX_PER_M):-int(OUTER_MARGIN * PX_PER_M)] = False

    def update(self, gray):
        """Computes the homography again if it sees the 4 markers; otherwise it keeps the last one."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        seen = {} if ids is None else {str(int(i)): c[0].mean(axis=0) for i, c in zip(ids.ravel(), corners)}
        if all(k in seen for k in self.arucos):
            origin = np.array([self.arucos[k] for k in self.arucos], np.float32)
            destination = np.array([seen[k] for k in self.arucos], np.float32)
            self.homography = cv2.findHomography(origin, destination)[0]
        return len(seen)

    def image_corners(self):
        return cv2.perspectiveTransform(self.table[None], self.homography)[0]

    def rectify(self, bgr):
        destination = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], np.float32)
        m = cv2.getPerspectiveTransform(self.image_corners(), destination)
        return cv2.GaussianBlur(cv2.warpPerspective(bgr, m, (self.width, self.height)), (0, 0), 3)

    def analyze(self, crop):
        """(coverage of the zone, how rectangular it is) of the largest shape that is not the base color."""
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        base = np.median(lab[self.ring], axis=0)
        shape = (np.linalg.norm(lab - base, axis=2) > COLOR_THRESHOLD).astype(np.uint8)
        shape = cv2.morphologyEx(shape, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0
        largest = max(contours, key=cv2.contourArea)
        width, height = cv2.minAreaRect(largest)[1]
        filled = np.zeros_like(shape)
        cv2.drawContours(filled, [largest], -1, 1, -1)
        coverage = float(filled[self.inside].mean())
        return coverage, float(cv2.contourArea(largest) / max(1.0, width * height))


class PcbDetector(object):
    """Stable state of the PCB: True / False / None (does not see the markers). hands.py reuses it."""

    def __init__(self, zone):
        self.zone = zone
        self.present, self.candidate, self.t_candidate = None, None, 0.0
        self.score, self.rectangle = None, 0.0

    def update(self, bgr, t):
        """With zone.update() already called on this image."""
        self.score, self.rectangle = None, 0.0
        if self.zone.homography is not None:
            self.score, self.rectangle = self.zone.analyze(self.zone.rectify(bgr))
            now = self.score >= MIN_COVERAGE and self.rectangle >= MIN_RECTANGLE
            if now != self.candidate:
                self.candidate, self.t_candidate = now, t
            if self.present is None or t - self.t_candidate >= STABLE_S:
                self.present = self.candidate
        return self.present

    def message(self):
        return {"t": time.time(), "present": self.present,
                "score": None if self.score is None else round(self.score, 3)}

    def text(self):
        return "%s  covers %s  rectangular %.0f%%" % (
            {True: "PCB present", False: "no PCB", None: "markers not seen"}[self.present],
            "--" if self.score is None else "%.0f%%" % (100 * self.score), 100 * self.rectangle)


def open_camera(name):
    camera = cv2.VideoCapture(find_camera(name), cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not camera.isOpened():
        sys.exit("[error] cannot open camera %s: is another program using it?" % name)
    return camera


def main():
    p = argparse.ArgumentParser(description="Detects whether the PCB is in its place")
    p.add_argument("--show", action="store_true", help="window with the zone and the score")
    p.add_argument("--camera", default=CAMERA, help="name or number of the webcam (default %s)" % CAMERA)
    args = p.parse_args()

    zone = Zone()
    pcb = PcbDetector(zone)
    camera = open_camera(args.camera)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Watching the PCB zone. Sending to %s:%d. Ctrl+C to quit." % (IP, PCB_PORT))
    try:
        while True:
            t_tick = time.time()
            for _ in range(2):          # throw away the image left in the buffer
                ok, bgr = camera.read()
            if not ok:
                sys.exit("[error] the camera stopped sending images")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            seen = zone.update(gray)
            present = pcb.update(bgr, t_tick)
            sock.sendto(json.dumps(pcb.message()).encode("utf-8"), (IP, PCB_PORT))
            sys.stdout.write("\r\033[K%s  markers %d/4" % (pcb.text(), seen))
            sys.stdout.flush()
            if args.show:
                if zone.homography is not None:
                    cv2.polylines(bgr, [zone.image_corners().astype(np.int32)], True,
                                  (0, 200, 0) if present else (0, 0, 255), 2)
                cv2.imshow("pcb (q to quit)", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
