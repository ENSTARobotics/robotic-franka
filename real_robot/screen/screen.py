#!/usr/bin/env python3
"""Screen for the operator: what the system sees and hears and what the robot does. Runs on the host.

    python3 screen.py                 # 1280x720 window
    python3 screen.py --full          # full screen (for the operator's monitor)

Receives over UDP:
    5011  summary from supervisor.py (state, cause, T(t), face, voice, robot)
    5012  camera image already annotated by face.py (JPEG)
If one is missing, it says so on the screen and keeps waiting. q or Esc to quit.
"""
import argparse
import functools
import json
import os
import socket
import time

# Numerical libraries use every core by default and saturate the CPU: without an RT kernel that
# causes communication_constraints_violation on the robot. It must come before importing numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

# Fonts for the OpenCV (Qt) windows: when imported it points to a folder of its own that does not exist, and warns
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

IP = "127.0.0.1"
PANEL_PORT = 5011
IMAGE_PORT = 5012
WIDTH, HEIGHT = 1280, 720
CAMERA_WIDTH = 760
HZ = 10.0
NO_DATA_S = 1.0             # s without messages -> "not connected"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BACKGROUND, TEXT, GRAY = (24, 26, 30), (235, 235, 235), (140, 140, 140)
STATE_COLOR = {"NORMAL": (46, 160, 67), "CAUTIOUS": (230, 160, 20), "PAUSED": (200, 50, 50),
               "RETRACT": (40, 110, 200), None: (90, 90, 90)}
TITLE = {"NORMAL": "WORKING", "CAUTIOUS": "SLOWER", "PAUSED": "STOPPED",
         "RETRACT": "MOVING AWAY", None: "NOT CONNECTED"}
# Message for the operator by state and cause (trust.py)
MESSAGE = {
    ("NORMAL", "calm"): "All good: normal speed",
    ("CAUTIOUS", "tension"): "Going slower: I notice tension",
    ("CAUTIOUS", "no_perception"): "Going slower: I can't see or hear you well",
    ("CAUTIOUS", "calm"): "You seem calmer: picking up again",
    ("CAUTIOUS", "voice"): "Going slower, as you asked",
    ("PAUSED", "high_stress"): "I stopped: I notice stress. Take a moment",
    ("PAUSED", "start"): "Ready to start",
    ("PAUSED", "calibrating"): "Calibrating: look at the camera and talk normally",
    ("NORMAL", "manual"): "Normal speed",
    ("CAUTIOUS", "manual"): "Going slower",
    ("PAUSED", "manual"): "Paused",
    ("RETRACT", "manual"): "I moved away from the PCB",
    ("RETRACT", "hands"): "Hands near the PCB: moving away. Move them away",
    ("RETRACT", "hands_clear"): "Zone clear. Say \"continue\" to go on",
    ("CAUTIOUS", "after_hands"): "Going slowly for a moment after the stop",
}
EXPRESSION = {"angry": "Anger", "disgust": "Disgust", "fear": "Fear", "fearful": "Fear",
              "happy": "Happiness", "sad": "Sadness", "surprise": "Surprise", "surprised": "Surprise",
              "neutral": "Neutral"}


@functools.lru_cache(maxsize=None)
def font(size, bold=False):
    """Cached: opening the font for each text of each frame uses a lot of CPU."""
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def fitted_text(d, xy, text, max_width, size, bold=False, color=TEXT):
    """Writes the text, making the letters smaller until it fits in max_width."""
    while size > 14 and d.textlength(text, font=font(size, bold)) > max_width:
        size -= 1
    d.text(xy, text, font=font(size, bold), fill=color)


def stress_color(e):
    """Green (0) -> yellow (0.5) -> red (1)."""
    e = min(1.0, max(0.0, e))
    return (int(min(1.0, 2 * e) * 220), int(min(1.0, 2 * (1 - e)) * 180), 40)


def bar(d, x, y, width, value, color):
    d.rounded_rectangle((x, y, x + width, y + 14), 7, fill=(55, 58, 64))
    if value is not None:
        d.rounded_rectangle((x, y, x + max(14, int(width * min(1.0, max(0.0, value)))), y + 14), 7, fill=color)


def robot_activity(robot, state):
    """Text and progress (0..1) of what the robot is doing."""
    if not robot or not robot.get("connected"):
        return "Robot not connected", None
    if robot.get("done"):
        return "Routine finished", 1.0
    seg, action = robot.get("segment") or "", robot.get("action") or ""
    n = max(1, robot.get("n_segments") or 1)
    progress = min(1.0, (robot.get("seg_index") or 0) / float(n))
    point = seg.split("_p")[-1] if "_p" in seg else ""
    if action == "retract":
        text = "Moving away from the PCB"
    elif action == "soldering":
        text = "Soldering point %s" % point if point else "Soldering"
    else:
        text = {"above": "Going to point %s" % point, "down": "Going down to point %s" % point,
                "up": "Going up from point %s" % point}.get(seg.split("_")[0]) or {
                "home": "Going back to the start position", "approach": "Approaching the PCB",
                "solder": "Going down to solder", "retreat": "Retreating", "end": "Routine finished"
                }.get(seg, seg)
    total = " of %d" % max(1, (n - 2) // 3) if point else ""
    if not action and state in ("PAUSED", "RETRACT"):
        text = "Paused (point %s%s)" % (point, total) if point else "Paused"
    elif point and action != "retract":         # "Moving away from the PCB" names no point
        text += total
    return text, progress


def source_section(d, x, y, title, data, empty):
    d.text((x, y), title, font=font(18, True), fill=GRAY)
    if data and data.get("valid"):
        exp = EXPRESSION.get((data.get("label") or "").lower(), data.get("label") or "?")
        stress = float(data.get("stress") or 0.0)
        d.text((x, y + 26), exp, font=font(30, True), fill=TEXT)
        d.text((x + 250, y + 36), "confidence %.0f%%" % (100 * float(data.get("confidence") or 0)),
               font=font(15), fill=GRAY)
        d.text((x, y + 70), "stress", font=font(15), fill=GRAY)
        bar(d, x + 60, y + 72, 380, stress, stress_color(stress))
    elif data and data.get("label"):                 # last result, no longer weighing in the trust
        exp = EXPRESSION.get(data["label"].lower(), data["label"])
        d.text((x, y + 26), "%s (%s)" % (exp, empty), font=font(26), fill=GRAY)
        d.text((x, y + 70), "stress", font=font(15), fill=GRAY)
        bar(d, x + 60, y + 72, 380, float(data.get("stress") or 0.0), GRAY)
    else:
        d.text((x, y + 26), empty if data else "no signal", font=font(26), fill=GRAY)
        d.text((x, y + 70), "stress", font=font(15), fill=GRAY)
        bar(d, x + 60, y + 72, 380, None, None)


def compose(panel, t_panel, camera, t_camera, now):
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(img)

    # Camera (left)
    if camera is not None and now - t_camera <= NO_DATA_S:
        rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        scale = min(CAMERA_WIDTH / float(rgb.shape[1]), (HEIGHT - 40) / float(rgb.shape[0]))
        photo = Image.fromarray(rgb).resize((int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
        img.paste(photo, (20 + (CAMERA_WIDTH - photo.width) // 2, (HEIGHT - photo.height) // 2))
    else:
        d.rounded_rectangle((20, 20, 20 + CAMERA_WIDTH, HEIGHT - 20), 12, outline=(70, 70, 70), width=2)
        d.text((60, HEIGHT // 2 - 20), "No camera (is face.py running?)", font=font(24), fill=GRAY)

    # Panel (right)
    x = CAMERA_WIDTH + 50
    width = WIDTH - x - 20
    connected = panel is not None and now - t_panel <= NO_DATA_S
    state = panel.get("state") if connected else None
    d.rounded_rectangle((x, 20, x + width, 150), 14, fill=STATE_COLOR.get(state, STATE_COLOR[None]))
    d.text((x + 20, 32), TITLE.get(state, state or ""), font=font(44, True), fill=(255, 255, 255))
    if connected:
        message = MESSAGE.get((state, panel.get("cause")), panel.get("reason") or "")
    else:
        message = "Is supervisor.py running?"
    fitted_text(d, (x + 20, 98), message, width - 40, 21, color=(255, 255, 255))

    sources = (panel or {}).get("sources", {}) if connected else {}
    source_section(d, x, 180, "YOUR FACE", sources.get("face"), "can't see you")
    source_section(d, x, 290, "YOUR VOICE", sources.get("audio"), "silence")

    d.text((x, 400), "TRUST", font=font(18, True), fill=GRAY)
    T = panel.get("T") if connected else None
    d.text((x + 140, 400), "--" if T is None else "%.0f%%" % (100 * T), font=font(18, True), fill=TEXT)
    bar(d, x, 430, width - 10, T, stress_color(1.0 - T) if T is not None else None)

    d.text((x, 480), "ROBOT", font=font(18, True), fill=GRAY)
    text, progress = robot_activity((panel or {}).get("robot") if connected else None, state)
    fitted_text(d, (x, 508), text, width - 10, 26, bold=True)
    bar(d, x, 550, width - 10, progress, (90, 140, 220))

    footer = "%s mode" % ("automatic" if panel.get("mode") == "AUTO" else "manual") if connected else ""
    d.text((x, HEIGHT - 45), footer, font=font(15), fill=GRAY)
    d.text((x + width - 70, HEIGHT - 45), time.strftime("%H:%M:%S"), font=font(15), fill=GRAY)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description="Screen for the operator")
    p.add_argument("--full", action="store_true", help="full screen")
    args = p.parse_args()

    sock_panel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_panel.bind((IP, PANEL_PORT))
    sock_panel.setblocking(False)
    sock_image = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_image.bind((IP, IMAGE_PORT))
    sock_image.setblocking(False)

    window = "Soldering robot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.full:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window, WIDTH, HEIGHT)

    panel, t_panel, camera, t_camera = None, 0.0, None, 0.0
    try:
        while True:
            t_tick = time.time()
            while True:
                try:
                    data, _ = sock_panel.recvfrom(65536)
                    panel, t_panel = json.loads(data.decode("utf-8")), t_tick
                except (BlockingIOError, socket.error):
                    break
                except ValueError:
                    pass
            while True:
                try:
                    data, _ = sock_image.recvfrom(65536)
                except (BlockingIOError, socket.error):
                    break
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    camera, t_camera = image, t_tick

            cv2.imshow(window, compose(panel, t_panel, camera, t_camera, t_tick))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
