#!/usr/bin/env python3
"""Trust supervisor (host): perception -> T(t) -> NORMAL/CAUTIOUS/PAUSED -> robot.

    python3 supervisor.py --v-normal 0.6 --v-cautious 0.3

Receives perception (face.py, voice.py) on UDP 5010, computes T(t) and the state machine of
trust.py, and sends the state to the controller over UDP 5005, like keyboard_supervisor.py.
Sends a summary to screen.py over UDP 5011.
Voice orders from dialogue.py (UDP 5013): start = a; pause = p; slow = CAUTIOUS 20 s;
resume = automatic from CAUTIOUS 20 s. While the robot speaks, the voice emotion is ignored.
Hands (perception/hands.py, UDP 5016): if there are hands next to the PCB when the robot was about
to move, RETRACT (stops and goes up 5 cm) and it stays like that, in any mode, until the zone is
clear AND the operator says "continue" (or presses a, n or c). Without hands.py open nothing is watched.
Keys (in its terminal, or {"key": "a"} over UDP 5017 from panel/panel.py):
       k = calibrate the operator's baseline (robot paused; voice orders are ignored)
       a = AUTOMATIC (T(t) decides the state)
       n / c / p / r = MANUAL: NORMAL / CAUTIOUS / PAUSED / RETRACT (override automatic)
       q = quit
Starts in MANUAL + PAUSED: the robot does not move until you press a (or n/c).
CSV in logs/supervisor_auto_<date>.csv with T(t), each source, the state and the robot.
"""
import argparse
import csv
import json
import os
import select
import socket
import sys
import termios
import time
import tty

from trust import Trust, StateMachine

IP = "127.0.0.1"
CMD_PORT = 5005
STATE_PORT = 5006
PERCEPTION_PORT = 5010
PANEL_PORT = 5011           # summary for screen.py
COMMANDS_PORT = 5013        # orders from dialogue.py (voice)
DIALOGUE_PORT = 5014        # summary for dialogue.py
HANDS_PORT = 5016           # perception/hands.py
CONTROL_PORT = 5017         # keys from the control panel (panel/panel.py)
GUI_PORT = 5018             # summary for the control panel
HANDS_MAX_AGE = 0.5         # s: an older hands message does not count
PANEL_HZ = 10.0
ECHO_MARGIN = 0.5           # s after the robot speaks in which the voice emotion is ignored
CALIBRATION_S = 20.0        # s the calibration with the k key lasts
VOICE_SLOW_S = 20.0         # s a "slow down" / "continue" by voice lasts before it can go up
HZ = 20.0
SCREEN_HZ = 5.0             # redraws per second at most (UDP sending keeps HZ)
ROBOT_TIMEOUT = 0.5
SOURCES = ("face", "audio")

KEYS = {"n": "NORMAL", "c": "CAUTIOUS", "p": "PAUSED", "r": "RETRACT"}
COLOR = {"NORMAL": "\033[42;30m", "CAUTIOUS": "\033[43;30m", "PAUSED": "\033[41;37m",
         "RETRACT": "\033[44;37m"}
RESET = "\033[0m"

COLUMNS = (["t_host", "key", "mode", "cmd_state", "cmd_speed_scale", "cmd_retract", "T", "stress",
            "no_data", "reason", "baseline_face", "baseline_audio", "hands_near", "hands_lock"]
           + ["%s_%s" % (f, k) for f in SOURCES for k in ("stress", "confidence", "label", "age_s")]
           + ["rob_age_s", "rob_state", "rob_segment", "rob_action", "rob_speed_scale", "rob_event", "rob_done",
              "rob_error", "ee_x", "ee_y", "ee_z"])


def color(state):
    return COLOR.get(state, "") + " %-8s " % state + RESET


def read_all(sock):
    """Every pending JSON on the socket (the ones that do not parse are ignored)."""
    messages = []
    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except (BlockingIOError, socket.error):
            return messages
        try:
            messages.append(json.loads(data.decode("utf-8")))
        except ValueError:
            pass


def main():
    p = argparse.ArgumentParser(description="Trust supervisor: perception -> robot state")
    p.add_argument("--v-normal", type=float, default=0.3, help="speed_scale in NORMAL")
    p.add_argument("--v-cautious", type=float, default=0.15, help="speed_scale in CAUTIOUS")
    p.add_argument("--log-dir", default=os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs")))
    args = p.parse_args()
    speed = {"NORMAL": args.v_normal, "CAUTIOUS": args.v_cautious, "PAUSED": 0.0, "RETRACT": 0.0}
    keyboard = sys.stdin.isatty()      # without a terminal (started by the control panel), keys only over UDP

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, time.strftime("supervisor_auto_%Y%m%d_%H%M%S.csv"))
    f_csv = open(csv_path, "w", newline="")
    log = csv.writer(f_csv)
    log.writerow(COLUMNS)

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_state.bind((IP, STATE_PORT))
    sock_state.setblocking(False)
    sock_perc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_perc.bind((IP, PERCEPTION_PORT))
    sock_perc.setblocking(False)
    sock_voice = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_voice.bind((IP, COMMANDS_PORT))
    sock_voice.setblocking(False)
    sock_hands = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_hands.bind((IP, HANDS_PORT))
    sock_hands.setblocking(False)
    sock_control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_control.bind((IP, CONTROL_PORT))
    sock_control.setblocking(False)
    mute_audio_until = 0.0
    hands, t_hands = {}, None
    hands_lock = False          # hands next to the PCB: RETRACT until the zone is clear + "continue"

    trust, fsm = Trust(), StateMachine()
    mode, manual = "MANUAL", "PAUSED"
    t_calibration_end, calibration = None, ""   # calibration: result of the last one, for dialogue.py
    robot, t_robot, last_event = {}, None, None
    text_prev, t_screen, t_flush, t_panel = "", 0.0, 0.0, 0.0

    def line(text):
        nonlocal text_prev
        sys.stdout.write(("\r\033[K" if keyboard else "") + text + "\n")
        sys.stdout.flush()
        text_prev = ""

    print("Trust supervisor. k = calibrate | a = AUTOMATIC | n/c/p/r = manual NORMAL/CAUTIOUS/PAUSED/RETRACT"
          " | q = quit")
    print("NORMAL %.2f, CAUTIOUS %.2f. Starts in MANUAL + PAUSED. Perception on UDP %d. Log: %s\n"
          % (args.v_normal, args.v_cautious, PERCEPTION_PORT, csv_path))

    if keyboard:
        fd = sys.stdin.fileno()
        terminal_config = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    else:
        print("No terminal: keys only over UDP %d (control panel)." % CONTROL_PORT)
    period = 1.0 / HZ
    try:
        while True:
            t_tick = time.time()

            # Hands next to the PCB (hands.py). Without recent messages it is unknown: it does not count as near
            for m in read_all(sock_hands):
                hands, t_hands = m, t_tick
            watching = t_hands is not None and t_tick - t_hands <= HANDS_MAX_AGE and hands.get("near") is not None
            hands_near = watching and hands.get("near") is True

            def release_hands(origin):
                """"continue" (or a/n/c) with the zone clear removes the lock; with hands still there, it does not."""
                nonlocal hands_lock
                if hands_near:
                    line("[hands] %s ignored: there are still hands next to the PCB" % origin)
                    return False
                hands_lock = False
                line("[hands] zone clear and %s: resuming" % origin)
                return True

            # Keyboard and control panel
            keys = []
            if keyboard and select.select([fd], [], [], 0)[0]:
                keys.append(os.read(fd, 32).decode("utf-8", "ignore")[-1:].lower())
            keys += [str(m.get("key", ""))[:1].lower() for m in read_all(sock_control)]
            key, quit_now = "", False
            for key in keys:
                if key == "q":
                    quit_now = True
                    break
                if t_calibration_end is not None and (key in ("a", "k") or key in KEYS):
                    t_calibration_end, calibration = None, "cancelled"
                    trust.cancel_calibration()
                    line("[calibration] cancelled")
                if key == "k":
                    mode, manual = "MANUAL", "PAUSED"
                    t_calibration_end = t_tick + CALIBRATION_S
                    trust.start_calibration()
                    line("[key k] calibrating %.0f s: look at the camera and talk normally" % CALIBRATION_S)
                elif key == "a":
                    mode = "AUTO"
                    line("[key a] AUTOMATIC mode: T(t) decides the state")
                elif key in KEYS:
                    mode, manual = "MANUAL", KEYS[key]
                    line("[key %s] MANUAL mode -> %s" % (key, color(manual)))
                elif key.strip():
                    line("[key %r] not recognized (k, a, n, c, p, r, q)" % key)
                if hands_lock and key in ("a", "n", "c"):
                    release_hands("key " + key)
            if quit_now:
                break

            # Voice orders (dialogue.py); logged in the "key" column as voice:<order>
            for m in read_all(sock_voice):
                order = m.get("command")
                if order == "speaking":
                    mute_audio_until = float(m.get("until", 0.0)) + ECHO_MARGIN
                    continue
                if t_calibration_end is not None:
                    continue                    # we asked them to talk: a misheard "yes" must not start the robot
                if hands_lock and order in ("start", "resume") and not release_hands("voice " + order):
                    continue
                if order == "start":
                    mode = "AUTO"
                elif order == "pause":
                    mode, manual = "MANUAL", "PAUSED"
                elif order == "resume":
                    mode = "AUTO"
                    fsm.force("CAUTIOUS", "voice", VOICE_SLOW_S, t_tick)
                elif order == "slow":
                    if mode == "AUTO":
                        fsm.force("CAUTIOUS", "voice", VOICE_SLOW_S, t_tick)
                    else:
                        manual = "CAUTIOUS"
                else:
                    continue
                key = "voice:" + order
                line("[voice] %s -> mode %s, %s" % (order, mode, color(fsm.state if mode == "AUTO" else manual)))

            if t_calibration_end is not None and t_tick >= t_calibration_end:
                t_calibration_end = None
                baseline = trust.finish_calibration()
                missing = [f for f in SOURCES if f not in baseline]
                calibration = "ok" if not missing else "no_" + missing[0] if baseline else "no_data"
                line("[calibration] baseline: " + ", ".join(
                    "%s %s" % (f, "%.2f" % baseline[f] if f in baseline else "no data, stays uncalibrated")
                    for f in SOURCES))

            # Perception -> T(t) -> state machine (always runs, also in manual)
            for m in read_all(sock_perc):
                if m.get("source") == "audio" and t_tick < mute_audio_until:
                    continue                    # it is the robot speaking, not the operator
                trust.receive(m, t_tick)
            T = trust.update(t_tick)
            no_data = trust.no_data(t_tick)
            if fsm.update(T, no_data, t_tick):
                line("[trust] %s %s: %s" % ("->" if mode == "AUTO" else "(suggested, manual mode)",
                                            color(fsm.state), fsm.reason))

            state = fsm.state if mode == "AUTO" else manual
            # Hands: they only lock if the robot was about to move (before starting, the operator places the PCB).
            # If it was already paused, the lock comes as soon as something tries to resume with the hands there
            if hands_near and not hands_lock and state in ("NORMAL", "CAUTIOUS"):
                hands_lock = True
                line("[hands] hands next to the PCB -> %s until the zone is clear and \"continue\"" % color("RETRACT"))
            if hands_lock:
                state = "RETRACT"
            cause = ("calibrating" if t_calibration_end is not None else
                     ("hands" if hands_near else "hands_clear") if hands_lock else
                     fsm.cause if mode == "AUTO" else "manual")
            cmd = {"t": time.time(), "enabled": state != "PAUSED", "speed_scale": speed[state],
                   "retract": state == "RETRACT", "state": "PAUSED" if state == "RETRACT" else state,
                   "trust": None if T is None else round(T, 3),
                   "reason": "hands next to the PCB" if hands_lock else fsm.reason if mode == "AUTO" else "manual"}
            sock_cmd.sendto(json.dumps(cmd).encode("utf-8"), (IP, CMD_PORT))

            # Robot's state
            for m in read_all(sock_state):
                robot, t_robot = m, t_tick
            age = t_tick - t_robot if t_robot is not None else None
            if robot.get("event") and robot.get("event") != last_event:
                last_event = robot["event"]
                line("[robot] " + last_event)

            # Summary for the operator's screen
            valid = trust.valid(t_tick)
            if t_tick - t_panel >= 1.0 / PANEL_HZ:
                t_panel = t_tick
                sources = {}
                for f, (t_f, m) in trust.last.items():
                    sources[f] = {"label": m.get("label") or "", "stress": m.get("stress"),
                                  "confidence": m.get("confidence"), "valid": f in valid,
                                  "age": round(t_tick - t_f, 2)}
                panel = {"t": t_tick, "mode": mode, "state": state, "cause": cause, "reason": cmd["reason"],
                         "calibration": calibration, "T": cmd["trust"], "sources": sources,
                         "hands": {"watching": watching, "near": hands_near, "lock": hands_lock},
                         "event": last_event or "",
                         "robot": {"connected": age is not None and age <= ROBOT_TIMEOUT,
                                   "segment": robot.get("segment", ""), "action": robot.get("action", ""),
                                   "seg_index": robot.get("seg_index", 0), "n_segments": robot.get("n_segments", 0),
                                   "done": robot.get("done", False)}}
                for port in (PANEL_PORT, DIALOGUE_PORT, GUI_PORT):
                    sock_cmd.sendto(json.dumps(panel).encode("utf-8"), (IP, port))

            # Terminal screen
            sources_txt = []
            for f in SOURCES:
                if f in valid:
                    m = trust.last[f][1]
                    sources_txt.append("%s %s s%.2f c%.2f" % (f, m.get("label") or "", valid[f][0], valid[f][1]))
                elif f in trust.last:
                    sources_txt.append("%s --" % f)
            T_txt = "T=--  " if T is None else "T=%.2f" % T
            if age is None:
                rob_txt = "\033[41;37m NO ROBOT \033[0m"
            elif age > ROBOT_TIMEOUT:
                rob_txt = "\033[41;37m ROBOT LOST \033[0m"
            else:
                rob_txt = "%s %s" % (robot.get("segment", "?"), {"segment": "MOVING", "retract": "LIFTING",
                                     "soldering": "SOLDERING"}.get(robot.get("action"), "still"))
            hands_txt = ("\033[41;37m HANDS \033[0m" if hands_near else
                         "\033[44;37m say continue \033[0m" if hands_lock else "" if watching else "hands --")
            text = "\r\033[K%-6s %s %s | %s | %s %s" % (mode, T_txt, color(state),
                                                       ", ".join(sources_txt) or "no perception", rob_txt, hands_txt)
            if keyboard and text != text_prev and t_tick - t_screen >= 1.0 / SCREEN_HZ:
                sys.stdout.write(text)
                sys.stdout.flush()
                text_prev, t_screen = text, t_tick

            # CSV
            row = [round(t_tick, 3), key, mode, cmd["state"], cmd["speed_scale"], cmd["retract"],
                   "" if T is None else round(T, 4), "" if trust.stress is None else round(trust.stress, 4),
                   no_data, fsm.reason, trust.baseline.get("face", ""), trust.baseline.get("audio", ""),
                   hands_near if watching else "", hands_lock]
            for f in SOURCES:
                if f in trust.last:
                    t_f, m = trust.last[f]
                    row += [m.get("stress"), m.get("confidence"), m.get("label"), round(t_tick - t_f, 3)]
                else:
                    row += ["", "", "", ""]
            ee = robot.get("ee_xyz") or [""] * 3
            row += ["" if age is None else round(age, 3)] + [robot.get(k, "") for k in
                    ("state", "segment", "action", "speed_scale", "event", "done", "error")] + list(ee)
            log.writerow(row)
            if t_tick - t_flush >= 1.0:
                f_csv.flush()
                t_flush = t_tick

            time.sleep(max(0.0, period - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            sock_cmd.sendto(json.dumps({"t": time.time(), "enabled": False, "speed_scale": 0.0,
                                        "retract": False, "state": "PAUSED"}).encode("utf-8"),
                            (IP, CMD_PORT))
            time.sleep(0.02)
        if keyboard:
            termios.tcsetattr(fd, termios.TCSADRAIN, terminal_config)
        f_csv.close()
        print("\nSupervisor closed (last command: PAUSED). Log in %s" % csv_path)


if __name__ == "__main__":
    main()
