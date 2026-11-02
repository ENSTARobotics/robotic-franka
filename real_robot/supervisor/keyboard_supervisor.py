#!/usr/bin/env python3
"""Keyboard supervisor (phase 1 of the UDP bridge). Runs on the host.

    python3 keyboard_supervisor.py [--v-normal 0.3] [--v-cautious 0.15]

Keys: n = NORMAL, c = CAUTIOUS, p = PAUSED, r = RETRACT (goes up 5 cm and waits), q = quit.
- Sends the command to UDP 5005 at ~20 Hz (starts in PAUSED).
- Receives and shows the robot's state on UDP 5006.
- Saves everything to logs/supervisor_<date>.csv (one row per tick).
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

IP = "127.0.0.1"
CMD_PORT = 5005
STATE_PORT = 5006
HZ = 20.0
ROBOT_TIMEOUT = 0.5         # s without the robot's state -> shown as lost
SCREEN_HZ = 5.0             # redraws per second at most (UDP sending keeps HZ)

KEYS = {"n": "NORMAL", "c": "CAUTIOUS", "p": "PAUSED", "r": "RETRACT"}
COLOR = {"NORMAL": "\033[42;30m", "CAUTIOUS": "\033[43;30m", "PAUSED": "\033[41;37m",
         "RETRACT": "\033[44;37m"}
RESET = "\033[0m"

COLUMNS = (["t_host", "key", "cmd_state", "cmd_speed_scale", "cmd_enabled", "cmd_retract",
            "rob_age_s", "rob_t", "rob_state", "rob_segment", "rob_seg_index", "rob_n_segments",
            "rob_moving", "rob_action", "rob_speed_scale", "rob_comm_ok", "rob_done", "rob_error", "rob_event"]
           + ["q%d" % i for i in range(1, 8)] + ["ee_x", "ee_y", "ee_z"])


def color(state):
    return COLOR.get(state, "") + " %-8s " % state + RESET


def main():
    p = argparse.ArgumentParser(description="Keyboard supervisor for cycle_controller.py")
    p.add_argument("--v-normal", type=float, default=0.3, help="speed_scale in NORMAL")
    p.add_argument("--v-cautious", type=float, default=0.15, help="speed_scale in CAUTIOUS")
    p.add_argument("--log-dir", default=os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs")))
    args = p.parse_args()
    speed = {"NORMAL": args.v_normal, "CAUTIOUS": args.v_cautious, "PAUSED": 0.0, "RETRACT": 0.0}
    if not sys.stdin.isatty():
        sys.exit("[error] A terminal is needed to read the keyboard.")

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, time.strftime("supervisor_%Y%m%d_%H%M%S.csv"))
    f_csv = open(csv_path, "w", newline="")
    log = csv.writer(f_csv)
    log.writerow(COLUMNS)

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_state.bind((IP, STATE_PORT))
    sock_state.setblocking(False)

    state = "PAUSED"
    robot = {}
    t_robot = None
    last_event = None
    text_prev, t_screen, t_flush = "", 0.0, 0.0

    def send(st):
        # RETRACT travels as PAUSED + retract=true (UDP contract); enabled=true because it moves
        cmd = {"t": time.time(), "enabled": st != "PAUSED", "speed_scale": speed[st],
               "retract": st == "RETRACT", "state": "PAUSED" if st == "RETRACT" else st}
        sock_cmd.sendto(json.dumps(cmd).encode("utf-8"), (IP, CMD_PORT))
        return cmd

    def line(text):
        nonlocal text_prev
        sys.stdout.write("\r\033[K" + text + "\n")
        text_prev = ""          # the status line was erased: redraw it on the next tick

    print("Keyboard supervisor. n = NORMAL (%.2f), c = CAUTIOUS (%.2f), p = PAUSED,"
          " r = RETRACT (goes up 5 cm and waits), q = quit" % (args.v_normal, args.v_cautious))
    print("Starts in PAUSED. Log: %s\n" % csv_path)

    fd = sys.stdin.fileno()
    terminal_config = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    period = 1.0 / HZ
    try:
        while True:
            t_tick = time.time()

            # Keyboard
            # (os.read and not sys.stdin.read: Python's buffer hid keys from select)
            key = ""
            if select.select([fd], [], [], 0)[0]:
                key = os.read(fd, 32).decode("utf-8", "ignore")[-1:].lower()
                if key == "q":
                    break
                if key in KEYS:
                    state = KEYS[key]
                    line("[key %s] command -> %s" % (key, color(state)))
                elif key.strip():
                    line("[key %r] not recognized (n, c, p, r, q)" % key)

            cmd = send(state)

            # Robot's state (the last one that arrived)
            while True:
                try:
                    data, _ = sock_state.recvfrom(65536)
                except (BlockingIOError, socket.error):
                    break
                try:
                    robot = json.loads(data.decode("utf-8"))
                    t_robot = time.time()
                except ValueError:
                    pass

            age = time.time() - t_robot if t_robot is not None else None
            if robot.get("event") and robot.get("event") != last_event:
                last_event = robot["event"]
                line("[robot] " + last_event)

            # Screen
            if age is None:
                rob_txt = ("\033[41;37m NO DATA FROM THE ROBOT \033[0m "
                           "(is cycle_controller.py running, with Enter pressed?)")
            elif robot.get("done"):
                rob_txt = "finished " + (("ERROR: " + robot["error"]) if robot.get("error") else "OK")
            elif age > ROBOT_TIMEOUT:
                rob_txt = "\033[41;37m ROBOT LOST (%.1f s) \033[0m" % age
            else:
                rob_txt = "%s %-12s %2d/%-2d %s v=%.2f%s" % (
                    color(robot.get("state", "?")), robot.get("segment", "?"),
                    robot.get("seg_index", 0) + 1, robot.get("n_segments", 0),
                    {"segment": "MOVING  ", "retract": "LIFTING ", "soldering": "SOLDERING"}.get(
                        robot.get("action"), "still   "),
                    robot.get("speed_scale", 0.0),
                    "" if robot.get("comm_ok") else " no-comm")
            # Redraw only when it changes and at SCREEN_HZ at most: redrawing at 20 Hz (above all in the
            # VS Code terminal) uses CPU that franka_interface needs without an RT kernel
            text = "\r\033[KSENT %s v=%.2f | ROBOT %s" % (color(state), speed[state], rob_txt)
            if text != text_prev and t_tick - t_screen >= 1.0 / SCREEN_HZ:
                sys.stdout.write(text)
                sys.stdout.flush()
                text_prev, t_screen = text, t_tick

            # CSV
            q = robot.get("q") or [""] * 7
            ee = robot.get("ee_xyz") or [""] * 3
            log.writerow([round(t_tick, 3), key, cmd["state"], cmd["speed_scale"], cmd["enabled"], cmd["retract"],
                          "" if age is None else round(age, 3)]
                         + [robot.get(k, "") for k in ("t", "state", "segment", "seg_index", "n_segments",
                                                       "moving", "action", "speed_scale", "comm_ok", "done",
                                                       "error", "event")]
                         + list(q) + list(ee))
            if t_tick - t_flush >= 1.0:
                f_csv.flush()
                t_flush = t_tick

            time.sleep(max(0.0, period - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        # On exit, leave the robot in PAUSED (and even if it does not arrive, the timeout stops it)
        for _ in range(5):
            send("PAUSED")
            time.sleep(0.02)
        termios.tcsetattr(fd, termios.TCSADRAIN, terminal_config)
        f_csv.close()
        print("\nSupervisor closed (last command: PAUSED). Log in %s" % csv_path)


if __name__ == "__main__":
    main()
