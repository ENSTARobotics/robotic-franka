#!/usr/bin/env python3
""""Soldering" cycle by segments, controlled over UDP (phase 1 of the bridge).

Runs in the container with frankapy:
    python3 cycle_controller.py                 # approach/solder/retreat cycle
    python3 cycle_controller.py --pcb           # grid of points over the PCB (pcb_plan.py)
    python3 cycle_controller.py --simulated     # no robot (fake arm, for the laptop)

- Listens to the supervisor's commands on UDP 5005 (JSON):
    {"t": .., "enabled": true, "speed_scale": 0.3, "retract": false, "state": "NORMAL"}
- Sends the robot's state to UDP 5006 at ~20 Hz.
- PAUSED, enabled=false or >0.5 s without valid commands -> stop_skill() and hold still.
- retract=true (with enabled=true) -> stop the segment, lift the tool 5 cm vertically, slowly and
  at a fixed speed, and hold still. Without communication it never retracts.
- Back to NORMAL/CAUTIOUS, the current segment resumes from the current position.
- speed_scale is set when each segment starts (or resumes after a pause).
- If it stops while waiting at a point ("soldering"), that point counts as done.
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import pcb_plan

IP = "127.0.0.1"            # the container uses --net=host
CMD_PORT = 5005
STATE_PORT = 5006
HZ = 20.0
CMD_TIMEOUT = 0.5           # s without commands -> PAUSED (fail-safe)

# Each segment's duration depends on what moves (the tip and the joints), it is not fixed:
#   duration = 1.875 * max(tcp_distance / V_TCP_MAX, max|dq| / PEAK_VEL_MAX) / speed_scale
# speed_scale multiplies every speed, so CAUTIOUS (0.3) takes twice as long as NORMAL (0.6).
V_TCP_MAX = 0.25            # m/s, peak tool speed with speed_scale = 1.0; --max-tcp-vel
V_TCP_CEILING = 0.5         # m/s, never allow more from the command line
MIN_DURATION = 0.5          # s, never send shorter motions
MIN_SCALE = 0.05            # avoids absurd durations
PEAK_VEL_MAX = 1.0          # rad/s, peak joint speed with speed_scale = 1.0; --max-peak-vel
PEAK_VEL_CEILING = 1.0      # rad/s, never allow more from the command line
PEAK_FACTOR = 1.875         # peak / mean in a min-jerk trajectory
ARRIVAL_TOL = 0.02          # rad, largest error accepted when a segment ends
RETRACT_M = 0.05            # m the tool goes up when retracting
RETRACT_DUR = 2.5           # s, fixed and slow speed (peak ~3.8 cm/s); --retract-dur
RETRACT_DUR_MIN = 2.0       # s, the retraction is never faster than this

SEQUENCE = ["approach", "solder", "retreat"]
STATES = ("NORMAL", "CAUTIOUS", "PAUSED")

# Only for --simulated if poses.json does not exist (values from the repo's zones.yaml)
DEMO_POSES = {
    "home": [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
    "approach": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5],
    "solder": [0.05, -0.45, 0.0, -1.95, 0.0, 1.45, 0.45],
    "retreat": [0.0, -0.6, 0.0, -2.1, 0.0, 1.5, 0.6],
}


class Pose(object):
    def __init__(self, translation):
        self.translation = translation


class SimulatedArm(object):
    """Imitates the part of FrankaArm that we use, with min-jerk interpolation."""

    def __init__(self, q0):
        self._q = list(q0)
        self._q_start = None
        self._target = None
        self._t0 = 0.0
        self._dur = 1.0

    def _update(self):
        if self._target is None:
            return
        s = min(1.0, (time.time() - self._t0) / self._dur)
        a = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        self._q = [qi + a * (qf - qi) for qi, qf in zip(self._q_start, self._target)]
        if s >= 1.0:
            self._target = None

    def goto_joints(self, joints, duration=5, block=True, **kwargs):
        # Same check as frankapy
        assert type(duration) in (int, float), "Incorrect time type. Should be int or float."
        self._update()
        self._q_start = list(self._q)
        self._target = list(joints)
        self._t0 = time.time()
        self._dur = float(duration)
        while block and not self.is_skill_done():
            time.sleep(0.01)

    def goto_pose(self, pose, duration=3, block=True, **kwargs):
        # Inverse of get_pose()'s made-up position: z depends only on q[1]
        q = self.get_joints()
        q[1] += (pose.translation[2] - self.get_pose().translation[2]) / 0.2
        self.goto_joints(q, duration=duration, block=block)

    def stop_skill(self):
        self._update()
        self._target = None

    def is_skill_done(self):
        self._update()
        return self._target is None

    def get_joints(self):
        self._update()
        return list(self._q)

    def get_pose(self):
        # Made-up position, only to have numbers that change
        q = self.get_joints()
        r = 0.3 + 0.1 * (q[3] + 2.4)
        return Pose([r * math.cos(q[0]), r * math.sin(q[0]), 0.3 + 0.2 * (q[1] + 0.8)])


def max_dist(a, b):
    """Largest joint difference (rad) between two configurations."""
    return max(abs(x - y) for x, y in zip(a, b))


def compute_duration(q, target, scale, v_tcp, vel_max):
    """Duration from what is left to travel (works the same when resuming from a pause)."""
    rest_q = max_dist(q, target)
    rest_tcp = math.sqrt(sum((a - b) ** 2 for a, b in zip(pcb_plan.fk(q, 0.0)[0],
                                                          pcb_plan.fk(target, 0.0)[0])))
    dur = PEAK_FACTOR * max(rest_tcp / v_tcp, rest_q / vel_max) / scale
    # Explicit float(): frankapy requires a Python float/int, not numpy
    return float(max(MIN_DURATION, dur))


def load_poses(path, simulated):
    if not os.path.exists(path):
        if simulated:
            print("[warning] %s does not exist: using demo poses (simulated only)." % path)
            return dict(DEMO_POSES)
        sys.exit("[error] %s does not exist. Record the poses with record_poses.py." % path)
    with open(path) as f:
        poses = json.load(f)
    for name in ["home"] + SEQUENCE:
        if name not in poses or len(poses[name]) != 7:
            sys.exit("[error] %s: pose '%s' is missing (7 values)." % (path, name))
    return {k: [float(x) for x in v] for k, v in poses.items()}


def cycle_plan(poses, cycles):
    names = SEQUENCE * cycles + ["home"]
    return [{"name": n, "q": poses[n], "wait": 0.0, "factor": 1.0} for n in names]


def parse_command(data):
    """Returns the validated command, or None if it is not valid (it is ignored)."""
    try:
        cmd = json.loads(data.decode("utf-8"))
        state = cmd["state"]
        scale = float(cmd["speed_scale"])
        enabled = bool(cmd.get("enabled", True))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if state not in STATES or math.isnan(scale) or not 0.0 <= scale <= 1.0:
        return None
    return {"state": state, "speed_scale": scale, "enabled": enabled,
            "retract": bool(cmd.get("retract", False))}


def main():
    p = argparse.ArgumentParser(description="Soldering cycle controlled over UDP")
    p.add_argument("--simulated", action="store_true", help="no robot: fake arm")
    p.add_argument("--poses", default=os.path.join(pcb_plan.DATA_DIR, "poses.json"))
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--pcb", action="store_true", help="routine of points over the PCB instead of the cycle")
    p.add_argument("--max-tcp-vel", type=float, default=V_TCP_MAX,
                   help="m/s, peak tool speed with speed_scale 1.0 (default %.2f)" % V_TCP_MAX)
    p.add_argument("--max-peak-vel", type=float, default=PEAK_VEL_MAX,
                   help="rad/s, peak joint speed with speed_scale 1.0 (default %.1f)" % PEAK_VEL_MAX)
    p.add_argument("--retract-dur", type=float, default=RETRACT_DUR,
                   help="s it takes to go up %.0f cm (default %.1f)" % (RETRACT_M * 100, RETRACT_DUR))
    p.add_argument("--max-scale", type=float, default=0.3,
                   help="cap on speed_scale (default 0.3, first test)")
    pcb_plan.add_arguments(p)
    args = p.parse_args()
    if not 0.0 < args.max_scale <= 1.0:
        sys.exit("[error] --max-scale must be in (0, 1].")
    if not 0.0 < args.max_tcp_vel <= V_TCP_CEILING:
        sys.exit("[error] --max-tcp-vel must be in (0, %.2f] m/s." % V_TCP_CEILING)
    if not 0.0 < args.max_peak_vel <= PEAK_VEL_CEILING:
        sys.exit("[error] --max-peak-vel must be in (0, %.1f] rad/s." % PEAK_VEL_CEILING)
    if args.retract_dur < RETRACT_DUR_MIN:
        sys.exit("[error] --retract-dur must be >= %.1f s." % RETRACT_DUR_MIN)

    poses = load_poses(args.poses, args.simulated)
    if args.pcb:
        try:
            _, info = pcb_plan.load(os.path.dirname(os.path.abspath(args.poses)))
            plan, report = pcb_plan.plan_from_args(args, poses, info)
        except (IOError, ValueError, KeyError) as e:
            sys.exit("[error] invalid PCB plan: %s" % e)
    else:
        plan, report = cycle_plan(poses, args.cycles), [
            "Plan: %d cycles %s and back to home." % (args.cycles, " -> ".join(SEQUENCE))]

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd.bind((IP, CMD_PORT))
    sock_cmd.setblocking(False)
    sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (IP, STATE_PORT)

    if args.simulated:
        fa = SimulatedArm(poses["home"])
    else:
        from frankapy import FrankaArm
        fa = FrankaArm()

    print("\n=== cycle_controller (%s) ===" % ("SIMULATED" if args.simulated else "REAL ROBOT"))
    print("\n".join(report))
    print("Total: %d segments." % len(plan))
    print("speed_scale cap: %.2f | peak vel. <= %.2f rad/s | min. duration %.1f s"
          % (args.max_scale, args.max_peak_vel * args.max_scale, MIN_DURATION))
    print("Peak speed with speed_scale 1.0: tool %.2f m/s, joints %.1f rad/s"
          % (args.max_tcp_vel, args.max_peak_vel))
    print("Retraction: goes up %.0f cm in %.1f s" % (RETRACT_M * 100, args.retract_dur))
    print("First segment: from the current position to '%s' (max. dist. %.2f rad)."
          % (plan[0]["name"], max_dist(fa.get_joints(), plan[0]["q"])))
    print("The robot only moves if the supervisor sends NORMAL/CAUTIOUS.")
    input("Emergency stop in your hand. Press Enter to start (Ctrl+C to quit)... ")

    idx = 0                  # current segment in plan
    q_segment_start = None   # None = the segment has not started yet
    segment_scale = 0.0
    action = None            # None, "segment", "retract" or "soldering" (waiting at a point)
    t_wait_end = 0.0
    retracted = False        # the retraction for this stop was already done
    warned_cap = False       # already warned that the supervisor asks for more than --max-scale
    last_cmd = None
    t_last_cmd = None
    effective_prev = None
    event = "waiting for supervisor"
    error = ""
    print("[event] " + event)

    def report_event(text):
        print("[event] " + text)
        return text

    period = 1.0 / HZ
    try:
        while (idx < len(plan) or action == "soldering") and not error:
            t_tick = time.time()

            # 1) Read every pending command, keep the last valid one
            while True:
                try:
                    data, _ = sock_cmd.recvfrom(4096)
                except (BlockingIOError, socket.error):
                    break
                cmd = parse_command(data)
                if cmd is not None:
                    last_cmd, t_last_cmd = cmd, time.time()

            # 2) Effective state (fail-safe: when in doubt, PAUSED)
            comm_ok = t_last_cmd is not None and time.time() - t_last_cmd <= CMD_TIMEOUT
            if not comm_ok:
                effective, reason = "PAUSED", "no communication"
            elif not last_cmd["enabled"]:
                effective, reason = "PAUSED", "motion disabled"
            elif last_cmd["retract"]:
                effective, reason = "RETRACT", "supervisor"
            elif last_cmd["speed_scale"] <= 0.0:
                effective, reason = "PAUSED", "motion disabled"
            else:
                effective, reason = last_cmd["state"], "supervisor"
            if effective != effective_prev:
                event = report_event("state %s (%s)" % (effective, reason))
                effective_prev = effective

            # 3) Act
            if effective in ("PAUSED", "RETRACT"):
                if action == "segment":
                    fa.stop_skill()
                    action = None
                    event = report_event("pause: stop_skill() in '%s'" % plan[idx]["name"])
                elif action == "soldering":
                    action = None
                    event = report_event("pause: point '%s' interrupted" % plan[idx - 1]["name"])
                elif action == "retract" and effective == "PAUSED":
                    fa.stop_skill()
                    action = None
                    event = report_event("pause: retraction stopped")

            if effective == "RETRACT":
                if action == "retract" and fa.is_skill_done():
                    action = None
                    retracted = True
                    event = report_event("retraction done: holding still")
                elif action is None and not retracted:
                    T = fa.get_pose()
                    t = [float(x) for x in T.translation]
                    t[2] += RETRACT_M
                    T.translation = t
                    fa.goto_pose(T, duration=float(args.retract_dur), block=False)
                    action = "retract"
                    event = report_event("retraction: going up %.0f cm in %.1f s" % (RETRACT_M * 100, args.retract_dur))
            elif effective != "PAUSED":
                if action == "retract":
                    fa.stop_skill()
                    action = None
                retracted = False
                if action == "soldering" and time.time() >= t_wait_end:
                    action = None
                    event = report_event("point '%s' done" % plan[idx - 1]["name"])
                if action == "segment" and fa.is_skill_done():
                    action = None
                    seg = plan[idx]
                    q_end = fa.get_joints()
                    err = max_dist(q_end, seg["q"])
                    if err > ARRIVAL_TOL:
                        text = "ERROR: '%s' ended %.3f rad from the target" % (seg["name"], err)
                        if max_dist(q_end, q_segment_start) < ARRIVAL_TOL:
                            # frankapy does not report it (ignore_errors=True): the robot rejected the motion
                            text += (" -- the robot did NOT move: look for errors in the franka_interface"
                                     " terminal and at the lights/Desk (stop pressed or reflex error)")
                        error = event = report_event(text)
                        break
                    event = report_event("segment '%s' done (%d/%d)" % (seg["name"], idx + 1, len(plan)))
                    idx += 1
                    q_segment_start = None
                    if seg["wait"] > 0.0:
                        action = "soldering"
                        t_wait_end = time.time() + seg["wait"]
                        event = report_event("soldering at '%s' (%.1f s)" % (seg["name"], seg["wait"]))

                if action is None and idx < len(plan):
                    seg = plan[idx]
                    q = fa.get_joints()
                    resuming = q_segment_start is not None
                    if not resuming:
                        q_segment_start = q
                    if max_dist(q, seg["q"]) < ARRIVAL_TOL:
                        # Already there (e.g. paused right at the end): go to the next one
                        idx += 1
                        q_segment_start = None
                    else:
                        requested = last_cmd["speed_scale"]
                        segment_scale = min(max(requested, MIN_SCALE), args.max_scale)
                        dur = seg["factor"] * compute_duration(q, seg["q"], segment_scale,
                                                               args.max_tcp_vel, args.max_peak_vel)
                        if requested > args.max_scale + 1e-9 and not warned_cap:
                            warned_cap = True
                            report_event("[warning] the supervisor asks for scale %.2f but the cap is %.2f:"
                                         " start with --max-scale %.2f" % (requested, args.max_scale, requested))
                        fa.goto_joints([float(x) for x in seg["q"]], duration=float(dur), block=False)
                        action = "segment"
                        event = report_event("%s '%s' (%d/%d) scale %.2f, %.1f s"
                                             % ("resume" if resuming else "start", seg["name"], idx + 1,
                                                len(plan), segment_scale, dur))

            # 4) Send the robot's state. A single frankapy query per cycle: each query is a ROS
            # service and, without an RT kernel, the extra load causes
            # communication_constraints_violation. The TCP position comes from the FK.
            q = [float(x) for x in fa.get_joints()]
            state = {
                "t": time.time(),
                "q": [round(x, 5) for x in q],
                "ee_xyz": [round(x, 4) for x in pcb_plan.fk(q, 0.0)[0]],
                "segment": plan[idx]["name"] if idx < len(plan) else "end",
                "seg_index": idx,
                "n_segments": len(plan),
                "moving": action in ("segment", "retract"),
                "action": action or "",
                "state": effective,
                "speed_scale": segment_scale,
                "comm_ok": comm_ok,
                "event": event,
                "done": False,
                "error": "",
            }
            sock_state.sendto(json.dumps(state).encode("utf-8"), destination)

            time.sleep(max(0.0, period - (time.time() - t_tick)))
    except KeyboardInterrupt:
        event = report_event("Ctrl+C: stopping")
    except Exception as e:  # goto rejected (limits, virtual walls...), ROS failure, etc.
        error = event = report_event("ERROR: %r" % (e,))
    finally:
        if action in ("segment", "retract"):
            try:
                fa.stop_skill()
            except Exception as e:
                print("[error] stop_skill() failed: %r -> use the emergency stop" % (e,))
        if not error and idx >= len(plan):
            event = report_event("routine finished")
        final = {"t": time.time(), "q": [], "ee_xyz": [], "segment": "end", "seg_index": idx,
                 "n_segments": len(plan), "moving": False, "action": "", "state": "PAUSED",
                 "speed_scale": 0.0, "comm_ok": False, "event": event, "done": True, "error": error}
        for _ in range(5):
            sock_state.sendto(json.dumps(final).encode("utf-8"), destination)
            time.sleep(0.05)
        sock_cmd.close()
        sock_state.close()


if __name__ == "__main__":
    main()
