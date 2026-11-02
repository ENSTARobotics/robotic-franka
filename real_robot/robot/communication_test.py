#!/usr/bin/env python3
"""Tests what makes move_robot.py (works) different from cycle_controller.py (fails with
communication_constraints_violation). Runs in the container, with the stack started.

    python3 communication_test.py           # the 4 tests, with Enter before each one
    python3 communication_test.py 3         # only test 3

Each test turns the wrist (joint 7) 20 degrees and back, 5 s each way:
  1. blocking goto_joints                          (like move_robot.py)
  2. goto_joints(block=False) + waiting with is_skill_done()
  3. same as 2 + get_joints() at 20 Hz             (what the controller does now)
  4. same as 3 + get_pose() at 20 Hz               (what the controller did before)
If one fails, look at control_command_success_rate in the franka_interface terminal.
"""
import math
import sys
import time

TURN = math.radians(20)
DURATION = 5.0
HZ = 20.0
TOL = 0.02                  # rad
Q7_LIMIT = 2.8973

DESCRIPTION = {
    1: "blocking goto_joints (like move_robot.py)",
    2: "goto_joints(block=False) + is_skill_done()",
    3: "block=False + get_joints() at 20 Hz",
    4: "block=False + get_joints() and get_pose() at 20 Hz",
}


def move(fa, target, mode):
    """Returns (final error in rad, seconds it took)."""
    t0 = time.time()
    if mode == 1:
        fa.goto_joints(target, duration=DURATION)
    else:
        fa.goto_joints(target, duration=DURATION, block=False)
        while not fa.is_skill_done():
            if mode >= 3:
                fa.get_joints()
            if mode >= 4:
                fa.get_pose()
            time.sleep(1.0 / HZ)
    q = [float(x) for x in fa.get_joints()]
    return max(abs(a - b) for a, b in zip(q, target)), time.time() - t0


def main():
    modes = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    if any(m not in DESCRIPTION for m in modes):
        sys.exit("[error] valid tests: 1, 2, 3, 4")

    from frankapy import FrankaArm
    fa = FrankaArm()
    q0 = [float(x) for x in fa.get_joints()]
    sign = 1.0 if q0[6] + TURN < Q7_LIMIT - 0.1 else -1.0
    q1 = list(q0)
    q1[6] += sign * TURN
    print("\nOnly turns the wrist %+.0f degrees and back (%.0f s each way)." % (math.degrees(sign * TURN), DURATION))
    print("Check that the pen does not touch anything. Emergency stop in your hand.")

    results = []
    for mode in modes:
        input("\nTest %d: %s. Enter to run it (Ctrl+C to quit)... " % (mode, DESCRIPTION[mode]))
        ok = True
        for name, target in (("there", q1), ("back", q0)):
            try:
                err, dur = move(fa, target, mode)
            except ValueError as e:
                # frankapy rejects the target before moving (virtual walls): try the other way
                warning = ("  [error] %s\n  The arm is in a posture where turning the wrist hits frankapy's"
                           " virtual walls. Bring the robot to a higher posture (e.g. home) and repeat." % e)
                if name != "there" or q1[6] == q0[6] - sign * TURN:
                    sys.exit(warning)
                print("  frankapy rejected the turn (%s): trying the other way" % e)
                q1[6] = q0[6] - sign * TURN
                target = q1
                try:
                    err, dur = move(fa, target, mode)
                except ValueError:
                    sys.exit(warning)
            good = err < TOL
            print("  %-6s %s (error %.3f rad, %.1f s)" % (name, "OK" if good else "FAILED: did not arrive", err, dur))
            if not good:
                ok = False
                break
        results.append((mode, ok))
        if not ok:
            print("  -> Look at the franka_interface terminal (control_command_success_rate).")
            break

    print("\n=== Summary ===")
    for mode, ok in results:
        print("  %d. %-48s %s" % (mode, DESCRIPTION[mode], "OK" if ok else "FAILED"))


if __name__ == "__main__":
    main()
