#!/usr/bin/env python3
"""Opens or closes the Franka Hand, e.g. to put the pen between the fingers. Runs INSIDE the container, with
the stack running. It only moves the fingers, never the arm.

    python3 gripper.py open
    python3 gripper.py close              # grasps what is between the fingers with 20 N
    python3 gripper.py close --force 40
"""
import argparse

from frankapy import FrankaArm

FORCE = 20.0                # N: holds the pen without crushing it


def main():
    p = argparse.ArgumentParser(description="Opens or closes the Franka Hand")
    p.add_argument("action", choices=["open", "close"])
    p.add_argument("--force", type=float, default=FORCE, help="N when closing (default %.0f)" % FORCE)
    args = p.parse_args()

    # Its own ROS node name: with the default one, ROS would shut down the controller's node
    fa = FrankaArm(rosnode_name="aided_building_gripper")
    if args.action == "open":
        fa.open_gripper()
    else:
        # Width 0 with grasp: it closes until it holds something (the pen) with this force
        fa.goto_gripper(0.0, grasp=True, force=args.force)
    print("Gripper %s: width now %.1f mm" % ("opened" if args.action == "open" else "closed",
                                             1000 * fa.get_gripper_width()))


if __name__ == "__main__":
    main()
