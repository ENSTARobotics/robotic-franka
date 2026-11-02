#!/usr/bin/env python3
"""Takes the arm slowly to 'home' (data/poses.json) and stops there. Runs INSIDE the container, with the stack
running and the controller closed: only one program may drive the arm.

    python3 go_home.py              # speed_scale 0.3, like the slow routine
    python3 go_home.py --scale 0.15
"""
import argparse
import os

import cycle_controller as cc
import pcb_plan

SCALE = 0.3


def main():
    p = argparse.ArgumentParser(description="Takes the arm slowly to home")
    p.add_argument("--scale", type=float, default=SCALE, help="speed_scale (default %.2f)" % SCALE)
    p.add_argument("--poses", default=os.path.join(pcb_plan.DATA_DIR, "poses.json"))
    args = p.parse_args()
    if not 0.0 < args.scale <= SCALE:
        raise SystemExit("[error] --scale must be in (0, %.2f]" % SCALE)
    home = cc.load_poses(args.poses, False)["home"]

    from frankapy import FrankaArm
    # Its own ROS node name: with the default one, ROS would shut down another client's node
    fa = FrankaArm(rosnode_name="aided_building_home")
    q = fa.get_joints()
    if cc.max_dist(q, home) < cc.ARRIVAL_TOL:
        print("Already at home.")
        return
    dur = cc.compute_duration(q, home, args.scale, cc.V_TCP_MAX, cc.PEAK_VEL_MAX)
    print("Going home: largest joint move %.2f rad, %.1f s at scale %.2f" % (cc.max_dist(q, home), dur, args.scale))
    fa.goto_joints([float(x) for x in home], duration=dur)
    err = cc.max_dist(fa.get_joints(), home)
    if err > cc.ARRIVAL_TOL:
        raise SystemExit("[error] it stopped %.3f rad from home: look at the stack log and at Desk (reflex?)" % err)
    print("At home.")


if __name__ == "__main__":
    main()
