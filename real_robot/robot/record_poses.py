#!/usr/bin/env python3
"""Records the cycle poses by guiding the arm by hand. Runs in the container.

    python3 record_poses.py                          # the 4 poses in order
    python3 record_poses.py solder                   # record only one again (or several)
    python3 record_poses.py --pen-length 4.5         # cm of pen below frankapy's TCP
    python3 record_poses.py --corners --pen-length 1.5   # the 4 corners of the PCB

For each pose: Enter -> guide mode (free arm) -> you move the arm -> Enter -> it is locked
and the position is shown; you confirm and it is saved to poses.json right away.
It also saves position/orientation to pose_info.json and checks it against table.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
ORDER = ["home", "approach", "solder", "retreat"]
CORNERS = ["corner_1", "corner_2", "corner_3", "corner_4"]
DESCRIPTION = {
    "home": "safe position: arm folded and high, pen away from the table and the operator",
    "approach": "pen vertical, tip ~8 cm right ABOVE the soldering point",
    "solder": "pen tip ~5 mm above the cardboard (not touching it), same orientation",
    "retreat": "go up ~15 cm from solder and a bit toward the robot (away from the operator)",
    # Corners in order around the PCB (x toward the operator, y to the robot's left)
    "corner_1": "tip BARELY TOUCHING the corner closest to the robot, on the robot's RIGHT",
    "corner_2": "tip BARELY TOUCHING the corner closest to the robot, on the robot's LEFT",
    "corner_3": "tip BARELY TOUCHING the corner farthest from the robot, on the robot's LEFT",
    "corner_4": "tip BARELY TOUCHING the corner farthest from the robot, on the robot's RIGHT",
}
GUIDE_DURATION = 120.0      # s maximum in guide mode per pose


def inside_polygon(x, y, points):
    """Ray casting: is (x, y) inside the polygon?"""
    inside = False
    for i in range(len(points)):
        (x1, y1), (x2, y2) = points[i], points[i - 1]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def pen_tip(T, length_m):
    """Tip = frankapy's TCP + length along the tool's z axis."""
    R = T.rotation
    return [float(T.translation[i] + R[i][2] * length_m) for i in range(3)]


def check(name, tip, info, table):
    """Common-sense warnings (they do not block)."""
    warnings = []
    contour = [table["arucos"][k] for k in table["contour_order"]]
    height = tip[2] - table["table_z"]
    if height < 0.0:
        warnings.append("the tip is %.1f cm BELOW the table: is the pen length right?" % (-height * 100))
    if name == "solder" or name in CORNERS:
        if not inside_polygon(tip[0], tip[1], contour):
            warnings.append("the point is OUTSIDE the area of the ArUco markers")
    if name in CORNERS and height > 0.015:
        warnings.append("the tip is %.1f cm from the table: does it touch the cardboard?" % (height * 100))
        if height > 0.03:
            warnings.append("the tip is %.1f cm from the table (too high?)" % (height * 100))
    if name == "approach" and "solder" in info:
        s = info["solder"]["tip_xyz"]
        dxy = ((tip[0] - s[0]) ** 2 + (tip[1] - s[1]) ** 2) ** 0.5
        if dxy > 0.01:
            warnings.append("approach is %.1f cm off in XY from solder" % (dxy * 100))
    if name == "solder" and "approach" in info:
        a = info["approach"]["tip_xyz"]
        dxy = ((tip[0] - a[0]) ** 2 + (tip[1] - a[1]) ** 2) ** 0.5
        if dxy > 0.01:
            warnings.append("solder is %.1f cm off in XY from approach" % (dxy * 100))
    return warnings


def corners_summary(info):
    """Size, level difference and shape of the PCB from the 4 corners."""
    if not all(n in info for n in CORNERS):
        return
    pts = [info[n]["tip_xyz"] for n in CORNERS]
    print("\n=== PCB (pen tip at each corner) ===")
    turns = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        side = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        print("  side %d-%d: %.1f cm" % (i + 1, (i + 1) % 4 + 1, side * 100))
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    zs = [p[2] for p in pts]
    print("  center: x=%.3f y=%.3f m | surface height: %.1f to %.1f cm"
          % (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4, min(zs) * 100, max(zs) * 100))
    if not (all(g > 0 for g in turns) or all(g < 0 for g in turns)):
        print("  [warning] the corners do not form a convex quadrilateral in that order: check which is which")
    if max(zs) - min(zs) > 0.01:
        print("  [warning] more than 1 cm of level difference between corners: tilted PCB or one not touched?")


def main():
    p = argparse.ArgumentParser(description="Record cycle poses by guiding the arm by hand")
    p.add_argument("poses", nargs="*", help="poses to record (default: the 4 of the cycle)")
    p.add_argument("--corners", action="store_true", help="record the 4 corners of the PCB")
    p.add_argument("--pen-length", type=float, default=0.0,
                   help="cm the pen sticks out below frankapy's TCP (the point between the fingers)")
    args = p.parse_args()
    if not args.poses:
        args.poses = CORNERS if args.corners else ORDER
    for name in args.poses:
        if name not in ORDER + CORNERS:
            sys.exit("[error] unknown pose '%s'. Valid: %s" % (name, ", ".join(ORDER + CORNERS)))

    poses_path = os.path.join(DATA_DIR, "poses.json")
    info_path = os.path.join(DATA_DIR, "pose_info.json")
    table = load_json(os.path.join(DATA_DIR, "table.json"), None)
    if table is None:
        sys.exit("[error] %s is missing" % os.path.join(DATA_DIR, "table.json"))
    poses = load_json(poses_path, {})
    info = load_json(info_path, {})
    if os.path.exists(poses_path):
        backup = poses_path + time.strftime(".bak-%Y%m%d_%H%M%S")
        shutil.copy(poses_path, backup)
        print("Backup: %s" % backup)

    from frankapy import FrankaArm
    fa = FrankaArm()
    length = args.pen_length / 100.0

    print("\n=== Record poses: %s ===" % ", ".join(args.poses))
    print("Pen length below the TCP: %.1f cm" % args.pen_length)
    print("Hold the arm with one hand and keep the emergency stop at hand.")

    for name in args.poses:
        while True:
            print("\n--- %s: %s" % (name.upper(), DESCRIPTION[name]))
            r = input("Enter = start guide mode | s = skip this pose: ").strip().lower()
            if r == "s":
                break
            fa.run_guide_mode(duration=GUIDE_DURATION, block=False)
            print("Guide mode on (max. %.0f s). Move the arm to the pose." % GUIDE_DURATION)
            input("HOLD the arm and press Enter to lock it... ")
            fa.stop_skill()
            time.sleep(0.5)

            q = [round(float(x), 5) for x in fa.get_joints()]
            T = fa.get_pose()
            tip = pen_tip(T, length)
            print("q (rad):          %s" % q)
            print("frankapy TCP (m): x=%.3f y=%.3f z=%.3f" % tuple(T.translation))
            print("Tip (m):          x=%.3f y=%.3f z=%.3f  -> %.1f cm above the table"
                  % (tip[0], tip[1], tip[2], (tip[2] - table["table_z"]) * 100))
            for warning in check(name, tip, info, table):
                print("  [warning] " + warning)

            r = input("Enter = save | r = repeat | s = skip: ").strip().lower()
            if r == "r":
                continue
            if r == "s":
                break
            poses[name] = q
            info[name] = {
                "tcp_xyz": [round(float(v), 4) for v in T.translation],
                "tcp_quat_wxyz": [round(float(v), 5) for v in T.quaternion],
                "tip_xyz": [round(v, 4) for v in tip],
                "pen_length_cm": args.pen_length,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_json(poses_path, poses)
            save_json(info_path, info)
            print("Saved '%s' to %s" % (name, poses_path))
            break

    missing = [n for n in ORDER if n not in poses]
    print("\nDone. %s" % ("Missing: " + ", ".join(missing) if missing else "cycle poses complete."))
    corners_summary(info)


if __name__ == "__main__":
    main()
