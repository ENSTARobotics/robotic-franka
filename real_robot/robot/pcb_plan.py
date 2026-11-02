#!/usr/bin/env python3
""""Soldering" plan over the PCB from its 4 corners. Does not move the robot.

    python3 pcb_plan.py                       # check the plan (on the host or in the container)
    python3 pcb_plan.py --grid 2x3 --margin-cm 3 --height-mm 3

Reads data/poses.json (home, solder) and data/pose_info.json (corner_1..4). Each grid point is
solved with inverse kinematics to joint angles, to run it with goto_joints like the cycle.
The kinematics is the Panda's DH in pure Python; it was checked to match fa.get_pose() within
0.1 mm on the recorded poses.
"""
import argparse
import json
import math
import os
import sys

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
TCP_FRANKAPY = 0.1034       # m, from the flange to frankapy's default TCP (between the fingers)
DH = [(0, 0.333, 0), (0, 0, -math.pi / 2), (0, 0.316, math.pi / 2), (0.0825, 0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0, 0, math.pi / 2), (0.088, 0, math.pi / 2)]
LIM_LOW = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
LIM_HIGH = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
LIM_MARGIN = 0.05           # rad of margin to the joint limits

HEIGHT = 0.003              # m above the surface when "soldering"
LIFT = 0.06                 # m above the surface when moving between points
MARGIN = 0.03               # m from the edge of the PCB
WAIT = 1.5                  # s "soldering" at each point
DOWN_FACTOR = 1.5           # going down to the PCB takes 1.5 times longer (smoother)
MAX_DEVIATION = 0.01        # m, largest lateral deviation accepted when going up or down


# ---------- kinematics ----------

def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _dh(a, d, al, th):
    ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
    return [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]]


def fk(q, length):
    """Position of the pen tip and rotation of the tool."""
    T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for (a, d, al), th in zip(DH, q):
        T = _mul(T, _dh(a, d, al, th))
    T = _mul(T, _dh(0, 0.107 + TCP_FRANKAPY + length, 0, 0))
    return [T[i][3] for i in range(3)], [row[:3] for row in T[:3]]


def _error(q, p_target, R_target, length):
    p, R = fk(q, length)
    e = [p_target[i] - p[i] for i in range(3)]
    rot = [0.0, 0.0, 0.0]
    for c in range(3):
        a = [R[i][c] for i in range(3)]
        b = [R_target[i][c] for i in range(3)]
        rot[0] += 0.5 * (a[1] * b[2] - a[2] * b[1])
        rot[1] += 0.5 * (a[2] * b[0] - a[0] * b[2])
        rot[2] += 0.5 * (a[0] * b[1] - a[1] * b[0])
    return e + rot


def _solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        f = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[f] = M[f], M[c]
        for r in range(c + 1, n):
            k = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= k * M[c][j]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))) / M[r][r]
    return x


def ik(p_target, R_target, seed, length, iters=200):
    """Damped least squares from the seed (stays close to that posture)."""
    q = list(seed)
    h, lam2 = 1e-6, 1e-4
    for _ in range(iters):
        e = _error(q, p_target, R_target, length)
        if max(abs(x) for x in e[:3]) < 1e-5 and max(abs(x) for x in e[3:]) < 1e-4:
            return q
        # Numerical Jacobian of the reached position/orientation (= -d error / dq)
        J = []
        for j in range(7):
            qh = list(q)
            qh[j] += h
            eh = _error(qh, p_target, R_target, length)
            J.append([(e[i] - eh[i]) / h for i in range(6)])
        JJt = [[sum(J[k][i] * J[k][j] for k in range(7)) + (lam2 if i == j else 0.0) for j in range(6)]
               for i in range(6)]
        y = _solve(JJt, e)
        dq = [sum(J[k][i] * y[i] for i in range(6)) for k in range(7)]
        step = max(abs(x) for x in dq)
        if step > 0.1:
            dq = [x * 0.1 / step for x in dq]
        q = [q[k] + dq[k] for k in range(7)]
    return None


def within_limits(q):
    return all(lo + LIM_MARGIN <= x <= hi - LIM_MARGIN for x, lo, hi in zip(q, LIM_LOW, LIM_HIGH))


def check_segment(qa, qb, length, n=40):
    """Lowest height of the tip and lateral deviation from the straight line between the ends."""
    pa, _ = fk(qa, length)
    pb, _ = fk(qb, length)
    z_min, deviation = 9.0, 0.0
    for k in range(n + 1):
        s = k / float(n)
        p, _ = fk([x + s * (y - x) for x, y in zip(qa, qb)], length)
        z_min = min(z_min, p[2])
        line = [pa[i] + s * (pb[i] - pa[i]) for i in range(2)]
        deviation = max(deviation, math.hypot(p[0] - line[0], p[1] - line[1]))
    return z_min, deviation, pa, pb


# ---------- PCB ----------

def bilinear(E, u, v):
    """E = corners 1..4; u goes from 1 to 2 (y axis), v from 1 to 4 (x axis, toward the operator)."""
    return [(1 - u) * (1 - v) * E[0][i] + u * (1 - v) * E[1][i] + u * v * E[2][i] + (1 - u) * v * E[3][i]
            for i in range(3)]


def grid_points(E, rows, cols, margin):
    length_u = math.hypot(E[1][0] - E[0][0], E[1][1] - E[0][1])
    length_v = math.hypot(E[3][0] - E[0][0], E[3][1] - E[0][1])
    mu, mv = margin / length_u, margin / length_v
    if mu >= 0.5 or mv >= 0.5:
        raise ValueError("the margin is larger than half the PCB")

    def spread(n, m):
        return [0.5] if n == 1 else [m + (1 - 2 * m) * k / float(n - 1) for k in range(n)]

    points = []
    for i, v in enumerate(spread(rows, mv)):
        us = spread(cols, mu)
        for u in (us if i % 2 == 0 else us[::-1]):   # zigzag: shorter path
            points.append(bilinear(E, u, v))
    return points


def build_plan(poses, info, rows=2, cols=3, margin=MARGIN, height=HEIGHT, lift=LIFT,
               wait=WAIT, down_factor=DOWN_FACTOR):
    """Returns (plan, report). plan = list of {"name", "q", "wait", "factor"}."""
    missing = [n for n in ("corner_1", "corner_2", "corner_3", "corner_4", "solder") if n not in info]
    if missing or "home" not in poses or "solder" not in poses:
        raise ValueError("missing data: %s (record it with record_poses.py)" % ", ".join(missing or ["home/solder"]))
    length = info["corner_1"]["pen_length_cm"] / 100.0
    E = [info["corner_%d" % i]["tip_xyz"] for i in range(1, 5)]
    _, R = fk(poses["solder"], length)          # pen orientation recorded at the solder pose

    report = ["PCB: %dx%d grid, margin %.1f cm, solder at %.1f mm, move at %.1f cm, %.1f s per point"
              % (rows, cols, margin * 100, height * 1000, lift * 100, wait)]
    plan = [{"name": "home", "q": list(poses["home"]), "wait": 0.0, "factor": 1.0}]
    seed = list(poses["solder"])
    for k, surface in enumerate(grid_points(E, rows, cols, margin), 1):
        p_above = [surface[0], surface[1], surface[2] + lift]
        p_down = [surface[0], surface[1], surface[2] + height]
        q_above = ik(p_above, R, seed, length)
        q_down = ik(p_down, R, q_above, length) if q_above else None
        if q_above is None or q_down is None:
            raise ValueError("no IK solution for point p%d (%.3f, %.3f)" % (k, surface[0], surface[1]))
        for name, q in (("above", q_above), ("down", q_down)):
            if not within_limits(q):
                raise ValueError("p%d (%s) is close to a joint limit" % (k, name))
        seed = q_above
        report.append("  p%d: x=%.1f y=%.1f cm, surface z=%.1f cm"
                      % (k, surface[0] * 100, surface[1] * 100, surface[2] * 100))
        plan.append({"name": "above_p%d" % k, "q": q_above, "wait": 0.0, "factor": 1.0})
        plan.append({"name": "down_p%d" % k, "q": q_down, "wait": wait, "factor": down_factor})
        plan.append({"name": "up_p%d" % k, "q": q_above, "wait": 0.0, "factor": 1.0})
    plan.append({"name": "home", "q": list(poses["home"]), "wait": 0.0, "factor": 1.0})

    # Check each segment as goto_joints runs it (joint interpolation)
    z_surface = min(p[2] for p in E)
    worst_deviation = 0.0
    for a, b in zip(plan[1:-1], plan[2:-1]):    # segments between points (without the home ones)
        z_min, deviation, pa, pb = check_segment(a["q"], b["q"], length)
        if z_min < min(pa[2], pb[2]) - 0.002 or z_min < z_surface + height - 0.002:
            raise ValueError("segment %s -> %s goes down to z=%.1f cm" % (a["name"], b["name"], z_min * 100))
        vertical = b["name"].startswith(("down", "up"))
        if vertical:
            if deviation > MAX_DEVIATION:
                raise ValueError("segment %s -> %s deviates %.1f mm" % (a["name"], b["name"], deviation * 1000))
            worst_deviation = max(worst_deviation, deviation)
    for a, b in ((plan[0], plan[1]), (plan[-2], plan[-1])):
        z_min, _, pa, pb = check_segment(a["q"], b["q"], length)
        if z_min < min(pa[2], pb[2]) - 0.002:
            raise ValueError("segment %s -> %s goes below its ends" % (a["name"], b["name"]))
    report.append("Checked: %d segments, none goes below its ends, max. deviation going down/up %.1f mm"
                  % (len(plan), worst_deviation * 1000))
    return plan, report


def load(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "poses.json")) as f:
        poses = json.load(f)
    with open(os.path.join(data_dir, "pose_info.json")) as f:
        info = json.load(f)
    return poses, info


def grid(text):
    try:
        r, c = [int(x) for x in text.lower().split("x")]
        assert 1 <= r <= 5 and 1 <= c <= 5
        return r, c
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("use ROWSxCOLUMNS, e.g. 2x3 (max. 5x5)")


def add_arguments(p):
    p.add_argument("--grid", type=grid, default=(2, 3), help="rows x columns (default 2x3)")
    p.add_argument("--margin-cm", type=float, default=MARGIN * 100)
    p.add_argument("--height-mm", type=float, default=HEIGHT * 1000, help="soldering height above the PCB")
    p.add_argument("--lift-cm", type=float, default=LIFT * 100, help="height when moving between points")
    p.add_argument("--wait", type=float, default=WAIT, help="s soldering at each point")


def plan_from_args(args, poses, info):
    if not 2.0 <= args.height_mm <= 20.0:
        raise ValueError("--height-mm must be between 2 and 20")
    if not 3.0 <= args.lift_cm <= 15.0:
        raise ValueError("--lift-cm must be between 3 and 15")
    return build_plan(poses, info, args.grid[0], args.grid[1], args.margin_cm / 100.0,
                      args.height_mm / 1000.0, args.lift_cm / 100.0, max(0.0, args.wait))


def main():
    p = argparse.ArgumentParser(description="Check the soldering plan over the PCB (does not move the robot)")
    add_arguments(p)
    args = p.parse_args()
    poses, info = load()
    try:
        plan, report = plan_from_args(args, poses, info)
    except ValueError as e:
        sys.exit("[error] invalid plan: %s" % e)
    print("\n".join(report))
    print("Segments: " + ", ".join(s["name"] for s in plan))


if __name__ == "__main__":
    main()
