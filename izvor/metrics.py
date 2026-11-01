"""
Step 13b-3: all navigation metrics of step 13e, computed AFTER the run from its record.

    python -m evaluation.metrics ../results/runs/T1_A_realistic_s1        (one run, prints + metrics.json)
    python -m evaluation.metrics --all                                     (every run folder without metrics.json)

The robot code never computes these (GT is evaluation only, and computing them after
the run keeps the live loop fast). Inputs: meta.json, summary.json, ticks.csv.gz,
events.csv, map_topo.json, scans.npz of a run (see experiment.recorder) and the maze
geometry (mazes.maze_geometry). Definitions: docs/korak13.md, section 13e.

  S1-S4 mission         S2 also checks the body clearance from the GT trajectory
  L1-L4 localisation    (L5 NEES needs the EKF covariance -> configurations B-E)
  T1-T4 topological map
  K1    metric map      LiDAR points at the ESTIMATED pose vs the true walls
  K2    occupancy grid  robot's grid (map_grid.npz, or rebuilt from scans.npz) vs the true maze (step 14)
  E1-E3 efficiency
  P1-P8 trajectory quality
  R1/R2 are over many runs: evaluation.metrics.aggregate()
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from mazes.maze_geometry import WallMap
from mazes.maze_spec import HALF_FREE_WIDTH_M, MAZES, START_BACK_CLEARANCE_M

ROOT = Path(__file__).resolve().parents[2]
BODY_POINTS = np.array(((0.335, 0.0), (0.33, 0.09), (0.33, -0.09), (0.15, 0.17), (0.15, -0.17),
                        (-0.10, 0.17), (-0.10, -0.17), (0.0, 0.17), (0.0, -0.17)))
R_ASSOC_M = 0.373                     # MazeMemory association radius (T4 reserve)
STRAIGHT_STATES = ("FOLLOW", "APPROACH", "EXIT")
CLEAR_WARN_M = 0.050                  # P7: share of time closer than this
OSC_MIN_W = 0.01                      # P6: |omega_ref| above this counts for a sign change
DT = 0.01


# --------------------------------------------------------------------------- geometry (numpy)
class Geometry:
    def __init__(self, spec) -> None:
        self.spec = spec
        wm = WallMap(spec)
        self.V = np.array(wm.v, dtype=float).reshape(-1, 3)        # x, y0, y1
        self.H = np.array(wm.h, dtype=float).reshape(-1, 3)        # y, x0, x1
        self.rects = np.array([r[:4] for r in wm.rects], dtype=float)
        nodes = spec.nodes_m
        segs = [(*nodes[a], *nodes[b]) for a, b in spec.edges]
        rx, ry = nodes[spec.root]
        sx, sy, _ = spec.start_pose
        segs.append((rx, ry, sx, sy - START_BACK_CLEARANCE_M + HALF_FREE_WIDTH_M))   # START approach
        self.centre = np.array(segs, dtype=float)
        self.node_xy = np.array(list(nodes.values()) + [(sx, sy)], dtype=float)

    def dist_walls(self, px, py, chunk=20000):
        out = np.empty(len(px))
        for i in range(0, len(px), chunk):
            x, y = px[i:i + chunk, None], py[i:i + chunk, None]
            best = np.full(len(x), np.inf)
            if len(self.V):
                X, y0, y1 = self.V[:, 0], self.V[:, 1], self.V[:, 2]
                dy = np.where(y < y0, y0 - y, np.where(y > y1, y - y1, 0.0))
                best = np.minimum(best, np.hypot(x - X, dy).min(axis=1))
            if len(self.H):
                Y, x0, x1 = self.H[:, 0], self.H[:, 1], self.H[:, 2]
                dx = np.where(x < x0, x0 - x, np.where(x > x1, x - x1, 0.0))
                best = np.minimum(best, np.hypot(dx, y - Y).min(axis=1))
            out[i:i + chunk] = best
        return out

    def is_free(self, px, py, chunk=20000):
        out = np.empty(len(px), dtype=bool)
        r = self.rects
        for i in range(0, len(px), chunk):
            x, y = px[i:i + chunk, None], py[i:i + chunk, None]
            out[i:i + chunk] = ((x >= r[:, 0] - 1e-9) & (x <= r[:, 1] + 1e-9) &
                                (y >= r[:, 2] - 1e-9) & (y <= r[:, 3] + 1e-9)).any(axis=1)
        return out

    @staticmethod
    def seg_dist(px, py, segs, chunk=20000):
        """Distance to the nearest of segs (N x 4) and the index of that segment."""
        d_out, i_out = np.empty(len(px)), np.empty(len(px), dtype=int)
        ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
        vx, vy = bx - ax, by - ay
        L2 = np.maximum(vx * vx + vy * vy, 1e-12)
        for i in range(0, len(px), chunk):
            x, y = px[i:i + chunk, None], py[i:i + chunk, None]
            t = np.clip(((x - ax) * vx + (y - ay) * vy) / L2, 0.0, 1.0)
            d = np.hypot(x - (ax + t * vx), y - (ay + t * vy))
            j = d.argmin(axis=1)
            d_out[i:i + chunk] = d[np.arange(len(j)), j]
            i_out[i:i + chunk] = j
        return d_out, i_out


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rms(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v * v))) if v.size else None


def _max(v):
    v = np.asarray(v, dtype=float)
    return float(np.max(v)) if v.size else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------- loading
def load(folder: Path) -> dict:
    folder = Path(folder)
    d = {"dir": folder, "meta": json.loads((folder / "meta.json").read_text())}
    d["summary"] = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
    with gzip.open(folder / "ticks.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    num = ("t", "v_ref", "w_ref", "cmd_l", "cmd_r", "est_x", "est_y", "est_th", "gt_x", "gt_y", "gt_th", "collision",
           "P_xx", "P_xy", "P_yy", "P_xt", "P_yt", "P_tt", "ekf_rej")
    d["ticks"] = {k: np.array([_f(r.get(k)) for r in rows]) for k in num}
    for k in ("state", "nav_phase", "mission_phase"):
        d["ticks"][k] = np.array([r.get(k, "") for r in rows])
    d["events"] = []
    if (folder / "events.csv").exists():
        with (folder / "events.csv").open() as f:
            d["events"] = list(csv.DictReader(f))
    d["map"] = json.loads((folder / "map_topo.json").read_text()) if (folder / "map_topo.json").exists() else None
    d["scans"] = None
    if (folder / "scans.npz").exists():
        from experiment.recorder import load_scans
        d["scans"] = load_scans(folder)
    return d


def _scan_offsets(meta):
    m = (meta.get("lidar") or {}).get("mount") or {}
    ext = (meta.get("lidar") or {}).get("extrinsic")
    if ext is None:                                     # runs recorded before 13b-3: the calibration file
        p = ROOT / "results" / "lidar_calibration.json"
        cal = json.loads(p.read_text()) if p.exists() else {}
        ext = (cal.get("extrinsic_forward_m", 0.0), cal.get("extrinsic_left_m", 0.0), cal.get("extrinsic_yaw_rad", 0.0))
    return (m.get("x_forward_m", 0.1815) + ext[0], m.get("y_left_m", 0.0014) + ext[1],
            m.get("yaw_offset_rad", 0.0) + ext[2])


# --------------------------------------------------------------------------- metrics of one run
def compute(folder: Path, *, scan_step: int = 5, ray_step: int = 4) -> dict:
    d = load(folder)
    meta, summ, T = d["meta"], d["summary"], d["ticks"]
    spec = MAZES[meta["maze"]["id"]]
    geo = Geometry(spec)
    n = len(T["t"])
    gx, gy, gth = T["gt_x"], T["gt_y"], T["gt_th"]
    ex, ey, eth = T["est_x"], T["est_y"], T["est_th"]
    step = np.hypot(np.diff(gx, prepend=gx[0]), np.diff(gy, prepend=gy[0]))
    path = float(step.sum())
    out = {"run": d["dir"].name, "maze": spec.maze_id, "configuration": meta["configuration"]["name"],
           "noise": meta["noise"]["profile"], "seed": meta["noise"]["seed"], "mode": meta.get("mode"),
           "ticks": n, "path_length_gt_m": path}

    # ---- body clearance (P7, S2)
    c, s = np.cos(gth), np.sin(gth)
    bx = gx[:, None] + c[:, None] * BODY_POINTS[:, 0] - s[:, None] * BODY_POINTS[:, 1]
    by = gy[:, None] + s[:, None] * BODY_POINTS[:, 0] + c[:, None] * BODY_POINTS[:, 1]
    flat_x, flat_y = bx.ravel(), by.ravel()
    clr = geo.dist_walls(flat_x, flat_y)
    clr[~geo.is_free(flat_x, flat_y)] = 0.0
    clear = clr.reshape(n, -1).min(axis=1)
    dyn = _dynamic_clearance(d, T["t"], bx, by)          # step 14b: obstacles that were present
    if dyn is not None:
        clear = np.minimum(clear, dyn)
    out.update(_dynamic_metrics(d, dyn))
    if dyn is not None and "ekf_rej" in T and np.isfinite(T["ekf_rej"]).any():
        # EKF measurements rejected by the gate while an obstacle was within 0.6 m of the body (step 16)
        rej = np.diff(np.nan_to_num(T["ekf_rej"]), prepend=0.0) > 0
        near = np.isfinite(dyn) & (dyn < 0.6)
        out["DYN_ekf_rejected_near"] = int(np.count_nonzero(rej & near))
        out["DYN_ekf_rejected_elsewhere"] = int(np.count_nonzero(rej & ~near))

    # ---- S
    collided = bool(np.nansum(T["collision"]) > 0) or bool(clear.min() <= 0.0)
    s1 = summ.get("S1_explored")
    s3 = summ.get("S3_goal_reached")
    out.update({
        "S1_explored": s1, "S2_no_collision": not collided, "S3_goal_reached": s3,
        "S4_mission_success": bool(s1 and not collided and (s3 is not False)),
        "safety_abort": summ.get("safety_abort"), "fault": summ.get("fault"),
        "mission_reason": summ.get("mission_reason"), "explore_reason": summ.get("explore_reason"),
    })

    # ---- L
    pe = np.hypot(ex - gx, ey - gy)
    he = np.degrees(np.abs(_wrap(eth - gth)))
    turns = _turn_segments(T["state"])
    l3 = _junction_errors(d, spec)
    out.update({
        "L1_pos_rmse_m": _rms(pe), "L1_pos_max_m": _max(pe), "L1_pos_final_m": float(pe[-1]),
        "L2_heading_rmse_deg": _rms(he), "L2_heading_max_deg": _max(he),
        "L2_heading_final_deg": float(np.degrees(_wrap(eth[-1] - gth[-1]))),
        "L3_junction_err_mean_m": float(np.mean(l3)) if l3 else None, "L3_junction_err_max_m": _max(l3),
        "L4_pos_drift_pct": 100.0 * float(pe[-1]) / path if path > 0 else None,
        "L4_heading_drift_deg_per_turn": float(np.max(he)) / len(turns) if turns else None,
    })
    out.update(_nees(T, ex - gx, ey - gy, _wrap(eth - gth)))

    # ---- T
    match = [_f(e["match_distance"]) for e in d["events"] if e.get("association") == "MATCH"]
    exp_cycles = spec.expected_cycle_count
    out.update({
        "T1_map_correct": summ.get("T_map_correct"), "T2_duplicates": summ.get("T_duplicates"),
        "T2_false_merges": summ.get("T_false_merges"), "T_edges_wrong": summ.get("T_edges_wrong"),
        "T3_loop_closures": summ.get("T_loop_closures"), "T3_expected_cycles": exp_cycles,
        "T4_closure_reserve_m": (R_ASSOC_M - max(match)) if match else None,
        "T_memory_junctions": summ.get("T_memory_junctions"), "T_true_nodes_visited": summ.get("T_true_nodes_visited"),
    })

    # ---- K1 (+ LiDAR sensor check at the GT pose)
    out.update(_lidar_metrics(d, geo, scan_step, ray_step))

    # ---- K2 occupancy grid
    out.update(_grid_metrics(d, geo, spec))

    # ---- E
    out.update({k: summ.get(k) for k in ("E1_explore_s", "E2_goal_run_s", "E3_route_ratio",
                                         "goal_run_length_gt_m", "optimal_length_m")})

    # ---- P
    straight = np.isin(T["state"], STRAIGHT_STATES) & (T["v_ref"] > 0)
    # P1: distance to the centreline of the corridor the robot drives along (segments parallel to its
    # heading); P2: heading vs the corridor axis = the nearest multiple of 90 deg (axis-aligned mazes)
    horiz = np.abs(np.cos(gth)) >= np.abs(np.sin(gth))
    cs = geo.centre
    is_h = np.abs(cs[:, 3] - cs[:, 1]) < 1e-9
    cd = np.full(n, np.nan)
    for mask, segs in ((horiz, cs[is_h]), (~horiz, cs[~is_h])):
        if mask.any() and len(segs):
            cd[mask], _ = Geometry.seg_dist(gx[mask], gy[mask], segs)
    ang = np.degrees(np.abs(_wrap(4 * gth) / 4))
    out.update({
        "P1_centreline_rms_mm": 1000 * _rms(cd[straight]) if straight.any() else None,
        "P1_centreline_max_mm": 1000 * _max(cd[straight]) if straight.any() else None,
        "P2_axis_angle_rms_deg": _rms(ang[straight]) if straight.any() else None,
        "P2_axis_angle_max_deg": _max(ang[straight]) if straight.any() else None,
    })
    out.update(_ideal_path(d, spec, gx, gy, path))
    # each turn belongs to the last junction event before it; a turn without one is a step-14b obstacle
    # U-turn in the middle of a corridor (not a junction turn)
    ev_tick = np.array([int(_f(e["tick"])) for e in d["events"]]) if d["events"] else np.zeros(0, int)
    ev_dec = [e["decision"] for e in d["events"]]
    kcol = T["k"] if "k" in T else np.arange(1, n + 1)
    tacc, tpos, prev_k = [], [], -1
    for k, (a, b) in enumerate(turns):
        th_end = gth[min(b + 1, n - 1)]
        tacc.append(abs(math.degrees(math.remainder(th_end, math.pi / 2))))
        ka = int(kcol[a])
        idx = np.nonzero((ev_tick > prev_k) & (ev_tick <= ka))[0]
        prev_k = ka
        dec = ev_dec[idx[-1]] if idx.size else None
        if dec in ("LEFT", "RIGHT"):     # junction turns (dead ends: as close as safe)
            tpos.append(float(np.hypot(geo.node_xy[:, 0] - gx[a], geo.node_xy[:, 1] - gy[a]).min()))
    out.update({
        "turns": len(turns),
        "P4_turn_err_rms_deg": _rms(tacc), "P4_turn_err_max_deg": _max(tacc),
        "junction_turns": len(tpos),
        "P5_turn_point_rms_mm": 1000 * _rms(tpos) if tpos else None,
        "P5_turn_point_max_mm": 1000 * _max(tpos) if tpos else None,
    })
    w = T["w_ref"]
    dw = np.diff(w) / DT
    sm = straight[1:] & straight[:-1]
    sig = np.sign(np.where(np.abs(w) > OSC_MIN_W, w, 0.0))
    nz = sig[straight & (sig != 0)]
    osc = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size > 1 else 0
    straight_len = float(step[straight].sum())
    out.update({
        "P6_domega_rms_rad_s2": _rms(dw[sm]) if sm.any() else None,
        "P6_oscillations_per_m": osc / straight_len if straight_len > 0 else None,
        "P7_clearance_min_mm": 1000 * float(clear.min()), "P7_clearance_mean_mm": 1000 * float(clear.mean()),
        "P7_time_below_50mm_pct": 100.0 * float(np.mean(clear < CLEAR_WARN_M)),
        "P8_effort_per_m": float(np.sum(0.5 * (np.abs(T["cmd_l"]) + np.abs(T["cmd_r"]))) * DT / path) if path else None,
    })
    return out


def _turn_segments(state):
    turns, a = [], None
    for i, s in enumerate(state):
        if s == "TURN" and a is None:
            a = i
        elif s != "TURN" and a is not None:
            turns.append((a, i - 1))
            a = None
    return turns


DEAD_END_STOP_M = 0.335 + 0.057581      # axle -> end wall when FRONT_BLOCKED stops the robot


def _true_xy(spec, node):
    if node == "__START__":                 # the U-turn point in the START dead end
        sx, sy, _ = spec.start_pose
        return sx, sy - START_BACK_CLEARANCE_M + DEAD_END_STOP_M
    return spec.nodes_m.get(node)


def _junction_errors(d, spec):
    if not d["map"]:
        return []
    label = {}
    for e in d["events"]:
        label.setdefault(int(e["junction"]), e["gt_node"])
    errs = []
    for j in d["map"]["junctions"]:
        if j["id"] == 0 or j["id"] not in label:
            continue
        xy = _true_xy(spec, label[j["id"]])
        if xy is not None and label[j["id"]] != "__START__":
            errs.append(math.hypot(j["x"] - xy[0], j["y"] - xy[1]))
    return errs


def _ideal_path(d, spec, gx, gy, path):
    """P3: the ideal path = corridor centrelines through the same true nodes, in the same order."""
    sx, sy, _ = spec.start_pose
    pts = [(sx, sy)]
    for e in d["events"]:
        xy = _true_xy(spec, e["gt_node"])
        if xy is not None and (abs(xy[0] - pts[-1][0]) > 1e-9 or abs(xy[1] - pts[-1][1]) > 1e-9):
            pts.append(xy)
    if len(pts) < 2:
        return {"P3_ideal_dev_mean_mm": None, "P3_ideal_dev_max_mm": None, "P3_length_ratio": None}
    segs = np.array([(*pts[i], *pts[i + 1]) for i in range(len(pts) - 1)])
    dd, _ = Geometry.seg_dist(gx, gy, segs)
    ideal = float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum())
    return {"P3_ideal_dev_mean_mm": 1000 * float(dd.mean()), "P3_ideal_dev_max_mm": 1000 * float(dd.max()),
            "P3_ideal_length_m": ideal, "P3_length_ratio": path / ideal if ideal > 0 else None}


def _nees(T, ex, ey, eth, step: int = 10):
    """L5: normalised estimation error squared e^T P^-1 e of [x, y, theta] (Bar-Shalom). For a consistent
    filter it is chi-square with 3 dof: mean 3, 95 % of the samples below 7.815."""
    if "P_xx" not in T or not np.isfinite(T["P_xx"]).any():
        return {"L5_nees_mean": None, "L5_nees_in_95pct": None}
    i = np.arange(0, len(ex), step)
    i = i[np.isfinite(T["P_xx"][i])]
    P = np.zeros((len(i), 3, 3))
    P[:, 0, 0], P[:, 1, 1], P[:, 2, 2] = T["P_xx"][i], T["P_yy"][i], T["P_tt"][i]
    P[:, 0, 1] = P[:, 1, 0] = T["P_xy"][i]
    P[:, 0, 2] = P[:, 2, 0] = T["P_xt"][i]
    P[:, 1, 2] = P[:, 2, 1] = T["P_yt"][i]
    e = np.stack([ex[i], ey[i], eth[i]], axis=1)
    nees = np.einsum("ni,nij,nj->n", e, np.linalg.inv(P), e)
    return {"L5_nees_mean": float(np.mean(nees)), "L5_nees_median": float(np.median(nees)),
            "L5_nees_in_95pct": 100.0 * float(np.mean(nees <= 7.815))}


def _dynamic_clearance(d, t, bx, by):
    """Per tick: smallest distance of the body outline to the dynamic obstacles present (0 = inside),
    None for runs without obstacles."""
    p = d["dir"] / "dynamic.json"
    if not p.exists():
        return None
    from mazes.dynamic import DynObstacle
    obs = [DynObstacle.from_dict(o) for o in json.loads(p.read_text())["obstacles"]]
    out = np.full(len(t), np.inf)
    for o in obs:
        if o.triggered_t is None:
            continue
        for i in np.nonzero(t >= o.triggered_t)[0]:
            r = o.rect(float(t[i]))
            if r is None:
                continue
            dx = np.maximum.reduce([r[0] - bx[i], np.zeros_like(bx[i]), bx[i] - r[1]])
            dy = np.maximum.reduce([r[2] - by[i], np.zeros_like(by[i]), by[i] - r[3]])
            out[i] = min(out[i], float(np.hypot(dx, dy).min()))
    return out


def _dynamic_metrics(d, dyn_clear):
    p = d["dir"] / "dynamic.json"
    if not p.exists():
        return {"DYN_scenario": None}
    z = json.loads(p.read_text())
    log = z["log"]
    waits, t_stop, det = [], None, []
    for e in log:
        if e["event"] == "obstacle_stop":
            t_stop = e["t"]
            if e.get("front_m") is not None:
                det.append(e["front_m"])
        elif e["event"] in ("obstacle_resume", "obstacle_turn_back", "goal_blocked") and t_stop is not None:
            waits.append(e["t"] - t_stop)
            t_stop = None
    trig = [o["name"] for o in z["obstacles"] if o.get("triggered_t") is not None]
    from mazes.dynamic import DynObstacle
    obs = [DynObstacle.from_dict(o) for o in z["obstacles"]]
    false_stops = 0
    for e in log:
        if e["event"] != "obstacle_stop":
            continue
        near = False
        for o in obs:
            r = o.rect(e["t"])
            if r is not None:
                dx = max(r[0] - e["gt_x"], 0.0, e["gt_x"] - r[1])
                dy = max(r[2] - e["gt_y"], 0.0, e["gt_y"] - r[3])
                near |= math.hypot(dx, dy) < 1.0
        false_stops += not near
    finite = dyn_clear[np.isfinite(dyn_clear)] if dyn_clear is not None else np.array([])
    return {"DYN_scenario": z["scenario"], "DYN_placed": [o["name"] for o in z["obstacles"]], "DYN_triggered": trig,
            "DYN_stops": sum(e["event"] == "obstacle_stop" for e in log),
            "DYN_resumes": sum(e["event"] == "obstacle_resume" for e in log),
            "DYN_turnbacks": sum(e["event"] == "obstacle_turn_back" for e in log),
            "DYN_false_stops": false_stops,
            "DYN_goal_blocked": any(e["event"] == "goal_blocked" for e in log),
            "DYN_wait_s": waits, "DYN_stop_front_m": det,
            "DYN_min_clearance_mm": 1000 * float(finite.min()) if finite.size else None,
            "DYN_blocked": z.get("blocked", [])}


K2_TOL_M = 0.03                       # occupied cell = within 3 cm of a true wall surface (LiDAR +-30 mm 2 sigma)


def rebuild_grid(d):
    """The robot's grid from the record (runs recorded before step 14): same OccupancyGrid, same scans,
    same estimated poses as online."""
    from mapping.occupancy_grid import OccupancyGrid
    z = d["scans"]
    if z is None or len(z["t"]) == 0:
        return None
    fwd, left, yaw = _scan_offsets(d["meta"])
    P, R = z["est_pose"], z["ranges_measured"]
    g = OccupancyGrid(float(P[0, 0]), float(P[0, 1]))
    for i in range(len(P)):
        x, y, th = (float(v) for v in P[i])
        g.update(x + fwd * math.cos(th) - left * math.sin(th), y + fwd * math.sin(th) + left * math.cos(th),
                 th + yaw, z["angles"], R[i])
    return g


def _grid_metrics(d, geo, spec):
    """K2: cell-wise agreement of the robot's occupancy grid with the true maze.
    known cells = free or occupied;  free correct = cell centre in free space (or within K2_TOL of a wall);
    occupied correct = within K2_TOL of a true wall surface.  K2 agreement = correct / known.
    Coverage = true free cells (farther than K2_TOL from walls) that the grid has as free."""
    from mapping.occupancy_grid import OccupancyGrid
    none = {"K2_agreement_pct": None, "K2_occ_precision_pct": None, "K2_free_precision_pct": None,
            "K2_coverage_pct": None, "K2_cells_known": 0}
    path = d["dir"] / "map_grid.npz"
    g = OccupancyGrid.load(path) if path.exists() else rebuild_grid(d)
    if g is None:
        return none
    X, Y = g.cell_centres()
    occ, free = g.occupied(), g.free()
    known = occ | free
    if not known.any():
        return none
    px, py = X[known], Y[known]
    dw = geo.dist_walls(px, py)
    inside = geo.is_free(px, py)
    o = occ[known]
    ok_occ = o & (dw <= K2_TOL_M)
    ok_free = ~o & (inside | (dw <= K2_TOL_M))
    # true free cells of the whole maze at the grid resolution
    r = geo.rects
    xs = np.arange(r[:, 0].min(), r[:, 1].max(), g.res) + 0.5 * g.res
    ys = np.arange(r[:, 2].min(), r[:, 3].max(), g.res) + 0.5 * g.res
    TX, TY = (a.ravel() for a in np.meshgrid(xs, ys))
    tf = geo.is_free(TX, TY)
    TX, TY = TX[tf], TY[tf]
    tf = geo.dist_walls(TX, TY) > K2_TOL_M
    TX, TY = TX[tf], TY[tf]
    iy, ix = g.cell_of(TX, TY)
    ny, nx = g.shape
    inb = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    covered = np.zeros(len(TX), dtype=bool)
    covered[inb] = free[iy[inb], ix[inb]]
    _draw_grid(g, geo, d["dir"] / "map_grid.png", spec)
    return {"K2_agreement_pct": 100.0 * float((ok_occ | ok_free).sum()) / int(known.sum()),
            "K2_occ_precision_pct": 100.0 * float(ok_occ.sum()) / max(1, int(o.sum())),
            "K2_free_precision_pct": 100.0 * float(ok_free.sum()) / max(1, int((~o).sum())),
            "K2_coverage_pct": 100.0 * float(covered.mean()) if len(covered) else None,
            "K2_cells_known": int(known.sum()), "K2_resolution_m": g.res}


def _draw_grid(g, geo, path, spec):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x0, x1, y0, y1 = g.extent()
    img = np.full(g.shape, 0.5)
    img[g.free()] = 1.0
    img[g.occupied()] = 0.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("0.5")                      # outside the grid = unknown (grey), not white
    ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for X, a, b in geo.V:
        ax.plot([X, X], [a, b], color="tab:red", lw=0.6)
    for Y, a, b in geo.H:
        ax.plot([a, b], [Y, Y], color="tab:red", lw=0.6)
    r = geo.rects
    pad = 0.5
    ax.set_xlim(min(x0, r[:, 0].min() - pad), max(x1, r[:, 1].max() + pad))
    ax.set_ylim(min(y0, r[:, 2].min() - pad), max(y1, r[:, 3].max() + pad))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{spec.maze_id}: occupancy grid (black occupied, white free, grey unknown), true walls red", fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _lidar_metrics(d, geo, scan_step, ray_step):
    z = d["scans"]
    if z is None or len(z["t"]) == 0:
        return {"K1_wall_rmse_mm": None, "K1_points": 0, "lidar_gt_median_mm": None, "lidar_gt_p90_mm": None}
    fwd, left, yaw = _scan_offsets(d["meta"])
    ang = z["angles"][::ray_step]
    res = {}
    for key, pose_k, rng_k in (("est", "est_pose", "ranges_measured"), ("gt", "gt_pose", "ranges_ideal")):
        P = z[pose_k][::scan_step]
        R = z[rng_k][::scan_step, ::ray_step]
        x, y, th = P[:, 0:1], P[:, 1:2], P[:, 2:3]
        cx = x + fwd * np.cos(th) - left * np.sin(th)
        cy = y + fwd * np.sin(th) + left * np.cos(th)
        a = th + yaw + ang[None, :]
        px, py = (cx + R * np.cos(a)).ravel(), (cy + R * np.sin(a)).ravel()
        ok = np.isfinite(px)
        dist = geo.dist_walls(px[ok], py[ok])
        res[key] = dist
    e, g = res["est"], res["gt"]
    return {"K1_wall_rmse_mm": 1000 * _rms(e), "K1_wall_median_mm": 1000 * float(np.median(e)),
            "K1_within_2cm_pct": 100.0 * float(np.mean(e < 0.02)), "K1_points": int(e.size),
            "lidar_gt_median_mm": 1000 * float(np.median(g)), "lidar_gt_p90_mm": 1000 * float(np.percentile(g, 90))}


# --------------------------------------------------------------------------- many runs
def aggregate(rows: list[dict]) -> dict:
    """R1 success rate (and mean / sd of the main metrics) over runs of one configuration x noise x maze."""
    keys = ("L1_pos_rmse_m", "L1_pos_final_m", "L2_heading_rmse_deg", "P1_centreline_rms_mm", "P3_length_ratio",
            "E1_explore_s", "E3_route_ratio", "K1_wall_rmse_mm", "K2_agreement_pct")
    out = {"runs": len(rows), "R1_success_rate_pct": 100.0 * sum(bool(r.get("S4_mission_success")) for r in rows) / len(rows)
           if rows else None}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_mean"] = float(np.mean(v)) if v else None
        out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return out


# --------------------------------------------------------------------------- report
def report(m: dict) -> str:
    def f(v, fmt="{:.1f}"):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "NO"
        if isinstance(v, float):
            return fmt.format(v)
        return str(v)
    lines = [f"\nMetrics {m['run']}  ({m['maze']}, configuration {m['configuration']}, noise {m['noise']}, "
             f"seed {m['seed']}, {m.get('mode')})", "-" * 78,
             f"S  mission        : S1 {f(m['S1_explored'])}  S2 {f(m['S2_no_collision'])}  S3 {f(m['S3_goal_reached'])}"
             f"  ->  S4 {'SUCCESS' if m['S4_mission_success'] else 'FAILED'}",
             f"L1 position error : RMSE {f(1000 * m['L1_pos_rmse_m'], '{:.0f}')} mm, max "
             f"{f(1000 * m['L1_pos_max_m'], '{:.0f}')} mm, final {f(1000 * m['L1_pos_final_m'], '{:.0f}')} mm",
             f"L2 heading error  : RMSE {f(m['L2_heading_rmse_deg'], '{:.2f}')} deg, max "
             f"{f(m['L2_heading_max_deg'], '{:.2f}')} deg, final {f(m['L2_heading_final_deg'], '{:+.2f}')} deg",
             f"L3 junction error : mean {f(1000 * (m['L3_junction_err_mean_m'] or 0), '{:.0f}')} mm, max "
             f"{f(1000 * (m['L3_junction_err_max_m'] or 0), '{:.0f}')} mm (map junction centre vs true node)",
             f"L4 drift          : {f(m['L4_pos_drift_pct'], '{:.2f}')} % of the path, "
             f"{f(m['L4_heading_drift_deg_per_turn'], '{:.3f}')} deg per turn (max heading error / {m['turns']} turns)",
             f"L5 consistency    : NEES mean {f(m.get('L5_nees_mean'), '{:.2f}')} (consistent filter: 3), "
             f"{f(m.get('L5_nees_in_95pct'))} % of the time inside the 95 % bound",
             f"T  topological map: {'correct' if m['T1_map_correct'] else 'INCORRECT'}, duplicates "
             f"{m['T2_duplicates'] or 'none'}, false merges {m['T2_false_merges'] or 'none'}, loop closures "
             f"{m['T3_loop_closures']}/{m['T3_expected_cycles']}, closure reserve "
             f"{f(m['T4_closure_reserve_m'], '{:.3f}')} m",
             f"K1 metric map     : LiDAR points at the estimated pose vs walls RMSE {f(m['K1_wall_rmse_mm'])} mm, "
             f"{f(m.get('K1_within_2cm_pct'))} % within 2 cm  (sensor at GT pose: median "
             f"{f(m['lidar_gt_median_mm'])} mm, p90 {f(m['lidar_gt_p90_mm'])} mm)",
             f"K2 occupancy grid : agreement {f(m.get('K2_agreement_pct'))} %, occupied cells on a wall "
             f"{f(m.get('K2_occ_precision_pct'))} %, free cells in free space {f(m.get('K2_free_precision_pct'))} %, "
             f"coverage {f(m.get('K2_coverage_pct'))} %",
             *([f"DYN dynamic      : {m['DYN_scenario']}: placed {','.join(m['DYN_placed']) or '-'}, met "
                f"{','.join(m['DYN_triggered']) or '-'}; stops {m['DYN_stops']}, resumed {m['DYN_resumes']}, "
                f"turned back {m['DYN_turnbacks']}, false stops {m['DYN_false_stops']}, waits {', '.join(f'{w:.1f}' for w in m['DYN_wait_s']) or '-'} s, "
                f"min clearance to obstacles {f(m['DYN_min_clearance_mm'])} mm; EKF rejections near obstacles "
                f"{m.get('DYN_ekf_rejected_near', '-')} (elsewhere {m.get('DYN_ekf_rejected_elsewhere', '-')})"]
               if m.get("DYN_scenario") else []),
             f"E  efficiency     : exploration {f(m['E1_explore_s'], '{:.0f}')} s, goal run "
             f"{f(m['E2_goal_run_s'], '{:.0f}')} s, route ratio {f(m['E3_route_ratio'], '{:.3f}')}",
             f"P1 centreline     : RMS {f(m['P1_centreline_rms_mm'])} mm, max {f(m['P1_centreline_max_mm'])} mm",
             f"P2 axis angle     : RMS {f(m['P2_axis_angle_rms_deg'], '{:.2f}')} deg, max "
             f"{f(m['P2_axis_angle_max_deg'], '{:.2f}')} deg",
             f"P3 ideal path     : mean {f(m['P3_ideal_dev_mean_mm'])} mm, max {f(m['P3_ideal_dev_max_mm'])} mm, "
             f"length ratio {f(m['P3_length_ratio'], '{:.3f}')}",
             f"P4 turn accuracy  : RMS {f(m['P4_turn_err_rms_deg'], '{:.2f}')} deg, max "
             f"{f(m['P4_turn_err_max_deg'], '{:.2f}')} deg (heading after the turn vs the 90 deg grid)",
             f"P5 turn point     : RMS {f(m['P5_turn_point_rms_mm'])} mm, max {f(m['P5_turn_point_max_mm'])} mm "
             f"({m['junction_turns']} junction turns; axle vs junction centre)",
             f"P6 smoothness     : d(omega)/dt RMS {f(m['P6_domega_rms_rad_s2'], '{:.3f}')} rad/s^2, "
             f"{f(m['P6_oscillations_per_m'], '{:.2f}')} steering sign changes per m",
             f"P7 clearance      : min {f(m['P7_clearance_min_mm'])} mm, mean {f(m['P7_clearance_mean_mm'])} mm, "
             f"{f(m['P7_time_below_50mm_pct'], '{:.2f}')} % of the time below 50 mm",
             f"P8 drive effort   : {f(m['P8_effort_per_m'], '{:.3f}')} (|u| s per m)"]
    return "\n".join(lines)


def evaluate(folder: Path, quiet: bool = False) -> dict:
    m = compute(folder)
    (Path(folder) / "metrics.json").write_text(json.dumps(m, indent=1, default=_json_default))
    if not quiet:
        print(report(m))
    return m


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folders (results/runs/...)")
    ap.add_argument("--all", action="store_true", help="every run folder that has no metrics.json yet")
    args = ap.parse_args(argv)
    folders = [Path(p) for p in args.runs]
    if args.all:
        folders += [p for p in sorted((ROOT / "results" / "runs").glob("*")) if (p / "ticks.csv.gz").exists()
                    and not (p / "metrics.json").exists()]
    for p in folders:
        evaluate(p)


if __name__ == "__main__":
    main()
