# -*- coding: utf-8 -*-
"""Curvature and radius of a curve defined by several ArUco / AprilTag markers.

The markers sit along a cable (or any flexible object). In every frame the
detected markers are sorted by ID (lowest to highest) and a smooth curve is
fitted through them. The program then reports, for each marker and along the
whole curve:
  - curvature k  (1 / unit): how sharply the curve bends;
  - radius   R   (unit)    : 1 / |k|, the radius of the circle that touches the
                             curve at that point (large R = almost straight).

Dashboard (follows the window size):
- the video, with the curve drawn on it and colored by |curvature|, every
  marker outlined in its own color;
- on the right, the curvature profile along the curve for the current frame,
  with the value at each marker (and a 3-point cross-check);
- the average curvature and the average radius (over the markers detected in
  each frame) along the whole video;
- one row per marker with its detection strip and live k and R.

How the numbers are computed
- Marker position = center of the marker in the image. With --marker-size-mm,
  each marker also gives a local scale (mm per pixel) from its apparent size,
  so k and R come out in 1/mm and mm without any camera calibration. This
  assumes the markers roughly face the camera (a marker tilted by 25 degrees
  reads about 10% small). Without --marker-size-mm everything is in pixels.
- The curve is x(t), y(t): polynomials of degree --degree fitted by least
  squares to the detected marker centers, with t following the distance
  between consecutive markers. With few markers the fit is exact (3 markers
  -> degree 2); with more markers it smooths the noise. A higher --degree
  follows the markers more closely but amplifies detection noise.
- k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2). Sign: positive = the curve turns
  clockwise on screen when followed from the lowest to the highest ID.
- As a cross-check, k3 / R3 use only each marker and its two neighbors (the
  circle through three points); it needs no fit but is noisier.
- At least 3 detected markers are needed for curvature. If a marker is not
  detected in a frame, the curve is built from the others.

Examples:
  python fiducial_curvature.py cable.mp4 --dictionary DICT_APRILTAG_36h11
  python fiducial_curvature.py cable.mp4 --dictionary DICT_4X4_250 --marker-size-mm 30 --marker-ids 0 1 2 3 4 5
  python fiducial_curvature.py cable.mp4 --marker-size-mm 30 --degree 4 --smooth 5 --csv curvature.csv
  python fiducial_curvature.py cable.mp4 --marker-size-mm 30 --save dashboard.mp4 --width 1920 --height 1080

Controls:
  SPACE         pause / resume
  A / left      back 1 frame
  D / right     forward 1 frame
  J / L         back / forward about 1 second
  Home / End    start / end
  S             export the full dashboard (video playing + graphs) to an
                MP4 file next to the video, at the current window size
  Q / ESC       quit
"""
import argparse
import copy
import csv
import math
import os
import time
import warnings
from collections import Counter

import cv2
import numpy as np
from numpy.polynomial import Polynomial

DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

KEYS_LEFT = {2424832, 65361}
KEYS_RIGHT = {2555904, 65363}
KEYS_HOME = {2359296, 65360}
KEYS_END = {2293760, 65367}
WINDOW = "Fiducial Curvature"
TRACKBAR = "Frame"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_WINDOW = (640, 420)
MAX_MARKERS_LIMIT = 12
CURVE_SAMPLES = 160

# BGR
BG = (21, 24, 29)
PANEL = (31, 36, 44)
PANEL_2 = (38, 44, 53)
BORDER = (58, 66, 79)
TEXT = (241, 243, 246)
MUTED = (157, 166, 180)
GREEN = (111, 211, 108)
AMBER = (78, 191, 245)
RED = (96, 91, 235)
WHITE = (255, 255, 255)
CURVE_LINE = (215, 120, 200)

PALETTE = [
    (239, 164, 72), (78, 191, 245), (111, 211, 108), (215, 120, 200),
    (80, 100, 240), (220, 210, 90), (60, 140, 255), (200, 170, 255),
    (120, 230, 200), (230, 130, 120), (90, 200, 255), (180, 180, 180),
]


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def is_apriltag(name):
    return name.startswith("DICT_APRILTAG")


def parse_args():
    p = argparse.ArgumentParser(
        description="Curve through several ArUco/AprilTag markers (sorted by "
                    "ID) with the curvature and radius at each point."
    )
    p.add_argument("video", help="Path to a video file.")
    p.add_argument("--dictionary", choices=sorted(DICTIONARIES),
                   default="DICT_4X4_250",
                   help="Marker dictionary (ArUco or AprilTag). "
                        "Default: DICT_4X4_250.")
    p.add_argument("--marker-ids", type=int, nargs="+", default=None,
                   help="IDs that form the curve. Default: every ID found "
                        "(see --min-detections and --max-markers).")
    p.add_argument("--max-markers", type=int, default=12,
                   help=f"Maximum number of markers used (the most detected "
                        f"ones are kept; limit {MAX_MARKERS_LIMIT}). "
                        f"Default: {MAX_MARKERS_LIMIT}.")
    p.add_argument("--min-detections", type=int, default=3,
                   help="Ignore IDs detected in fewer than N frames "
                        "(filters out false positives). Default: 3.")
    p.add_argument("--marker-size-mm", type=float, default=None,
                   help="Real outer side of the markers in mm. Curvature and "
                        "radius are then in 1/mm and mm instead of 1/px, px.")
    p.add_argument("--degree", type=int, default=3,
                   help="Degree of the polynomial curve (2..6). It is lowered "
                        "automatically when fewer markers are visible. "
                        "Default: 3.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Median filter over N frames (odd) applied to every "
                        "marker position before fitting. Default: 1 (off).")
    p.add_argument("--max-radius", type=float, default=None,
                   help="Largest radius drawn in the radius graph and shown "
                        "as a number (larger = 'straight'). Default: 5x the "
                        "median radius.")
    p.add_argument("--detect-scale", type=float, default=None,
                   help="Scale applied only for detection. Default: 0.5 for "
                        "AprilTag and 1.0 for ArUco.")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Detect every N frames. Default: 1.")
    p.add_argument("--csv", metavar="OUTPUT", default=None,
                   help="Also write per-frame, per-marker results to this CSV.")
    p.add_argument("--width", type=int, default=1500,
                   help="Initial window width, also the export width with "
                        "--save. Default: 1500.")
    p.add_argument("--height", type=int, default=900,
                   help="Initial window height, also the export height with "
                        "--save. Default: 900.")
    p.add_argument("--save", metavar="OUTPUT", default=None,
                   help="Render the full dashboard (video playing with the "
                        "graphs) to this file (.mp4 or .avi) and exit, "
                        "without opening the window.")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        p.error(f"Video not found: {args.video}")
    if args.marker_size_mm is not None and args.marker_size_mm <= 0:
        p.error("--marker-size-mm must be greater than zero.")
    if not 2 <= args.degree <= 6:
        p.error("--degree must be between 2 and 6.")
    if args.smooth < 1 or args.smooth % 2 == 0:
        p.error("--smooth must be an odd number (1 = off).")
    if args.max_radius is not None and args.max_radius <= 0:
        p.error("--max-radius must be greater than zero.")
    if not 3 <= args.max_markers <= MAX_MARKERS_LIMIT:
        p.error(f"--max-markers must be between 3 and {MAX_MARKERS_LIMIT}.")
    if args.min_detections < 1:
        p.error("--min-detections must be at least 1.")
    if args.marker_ids is not None:
        unique = len(set(args.marker_ids))
        if unique < 3:
            p.error("--marker-ids needs at least 3 different IDs.")
        if unique > MAX_MARKERS_LIMIT:
            p.error(f"At most {MAX_MARKERS_LIMIT} markers can be used.")
    if args.detect_scale is None:
        args.detect_scale = 0.5 if is_apriltag(args.dictionary) else 1.0
    if not 0.1 <= args.detect_scale <= 1.0:
        p.error("--detect-scale must be between 0.1 and 1.0.")
    if args.frame_step < 1:
        p.error("--frame-step must be at least 1.")
    if args.width < 900 or args.height < 650:
        p.error("Use at least --width 900 --height 650.")
    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext not in (".mp4", ".avi"):
            p.error("--save must end in .mp4 or .avi.")
    return args


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def create_detector(name):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_APRILTAG if is_apriltag(name)
        else cv2.aruco.CORNER_REFINE_SUBPIX
    )
    return cv2.aruco.ArucoDetector(dictionary, prm)


def detect_scaled(detector, frame, scale):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 0.999:
        image = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        image = gray
    corners, ids, rejected = detector.detectMarkers(image)
    if scale < 0.999 and corners:
        corners = [np.asarray(c, np.float32) / scale for c in corners]
    return corners, ids, rejected


def eta_text(seconds):
    if not np.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------
# Curve math
# --------------------------------------------------------------------------
def fit_curve(points, scales, degree):
    """Polynomial curve through markers ordered by ID (points: (m, 2) px).

    Returns None when fewer than 2 markers are given or all of them coincide.
    The parameter t in [0, 1] follows the distance between consecutive markers.
    """
    m = len(points)
    if m < 2:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    chord = np.r_[0.0, np.cumsum(steps)]
    if chord[-1] < 1e-9:
        return None
    t = chord / chord[-1]
    deg = int(min(degree, m - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = Polynomial.fit(t, points[:, 0], deg, domain=[0.0, 1.0])
        py = Polynomial.fit(t, points[:, 1], deg, domain=[0.0, 1.0])
    return {"t": t, "scales": np.asarray(scales, float), "px": px, "py": py,
            "d1x": px.deriv(1), "d1y": py.deriv(1),
            "d2x": px.deriv(2), "d2y": py.deriv(2), "degree": deg,
            "count": m}


def curvature_px(fit, t):
    """Signed curvature (1/px) of the fitted curve at parameters t."""
    x1, y1 = fit["d1x"](t), fit["d1y"](t)
    x2, y2 = fit["d2x"](t), fit["d2y"](t)
    speed = np.maximum(np.hypot(x1, y1), 1e-9)
    return (x1 * y2 - y1 * x2) / speed ** 3


def menger_curvature_px(points):
    """3-point curvature (1/px) at the interior markers; NaN at the ends."""
    m = len(points)
    out = np.full(m, np.nan)
    for i in range(1, m - 1):
        p1, p2, p3 = points[i - 1], points[i], points[i + 1]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if a * b * c > 1e-12:
            out[i] = 2.0 * cross / (a * b * c)
    return out


def curve_samples(fit, count=CURVE_SAMPLES):
    """Dense samples of the curve: pixel points, k in unit^-1 and the arc
    length in unit (marker-size units when available, otherwise px)."""
    t = np.linspace(0.0, 1.0, count)
    xs, ys = fit["px"](t), fit["py"](t)
    scale = np.interp(t, fit["t"], fit["scales"])
    kappa = curvature_px(fit, t) / scale
    step = np.hypot(np.diff(xs), np.diff(ys)) * 0.5 * (scale[1:] + scale[:-1])
    arc = np.r_[0.0, np.cumsum(step)]
    return {"t": t, "x": xs, "y": ys, "kappa": kappa, "s": arc}


def frame_curve(data, index):
    """Everything drawn for one frame, or None if fewer than 2 markers."""
    rows = np.flatnonzero(data["found"][:, index])
    if rows.size < 2:
        return None
    pts = np.c_[data["x"][rows, index], data["y"][rows, index]]
    fit = fit_curve(pts, data["scale"][rows, index], data["args_degree"])
    if fit is None:
        return None
    sample = curve_samples(fit)
    marker_k = curvature_px(fit, fit["t"]) / fit["scales"]
    marker_s = np.interp(fit["t"], sample["t"], sample["s"])
    return {"rows": rows, "fit": fit, "sample": sample,
            "marker_k": marker_k, "marker_s": marker_s,
            "menger": menger_curvature_px(pts) / fit["scales"],
            "enough": rows.size >= 3}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def padded_range(low, high, include_zero=True, pad=0.12):
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    if span < 1e-12:
        return low - 0.5, high + 0.5
    return low - pad * span, high + pad * span


def moving_median(values, window):
    """Median over `window` frames, ignoring NaN; keeps NaN where input is NaN."""
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, half, constant_values=np.nan)
    stacked = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.nanmedian(stacked, axis=1)
    out[~np.isfinite(values)] = np.nan
    return out


def analyze_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    estimated = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = create_detector(args.dictionary)
    use_mm = args.marker_size_mm is not None
    wanted = set(args.marker_ids) if args.marker_ids else None
    width = height = 0
    records = []  # one dict per frame: id -> (cx, cy, quad, mm_per_px)
    started = time.perf_counter()
    next_percent = 0
    index = 0

    print(f"[ANALYSIS] {os.path.basename(args.video)}")
    print(f"Dictionary: {args.dictionary} | detect-scale: "
          f"{args.detect_scale:.2f}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if width == 0:
                height, width = frame.shape[:2]

            found_now = {}
            if index % args.frame_step == 0:
                corners, ids, _ = detect_scaled(detector, frame,
                                                args.detect_scale)
                if ids is not None and len(ids):
                    for marker_id, marker_corners in zip(ids.reshape(-1), corners):
                        marker_id = int(marker_id)
                        if marker_id in found_now:
                            continue
                        if wanted is not None and marker_id not in wanted:
                            continue
                        points = np.asarray(marker_corners,
                                            np.float32).reshape(4, 2)
                        center = points.mean(axis=0)
                        side_px = float(np.mean(np.linalg.norm(
                            points - np.roll(points, -1, axis=0), axis=1)))
                        mm_per_px = (args.marker_size_mm / side_px
                                     if use_mm and side_px > 1e-6 else 1.0)
                        found_now[marker_id] = (float(center[0]), float(center[1]),
                                                points, mm_per_px)
            records.append(found_now)
            index += 1

            if estimated > 0:
                pct = min(100, int(index / estimated * 100))
                if pct >= next_percent:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    speed = index / elapsed
                    eta = ((estimated - index) / speed
                           if speed > 0 else float("inf"))
                    print(f"  {pct:3d}% | {index}/{estimated} | "
                          f"{speed:6.1f} frames/s | ETA {eta_text(eta)}",
                          flush=True)
                    next_percent = pct + 1
    finally:
        cap.release()

    count = len(records)
    if count == 0:
        raise RuntimeError("The video delivered no frames.")

    # Which IDs form the curve
    seen = Counter(marker_id for record in records for marker_id in record)
    if args.marker_ids:
        ids = []
        for marker_id in dict.fromkeys(args.marker_ids):
            if seen.get(marker_id, 0) == 0:
                print(f"[WARNING] ID {marker_id} was never detected.")
            ids.append(marker_id)
        ids = sorted(ids)
    else:
        candidates = [(m, c) for m, c in seen.items() if c >= args.min_detections]
        dropped = len(seen) - len(candidates)
        candidates.sort(key=lambda item: (-item[1], item[0]))
        kept = candidates[:args.max_markers]
        if len(candidates) > len(kept):
            skipped = ", ".join(str(m) for m, _ in candidates[args.max_markers:])
            print(f"[WARNING] Using the {len(kept)} most detected markers; "
                  f"not used: {skipped} (see --marker-ids / --max-markers).")
        if dropped:
            print(f"[INFO] Ignored {dropped} ID(s) detected in fewer than "
                  f"{args.min_detections} frames.")
        ids = sorted(m for m, _ in kept)
    if len(ids) < 3:
        print(f"[WARNING] Only {len(ids)} marker(s) available: curvature needs "
              "at least 3 detected markers in the same frame.")

    total = len(ids)
    row_of = {marker_id: row for row, marker_id in enumerate(ids)}
    x = np.full((total, count), np.nan)
    y = np.full((total, count), np.nan)
    scale = np.full((total, count), np.nan)
    quads = np.full((total, count, 4, 2), np.nan, np.float32)
    for frame_index, record in enumerate(records):
        for marker_id, (cx, cy, points, mm_per_px) in record.items():
            row = row_of.get(marker_id)
            if row is None:
                continue
            x[row, frame_index], y[row, frame_index] = cx, cy
            scale[row, frame_index] = mm_per_px
            quads[row, frame_index] = points
    del records

    found = np.isfinite(x)
    if args.smooth > 1:
        for row in range(total):
            x[row] = moving_median(x[row], args.smooth)
            y[row] = moving_median(y[row], args.smooth)
            scale[row] = moving_median(scale[row], args.smooth)

    data = {
        "fps": fps, "n": count, "width": width, "height": height,
        "ids": ids, "colors": [PALETTE[i % len(PALETTE)] for i in range(total)],
        "x": x, "y": y, "scale": scale, "found": found, "quads": quads,
        "unit": "mm" if use_mm else "px", "args_degree": args.degree,
        "strip_cache": {},
    }

    # Curvature of every marker in every frame
    kappa = np.full((total, count), np.nan)
    kappa3 = np.full((total, count), np.nan)
    for frame_index in range(count):
        rows = np.flatnonzero(found[:, frame_index])
        if rows.size < 3:
            continue
        pts = np.c_[x[rows, frame_index], y[rows, frame_index]]
        fit = fit_curve(pts, scale[rows, frame_index], args.degree)
        if fit is None:
            continue
        kappa[rows, frame_index] = curvature_px(fit, fit["t"]) / fit["scales"]
        kappa3[rows, frame_index] = (menger_curvature_px(pts) / fit["scales"])
    with np.errstate(divide="ignore"):
        radius = 1.0 / np.abs(kappa)
        radius3 = 1.0 / np.abs(kappa3)
    radius[~np.isfinite(radius)] = np.nan
    radius3[~np.isfinite(radius3)] = np.nan

    # Average over the markers detected in each frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa_mean = np.nanmean(kappa, axis=0)
        radius_mean = np.nanmean(radius, axis=0)

    finite_k = kappa[np.isfinite(kappa)]
    if finite_k.size:
        lo, hi = np.percentile(finite_k, [1, 99])
        data["range_k"] = padded_range(float(lo), float(hi))
        data["vmax"] = max(float(np.percentile(np.abs(finite_k), 95)), 1e-12)
    else:
        data["range_k"] = (-1.0, 1.0)
        data["vmax"] = 1.0
    finite_r = radius[np.isfinite(radius)]
    if args.max_radius is not None:
        cap = float(args.max_radius)
    elif finite_r.size:
        cap = 5.0 * float(np.median(finite_r))
    else:
        cap = 1.0
    finite_km = kappa_mean[np.isfinite(kappa_mean)]
    if finite_km.size:
        lo, hi = np.percentile(finite_km, [1, 99])
        range_k_mean = padded_range(float(lo), float(hi))
    else:
        range_k_mean = (-1.0, 1.0)
    finite_rm = radius_mean[np.isfinite(radius_mean)]
    if args.max_radius is not None:
        top_r = float(args.max_radius)
    elif finite_rm.size:
        top_r = 1.1 * float(np.percentile(finite_rm, 99))
    else:
        top_r = 1.0
    data.update(kappa=kappa, kappa3=kappa3, radius=radius, radius3=radius3,
                radius_cap=cap, kappa_mean=kappa_mean, radius_mean=radius_mean,
                range_k_mean=range_k_mean, range_r_mean=(0.0, max(top_r, 1e-9)))

    if args.csv:
        write_csv(args.csv, data)
    return data


def write_csv(path, data):
    unit = data["unit"]
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "marker_id", "x_px", "y_px",
                         f"curvature_1_per_{unit}", f"radius_{unit}",
                         f"curvature_3pt_1_per_{unit}", f"radius_3pt_{unit}",
                         f"frame_mean_curvature_1_per_{unit}",
                         f"frame_mean_radius_{unit}"])
        for index in range(data["n"]):
            for row, marker_id in enumerate(data["ids"]):
                if not data["found"][row, index]:
                    continue
                values = [data["kappa"][row, index], data["radius"][row, index],
                          data["kappa3"][row, index], data["radius3"][row, index],
                          data["kappa_mean"][index], data["radius_mean"][index]]
                writer.writerow([index + 1, f"{index / data['fps']:.4f}", marker_id,
                                 f"{data['x'][row, index]:.2f}",
                                 f"{data['y'][row, index]:.2f}"] +
                                ["" if not np.isfinite(v) else f"{v:.6g}"
                                 for v in values])
    print(f"[OK] CSV written: {path}")


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def put(image, text, point, scale, color=TEXT, thickness=1):
    cv2.putText(image, str(text), point, FONT, scale, color,
                thickness, cv2.LINE_AA)


def put_outlined(image, text, point, scale, color):
    cv2.putText(image, str(text), point, FONT, scale, (10, 12, 15), 3, cv2.LINE_AA)
    cv2.putText(image, str(text), point, FONT, scale, color, 1, cv2.LINE_AA)


def text_width(text, scale, thickness=1):
    return cv2.getTextSize(str(text), FONT, scale, thickness)[0][0]


def fit_scale(text, max_width, scale):
    while scale > 0.25 and text_width(text, scale) > max_width:
        scale -= 0.02
    return scale


def panel(image, rect, color=PANEL):
    x0, y0, x1, y1 = rect
    cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), BORDER, 1)


def draw_runs(canvas, xs, ys, valid, color, thickness):
    """Polylines that break wherever the marker was not detected."""
    ids = np.flatnonzero(valid)
    if not ids.size:
        return
    runs = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    for run in runs:
        points = np.c_[xs[run], ys[run]].astype(np.int32)
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), max(1, thickness), color,
                       -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False,
                          color, thickness, cv2.LINE_AA)


def format_number(value, digits=3):
    if not np.isfinite(value):
        return "--"
    if value != 0 and (abs(value) < 10 ** -(digits - 1) or abs(value) >= 1e5):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


# --------------------------------------------------------------------------
# Dashboard parts
# --------------------------------------------------------------------------
def draw_header(canvas, args, data, rect, index, paused, k):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), BG, -1)
    family = "APRILTAG" if is_apriltag(args.dictionary) else "ARUCO"
    put(canvas, os.path.basename(args.video),
        (x0 + int(18 * k), y0 + int(27 * k)), 0.62 * k, TEXT, 1)

    total = len(data["ids"])
    now = int(data["found"][:, index].sum()) if total else 0
    id_list = " -> ".join(str(m) for m in data["ids"]) or "none"
    info = (f"{family} | {args.dictionary} | curve through IDs {id_list} | "
            f"polynomial degree {args.degree}")
    if now >= 3 and now == total:
        badge, color = f"CURVE {now}/{total} MARKERS", GREEN
    elif now >= 3:
        badge, color = f"CURVE {now}/{total} MARKERS", AMBER
    else:
        badge, color = f"NEED 3+ MARKERS ({now}/{total})", RED
    badge_w = text_width(badge, 0.45 * k) + int(30 * k)
    info_scale = fit_scale(info, (x1 - x0) - badge_w - int(60 * k), 0.42 * k)
    put(canvas, info, (x0 + int(18 * k), y0 + int(51 * k)), info_scale, MUTED, 1)
    cv2.rectangle(canvas, (x1 - badge_w - int(18 * k), y0 + int(13 * k)),
                  (x1 - int(18 * k), y0 + int(48 * k)), color, -1)
    put(canvas, badge, (x1 - badge_w - int(3 * k), y0 + int(37 * k)),
        0.45 * k, (20, 23, 28), 1)

    state = "PAUSED" if paused else "PLAYING"
    right = (f"{state}  |  {index / data['fps']:.2f} s  |  "
             f"frame {index + 1}/{data['n']}")
    put(canvas, right, (x1 - text_width(right, 0.42 * k) - int(18 * k),
                        y1 - int(8 * k)), 0.42 * k, MUTED, 1)


def curvature_colors(kappa, vmax):
    """BGR colors (m, 3) for |curvature| between 0 and vmax."""
    level = np.clip(np.abs(kappa) / vmax, 0.0, 1.0)
    indices = (level * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(indices, cv2.COLORMAP_TURBO).reshape(-1, 3)


def draw_video_panel(canvas, frame, data, index, rect, k, curve):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (7, 9, 12), -1)
    if frame is None:
        put(canvas, "No frame", (x0 + int(20 * k), y0 + int(35 * k)), 0.6 * k, MUTED)
        return
    height, width = frame.shape[:2]
    area_w, area_h = x1 - x0, y1 - y0
    scale = min(area_w / width, area_h / height)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    shown = cv2.resize(frame, (new_w, new_h),
                       interpolation=(cv2.INTER_AREA if scale < 1
                                      else cv2.INTER_LINEAR))

    # The curve, colored by |curvature| (a dark outline keeps it readable)
    if curve is not None:
        sample = curve["sample"]
        points = np.c_[sample["x"], sample["y"]] * scale
        colors = (curvature_colors(sample["kappa"], data["vmax"])
                  if curve["enough"] else None)
        thick = max(2, int(round(3 * k)))
        cv2.polylines(shown, [points.astype(np.int32).reshape(-1, 1, 2)], False,
                      (10, 12, 15), thick + 3, cv2.LINE_AA)
        for i in range(len(points) - 1):
            color = (tuple(int(c) for c in colors[i]) if colors is not None
                     else CURVE_LINE)
            cv2.line(shown, tuple(points[i].astype(int)),
                     tuple(points[i + 1].astype(int)), color, thick, cv2.LINE_AA)

    # Marker outlines and IDs
    for row, marker_id in enumerate(data["ids"]):
        quad = data["quads"][row, index]
        if not np.isfinite(quad).all():
            continue
        color = data["colors"][row]
        pts = (quad * scale).astype(np.int32)
        cv2.polylines(shown, [pts.reshape(-1, 1, 2)], True, color,
                      max(1, int(2 * k)), cv2.LINE_AA)
        center = pts.mean(axis=0).astype(int)
        cv2.circle(shown, tuple(center), max(3, int(4 * k)), WHITE, -1, cv2.LINE_AA)
        cv2.circle(shown, tuple(center), max(2, int(3 * k)), color, -1, cv2.LINE_AA)
        anchor = pts[np.argmin(pts[:, 1])]
        put_outlined(shown, f"ID {marker_id}",
                     (int(anchor[0]), max(int(14 * k), int(anchor[1]) - int(8 * k))),
                     0.55 * k, color)

    ox = x0 + (area_w - new_w) // 2
    oy = y0 + (area_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = shown
    cv2.rectangle(canvas, (x0, y0), (x1, y1), BORDER, 1)

    # Color legend for |curvature|
    bar_w, bar_h = int(150 * k), int(10 * k)
    bx, by = x0 + int(12 * k), y0 + int(14 * k)
    gradient = curvature_colors(np.linspace(0, data["vmax"], bar_w), data["vmax"])
    canvas[by:by + bar_h, bx:bx + bar_w] = np.repeat(gradient[None, :, :], bar_h, axis=0)
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), BORDER, 1)
    put_outlined(canvas, f"|k|: 0 .. {format_number(data['vmax'])} 1/{data['unit']}",
                 (bx, by + bar_h + int(15 * k)), 0.42 * k, TEXT)


def draw_profile(canvas, data, rect, index, k, curve):
    """Curvature k(s) along the curve for the current frame."""
    x0, y0, x1, y1 = rect
    panel(canvas, rect)
    unit = data["unit"]
    title = "CURVATURE ALONG THE CURVE"
    put(canvas, title, (x0 + int(14 * k), y0 + int(24 * k)),
        fit_scale(title, (x1 - x0) - int(24 * k), 0.48 * k), MUTED)

    left, right = x0 + int(58 * k), x1 - int(16 * k)
    top, bottom = y0 + int(40 * k), y1 - int(72 * k)
    if right - left < 40 or bottom - top < 40:
        return
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_2, -1)

    def message(text):
        put(canvas, text, (left + int(12 * k), (top + bottom) // 2),
            fit_scale(text, right - left - int(20 * k), 0.46 * k), MUTED)

    if curve is None:
        message("need at least 2 detected markers")
        return
    if not curve["enough"]:
        message("need 3 markers for curvature")
        return

    sample = curve["sample"]
    k_lo, k_hi = data["range_k"]
    length = max(float(sample["s"][-1]), 1e-9)

    def to_x(s):
        return left + np.asarray(s) / length * (right - left)

    def to_y(values):
        return bottom - (np.clip(values, k_lo, k_hi) - k_lo) / (k_hi - k_lo) * (bottom - top)

    for value in np.linspace(k_lo, k_hi, 5):
        yy = int(to_y(value))
        cv2.line(canvas, (left, yy), (right, yy), BORDER, 1)
        label = format_number(value, 2)
        put(canvas, label, (left - text_width(label, 0.34 * k) - int(5 * k),
                            yy + int(4 * k)), 0.34 * k, MUTED)
    zero = int(to_y(0.0))
    cv2.line(canvas, (left, zero), (right, zero), MUTED, 1)

    # Marker positions along the curve
    for slot, row in enumerate(curve["rows"]):
        xx = int(to_x(curve["marker_s"][slot]))
        cv2.line(canvas, (xx, top), (xx, bottom), BORDER, 1)

    xs = to_x(sample["s"])
    ys = to_y(sample["kappa"])
    pts = np.c_[xs, ys].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, CURVE_LINE, max(2, int(round(2 * k))),
                  cv2.LINE_AA)

    for slot, row in enumerate(curve["rows"]):
        color = data["colors"][row]
        xx = int(to_x(curve["marker_s"][slot]))
        cv2.circle(canvas, (xx, int(to_y(curve["marker_k"][slot]))),
                   max(4, int(6 * k)), color, -1, cv2.LINE_AA)       # fit
        if np.isfinite(curve["menger"][slot]):
            cv2.circle(canvas, (xx, int(to_y(curve["menger"][slot]))),
                       max(4, int(6 * k)), color, max(1, int(k)), cv2.LINE_AA)  # 3-point
        put_outlined(canvas, str(data["ids"][row]),
                     (xx - int(4 * k), top + int(12 * k)), 0.4 * k, color)

    put(canvas, f"s ({unit})  0", (left, bottom + int(15 * k)), 0.34 * k, MUTED)
    end_label = f"{length:.1f}"
    put(canvas, end_label, (right - text_width(end_label, 0.34 * k),
                            bottom + int(15 * k)), 0.34 * k, MUTED)
    put(canvas, "filled dot = fit | hollow = 3-point",
        (left, bottom + int(30 * k)), 0.34 * k, MUTED)

    # Length and radius statistics. The outer 5% at each end is left out: a
    # polynomial fit is the least reliable at its extremes.
    inner = (sample["t"] >= 0.05) & (sample["t"] <= 0.95)
    abs_k = np.abs(sample["kappa"][inner])
    k_peak, k_mean = float(abs_k.max()), float(abs_k.mean())
    min_r = f"{1.0 / k_peak:.1f}" if k_peak > 1e-12 else "inf"
    mean_r = f"{1.0 / k_mean:.1f}" if k_mean > 1e-12 else "inf"
    line1 = f"length {length:.1f} {unit}   markers {curve['fit']['count']}/{len(data['ids'])}"
    line2 = f"inner 90%: R min {min_r}, mean {mean_r} {unit}"
    put(canvas, line1, (x0 + int(14 * k), y1 - int(28 * k)),
        fit_scale(line1, (x1 - x0) - int(24 * k), 0.44 * k), TEXT)
    put(canvas, line2, (x0 + int(14 * k), y1 - int(9 * k)),
        fit_scale(line2, (x1 - x0) - int(24 * k), 0.42 * k), MUTED)


def draw_series_plot(canvas, data, rect, index, k, title, series, y_range,
                     include_zero, note=None):
    """Time graph: `series` is a list of (values[n], color). Values outside
    the range are clipped to the plot edges."""
    x0, y0, x1, y1 = rect
    panel(canvas, rect)
    ymin, ymax = y_range
    fps, count = data["fps"], data["n"]
    duration = max((count - 1) / fps, 1e-6)
    put(canvas, title, (x0 + int(16 * k), y0 + int(24 * k)), 0.48 * k, MUTED)
    if note:
        put(canvas, note, (x0 + int(16 * k) + text_width(title, 0.48 * k) + int(24 * k),
                           y0 + int(24 * k)), 0.48 * k, TEXT)

    left, right = x0 + int(66 * k), x1 - int(14 * k)
    top, bottom = y0 + int(36 * k), y1 - int(26 * k)
    if right - left < 40 or bottom - top < 30:
        return

    def to_x(seconds):
        return left + np.asarray(seconds) / duration * (right - left)

    def to_y(vals):
        return bottom - (np.clip(vals, ymin, ymax) - ymin) / (ymax - ymin) * (bottom - top)

    for value in np.linspace(ymin, ymax, 5):
        yy = int(to_y(value))
        cv2.line(canvas, (left, yy), (right, yy), BORDER, 1)
        put(canvas, format_number(value, 2), (x0 + int(6 * k), yy + int(4 * k)),
            0.34 * k, MUTED)
    for seconds in np.linspace(0, duration, 6):
        xx = int(to_x(seconds))
        cv2.line(canvas, (xx, top), (xx, bottom), BORDER, 1)
        label = f"{seconds:.1f}s"
        put(canvas, label, (xx - text_width(label, 0.33 * k) // 2,
                            y1 - int(8 * k)), 0.33 * k, MUTED)
    if include_zero and ymin < 0.0 < ymax:
        zero = int(to_y(0.0))
        cv2.line(canvas, (left, zero), (right, zero), MUTED, 1)

    times = np.arange(count) / fps
    stride = max(1, count // max(1, 2 * (right - left)))
    sampled_times = times[::stride]
    thickness = max(1, int(round(2 * k)))
    for values, color in series:
        sampled = values[::stride]
        valid = np.isfinite(sampled)
        draw_runs(canvas, to_x(sampled_times),
                  to_y(np.where(valid, sampled, 0.0)), valid, color, thickness)

    cursor_x = int(to_x(index / fps))
    cv2.line(canvas, (cursor_x, top), (cursor_x, bottom), WHITE,
             max(1, int(round(2 * k))), cv2.LINE_AA)


def strip_fraction(data, row, width):
    key = (row, width)
    cached = data["strip_cache"].get(key)
    if cached is not None:
        return cached
    values = data["found"][row].astype(np.float32)
    count = len(values)
    if count <= width:
        columns = values[(np.arange(width) * count) // max(1, width)]
    else:
        edges = np.linspace(0, count, width + 1).astype(int)
        columns = np.add.reduceat(values, edges[:-1]) / np.diff(edges)
    data["strip_cache"][key] = columns
    return columns


def draw_detection_rows(canvas, data, rect, index, k, row_h, row_gap):
    """One detection strip per marker with its live curvature and radius."""
    x0, y0, x1, y1 = rect
    total = len(data["ids"])
    label_w = text_width("ID 000", 0.4 * k) + int(16 * k)
    values_w = int(min(260 * k, (x1 - x0) * 0.30))
    sx0, sx1 = x0 + label_w, x1 - values_w - int(8 * k)
    if total == 0 or sx1 - sx0 < 40:
        return
    width = sx1 - sx0
    unit = data["unit"]
    base = np.array(BORDER, float)
    cap = data["radius_cap"]

    for row, marker_id in enumerate(data["ids"]):
        ry0 = y0 + row * (row_h + row_gap)
        ry1 = ry0 + row_h
        color = data["colors"][row]
        fraction = strip_fraction(data, row, width)
        strip = (base[None, :] * (1 - fraction[:, None]) +
                 np.array(color, float)[None, :] * fraction[:, None])
        canvas[ry0:ry1, sx0:sx1] = np.repeat(strip[None, :, :], row_h,
                                             axis=0).astype(np.uint8)
        baseline = ry1 - max(1, row_h // 5)
        put(canvas, f"ID {marker_id}", (x0, baseline), 0.4 * k, color, 1)

        kappa = data["kappa"][row, index]
        if np.isfinite(kappa):
            radius = data["radius"][row, index]
            radius_text = (f"R >{cap:.0f}" if not np.isfinite(radius) or radius > cap
                           else f"R {radius:.1f}")
            text = f"k {kappa:+.4f}  {radius_text}"
            text_color = TEXT
        else:
            text, text_color = "--", MUTED
        put(canvas, text, (sx1 + int(8 * k), baseline),
            fit_scale(text, values_w, 0.38 * k), text_color)

    cursor = sx0 + int(index / max(1, data["n"] - 1) * (width - 1))
    block_h = total * (row_h + row_gap) - row_gap
    cv2.line(canvas, (cursor, y0 - 3), (cursor, y0 + block_h + 3), WHITE,
             max(1, int(round(2 * k))), cv2.LINE_AA)
    put(canvas, f"Detection per marker  (k = curvature in 1/{unit}, R = radius in "
                f"{unit}; positive k = clockwise on screen)",
        (x0, y0 - int(8 * k)), 0.38 * k, MUTED)


def draw_toast(canvas, text, rect, k):
    x0, y0, x1, y1 = rect
    scale = fit_scale(text, (x1 - x0) - int(40 * k), 0.5 * k)
    width = text_width(text, scale) + int(24 * k)
    height = int(30 * k)
    bx0, by1 = x0 + int(12 * k), y1 - int(12 * k)
    cv2.rectangle(canvas, (bx0, by1 - height), (bx0 + width, by1), (12, 14, 18), -1)
    cv2.rectangle(canvas, (bx0, by1 - height), (bx0 + width, by1), BORDER, 1)
    put(canvas, text, (bx0 + int(12 * k), by1 - int(10 * k)), scale, TEXT)


# --------------------------------------------------------------------------
# Layout / frames
# --------------------------------------------------------------------------
def read_frame(cap, cache, index):
    if cache["index"] == index and cache["frame"] is not None:
        return cache["frame"]
    gap = index - cache["index"]
    if 0 < gap <= 5 and cache["index"] >= 0:
        for _ in range(gap - 1):
            cap.grab()
        ok, frame = cap.read()
    elif index == 0 and cache["index"] == -1:
        ok, frame = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    if ok and frame is not None:
        cache["index"] = index
        cache["frame"] = frame
    return cache["frame"]


def render(canvas, args, data, frame, index, paused, toast=None):
    """Draws the whole dashboard, adapting to the canvas size."""
    canvas[:] = BG
    height, width = canvas.shape[:2]
    k = max(0.6, min(1.8, min(width / 1500.0, height / 900.0)))

    margin = int(14 * k)
    header_h = int(72 * k)
    total = len(data["ids"])
    row_h = max(6, int(9 * k))
    row_gap = max(2, int(3 * k))
    block_h = max(1, total) * (row_h + row_gap) - row_gap
    block_label = int(22 * k)

    block_top = height - margin - block_h
    plots_bottom = block_top - block_label
    plots_h = max(int(130 * k), int(height * 0.24))
    plots_top = plots_bottom - plots_h
    body_top = header_h
    body_bottom = plots_top - margin
    body_h = max(50, body_bottom - body_top)

    profile_w = int(min(max(body_h * 1.05, width * 0.26), width * 0.42))
    profile_x1 = width - margin
    profile_x0 = profile_x1 - profile_w
    video_rect = (margin, body_top, profile_x0 - margin, body_bottom)
    profile_rect = (profile_x0, body_top, profile_x1, body_bottom)

    plot_w = (width - 3 * margin) // 2
    plot_left = (margin, plots_top, margin + plot_w, plots_bottom)
    plot_right = (margin * 2 + plot_w, plots_top, width - margin, plots_bottom)

    curve = frame_curve(data, index)
    unit = data["unit"]
    draw_header(canvas, args, data, (0, 0, width, header_h), index, paused, k)
    draw_video_panel(canvas, frame, data, index, video_rect, k, curve)
    draw_profile(canvas, data, profile_rect, index, k, curve)

    now_k, now_r = data["kappa_mean"][index], data["radius_mean"][index]
    draw_series_plot(canvas, data, plot_left, index, k,
                     f"AVERAGE CURVATURE OVER TIME (1/{unit})",
                     [(data["kappa_mean"], CURVE_LINE)], data["range_k_mean"],
                     True, note=("now " + format_number(now_k, 3)
                                 if np.isfinite(now_k) else "now --"))
    draw_series_plot(canvas, data, plot_right, index, k,
                     f"AVERAGE RADIUS OVER TIME ({unit})",
                     [(data["radius_mean"], CURVE_LINE)], data["range_r_mean"],
                     False, note=("now " + format_number(now_r, 3)
                                  if np.isfinite(now_r) else "now --"))
    draw_detection_rows(canvas, data,
                        (margin, block_top, width - margin, height - margin),
                        index, k, row_h, row_gap)
    if toast:
        draw_toast(canvas, toast, video_rect, k)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def fourcc_for(path):
    ext = os.path.splitext(path)[1].lower()
    return "MJPG" if ext == ".avi" else "mp4v"


def unique_output_path(video_path):
    stem = os.path.splitext(video_path)[0] + "_curvature_dashboard"
    candidate = stem + ".mp4"
    number = 1
    while os.path.exists(candidate):
        candidate = f"{stem}_{number}.mp4"
        number += 1
    return candidate


def export_video(args, data, out_path, width, height):
    """Renders the full dashboard, frame by frame, to a video file."""
    width -= width % 2
    height -= height % 2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not reopen the video for export.")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc_for(out_path)),
                             data["fps"], (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create the output file: {out_path}")

    canvas = np.empty((height, width, 3), np.uint8)
    cache = {"index": -1, "frame": None}
    started = time.perf_counter()
    next_percent = 0
    print(f"[EXPORT] {out_path} | {width}x{height} @ {data['fps']:.2f} fps",
          flush=True)
    try:
        for index in range(data["n"]):
            frame = read_frame(cap, cache, index)
            render(canvas, args, data, frame, index, paused=False)
            writer.write(canvas)
            pct = int((index + 1) / data["n"] * 100)
            if pct >= next_percent:
                elapsed = max(1e-6, time.perf_counter() - started)
                speed = (index + 1) / elapsed
                eta = (data["n"] - index - 1) / speed
                print(f"  {pct:3d}% | {index + 1}/{data['n']} | "
                      f"{speed:6.1f} frames/s | ETA {eta_text(eta)}", flush=True)
                next_percent = pct + 5
    finally:
        writer.release()
        cap.release()
    print(f"[OK] Saved: {out_path}", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def window_size(default):
    """Current drawable size of the window (so the dashboard follows it)."""
    try:
        _x, _y, w, h = cv2.getWindowImageRect(WINDOW)
        if w >= 200 and h >= 150:
            return max(int(w), MIN_WINDOW[0]), max(int(h), MIN_WINDOW[1])
    except cv2.error:
        pass
    return default


def main():
    args = parse_args()
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    data = analyze_video(args)
    total = len(data["ids"])
    print(f"[OK] {data['n']} frames @ {data['fps']:.2f} fps | {total} marker(s) | "
          f"curvature in 1/{data['unit']}")
    for row, marker_id in enumerate(data["ids"]):
        rate = 100.0 * float(data["found"][row].mean())
        valid = data["kappa"][row][np.isfinite(data["kappa"][row])]
        mean_text = (f"mean k {valid.mean():+.5f}" if valid.size else "no curvature")
        print(f"  ID {marker_id:>3}: detected in {rate:5.1f}% of frames | {mean_text}")
    mean_k = data["kappa_mean"][np.isfinite(data["kappa_mean"])]
    mean_r = data["radius_mean"][np.isfinite(data["radius_mean"])]
    if mean_k.size:
        print(f"  Average over markers and time: curvature {mean_k.mean():+.5f} "
              f"1/{data['unit']} | radius {mean_r.mean():.1f} {data['unit']}")
    frames_ok = int((data["found"].sum(axis=0) >= 3).sum())
    print(f"  Frames with 3+ markers (curvature available): {frames_ok}/{data['n']}")
    if total == 0:
        print("[WARNING] No marker found. Check the dictionary and the IDs; "
              "for small markers, try --detect-scale 1.0.")
    if data["unit"] == "px":
        print("[INFO] Values are in pixels. Use --marker-size-mm for mm.")

    if args.save:
        export_video(args, data, args.save, args.width, args.height)
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not reopen the video for display.")

    size = (args.width, args.height)
    canvas = np.full((size[1], size[0], 3), BG, np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, size[0], size[1])
    state = {"index": 0, "seek": None}

    def on_trackbar(value):
        if value != state["index"]:
            state["seek"] = value

    if data["n"] > 1:
        cv2.createTrackbar(TRACKBAR, WINDOW, 0, data["n"] - 1, on_trackbar)

    index = 0
    paused = False
    jump = max(1, int(round(data["fps"])))
    cache = {"index": -1, "frame": None}
    toast = None
    toast_until = 0.0
    print("SPACE pause | A/D +/-1 frame | J/L +/-1 s | Home/End | "
          "S export dashboard video | Q/ESC quit")

    try:
        while True:
            started = time.perf_counter()

            new_size = window_size(size)
            if new_size != size:
                size = new_size
                canvas = np.full((size[1], size[0], 3), BG, np.uint8)

            frame = read_frame(cap, cache, index)
            message = toast if time.time() < toast_until else None
            render(canvas, args, data, frame, index, paused, message)
            cv2.imshow(WINDOW, canvas)
            if data["n"] > 1:
                state["index"] = index
                cv2.setTrackbarPos(TRACKBAR, WINDOW, index)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            delay = 30 if paused else max(1, int(1000.0 / data["fps"] - elapsed_ms))
            key_ex = cv2.waitKeyEx(delay)
            key = key_ex & 0xFF if key_ex != -1 else -1
            if key in (ord("q"), ord("Q"), 27):
                break

            target = None
            if key == ord(" "):
                paused = not paused
            elif key in (ord("d"), ord("D")) or key_ex in KEYS_RIGHT:
                paused = True
                target = index + 1
            elif key in (ord("a"), ord("A")) or key_ex in KEYS_LEFT:
                paused = True
                target = index - 1
            elif key in (ord("l"), ord("L")):
                target = index + jump
            elif key in (ord("j"), ord("J")):
                target = index - jump
            elif key_ex in KEYS_HOME:
                target = 0
            elif key_ex in KEYS_END:
                target = data["n"] - 1
            elif key in (ord("s"), ord("S")):
                out_path = unique_output_path(args.video)
                render(canvas, args, data, frame, index, paused,
                       f"Exporting {os.path.basename(out_path)} ... see console")
                cv2.imshow(WINDOW, canvas)
                cv2.waitKey(1)
                try:
                    export_video(args, data, out_path, size[0], size[1])
                    toast = f"Saved: {os.path.basename(out_path)}"
                except Exception as error:  # keep the player alive
                    print(f"[ERROR] Export failed: {error}")
                    toast = "Export failed (see console)"
                toast_until = time.time() + 5.0

            if state["seek"] is not None:
                target = state["seek"]
                state["seek"] = None
            if target is None and not paused:
                target = index + 1
            if target is None:
                continue

            target = max(0, min(target, data["n"] - 1))
            if target == index:
                if not paused and index >= data["n"] - 1:
                    paused = True
                continue
            index = target
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
