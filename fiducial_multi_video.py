# -*- coding: utf-8 -*-
"""Single-video player that tracks SEVERAL ArUco / AprilTag markers at once.

Layout (follows the window size, resize it freely):
- header with file info and how many markers are detected right now;
- the video on the left (every tracked marker outlined in its own color) and,
  on its right, the XY plane with the accumulated trajectory of each marker;
- dX(t) and dY(t) graphs over the whole video, one colored line per marker;
- one detection strip per marker (also works as the color legend), with the
  current dX / dY and accumulated path length at the right.

By default each marker is measured from the first frame in which it was
detected, so all trajectories start at the origin and are easy to compare.
With --absolute the raw position is plotted instead (image pixels, or mm in
the camera frame when --marker-size-mm is given).

Examples:
  python fiducial_multi_video.py video.mp4 --dictionary DICT_APRILTAG_36h11
  python fiducial_multi_video.py video.mp4 --dictionary DICT_4X4_250 --marker-ids 0 1 2 3
  python fiducial_multi_video.py video.mp4 --dictionary DICT_APRILTAG_36h11 --marker-size-mm 50 --calibration camera_calibration.npz
  python fiducial_multi_video.py video.mp4 --absolute
  python fiducial_multi_video.py video.mp4 --save dashboard.mp4 --width 1920 --height 1080

Controls:
  SPACE         pause / resume
  A / left      back 1 frame
  D / right     forward 1 frame
  J / L         back / forward about 1 second
  Home / End    start / end
  S             export the full dashboard (video playing + graphs) to an
                MP4 file next to the video, at the current window size
  Q / ESC       quit

The graphs and the XY plane use the image convention: +X to the right,
+Y downwards.
"""
import argparse
import math
import os
import time
from collections import Counter

import cv2
import numpy as np

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
WINDOW = "Fiducial Tracking - Multiple Markers"
TRACKBAR = "Frame"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_WINDOW = (640, 420)
MAX_MARKERS_LIMIT = 14

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

# One color per marker (BGR), picked to stay distinguishable on a dark theme
PALETTE = [
    (239, 164, 72),    # blue
    (78, 191, 245),    # amber
    (111, 211, 108),   # green
    (215, 120, 200),   # magenta
    (80, 100, 240),    # red
    (220, 210, 90),    # cyan
    (60, 140, 255),    # orange
    (200, 170, 255),   # pink
    (120, 230, 200),   # lime
    (230, 130, 120),   # violet
    (90, 200, 255),    # yellow
    (180, 180, 180),   # gray
]


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def is_apriltag(name):
    return name.startswith("DICT_APRILTAG")


def parse_args():
    p = argparse.ArgumentParser(
        description="Track several ArUco/AprilTag markers in one video and show "
                    "each trajectory over time and in the XY plane."
    )
    p.add_argument("video", help="Path to a video file.")
    p.add_argument("--dictionary", choices=sorted(DICTIONARIES),
                   default="DICT_4X4_250",
                   help="Marker dictionary (ArUco or AprilTag). "
                        "Default: DICT_4X4_250.")
    p.add_argument("--marker-ids", type=int, nargs="+", default=None,
                   help="IDs to track. Default: every ID found in the video "
                        "(see --min-detections and --max-markers).")
    p.add_argument("--max-markers", type=int, default=8,
                   help=f"Maximum number of markers shown (the ones detected "
                        f"in most frames are kept; limit {MAX_MARKERS_LIMIT}). "
                        f"Default: 8.")
    p.add_argument("--min-detections", type=int, default=3,
                   help="Ignore IDs detected in fewer than N frames "
                        "(filters out false positives). Default: 3.")
    p.add_argument("--absolute", action="store_true",
                   help="Plot raw positions instead of the displacement from "
                        "each marker's first detection.")
    p.add_argument("--marker-size-mm", type=float, default=None,
                   help="Real outer side of the markers in mm. Values are then "
                        "reported in mm instead of pixels.")
    p.add_argument("--calibration", default=None,
                   help="NPZ file with camera_matrix and dist_coeffs.")
    p.add_argument("--detect-scale", type=float, default=None,
                   help="Scale applied only for detection. Default: 0.5 for "
                        "AprilTag and 1.0 for ArUco.")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Detect every N frames. Default: 1.")
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
    if not 1 <= args.max_markers <= MAX_MARKERS_LIMIT:
        p.error(f"--max-markers must be between 1 and {MAX_MARKERS_LIMIT}.")
    if args.min_detections < 1:
        p.error("--min-detections must be at least 1.")
    if args.marker_ids is not None and len(set(args.marker_ids)) > MAX_MARKERS_LIMIT:
        p.error(f"At most {MAX_MARKERS_LIMIT} markers can be tracked.")
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
        cv2.aruco.CORNER_REFINE_NONE if is_apriltag(name)
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


def object_points(size_mm):
    h = size_mm / 2.0
    return np.array([[-h, h, 0], [h, h, 0],
                     [h, -h, 0], [-h, -h, 0]], np.float32)


def load_calibration(path, width, height):
    if path:
        data = np.load(path)
        return (np.asarray(data["camera_matrix"], np.float64),
                np.asarray(data["dist_coeffs"], np.float64), True)
    focal = float(max(width, height))
    matrix = np.array([[focal, 0, width / 2],
                       [0, focal, height / 2],
                       [0, 0, 1]], np.float64)
    return matrix, np.zeros((5, 1), np.float64), False


def eta_text(seconds):
    if not np.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def padded_range(values, include_zero):
    """(low, high) with 8% padding; optionally forced to include zero."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    if span < 1e-9:
        return low - 0.5, high + 0.5
    return low - 0.08 * span, high + 0.08 * span


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
    obj = object_points(args.marker_size_mm) if use_mm else None
    wanted = set(args.marker_ids) if args.marker_ids else None
    camera_matrix = dist_coeffs = None
    calibrated = False
    width = height = 0
    records = []  # one dict per frame: id -> (x, y, quad)
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
                if use_mm:
                    camera_matrix, dist_coeffs, calibrated = load_calibration(
                        args.calibration, width, height)

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
                        if use_mm:
                            success, _rvec, tvec = cv2.solvePnP(
                                obj, points, camera_matrix, dist_coeffs,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            if not success:
                                continue
                            position = tvec.reshape(3)
                            x, y = float(position[0]), float(position[1])
                        else:
                            center = points.mean(axis=0)
                            x, y = float(center[0]), float(center[1])
                        found_now[marker_id] = (x, y, points)
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

    # Decide which IDs to keep
    seen = Counter(marker_id for record in records for marker_id in record)
    if args.marker_ids:
        ids = []
        for marker_id in dict.fromkeys(args.marker_ids):
            if seen.get(marker_id, 0) == 0:
                print(f"[WARNING] ID {marker_id} was never detected.")
            ids.append(marker_id)
    else:
        candidates = [(marker_id, c) for marker_id, c in seen.items()
                      if c >= args.min_detections]
        dropped_noise = len(seen) - len(candidates)
        candidates.sort(key=lambda item: (-item[1], item[0]))
        kept = candidates[:args.max_markers]
        if len(candidates) > len(kept):
            skipped = ", ".join(str(m) for m, _ in candidates[args.max_markers:])
            print(f"[WARNING] Showing the {len(kept)} most detected markers; "
                  f"not shown: {skipped} (use --marker-ids or --max-markers).")
        if dropped_noise:
            print(f"[INFO] Ignored {dropped_noise} ID(s) detected in fewer than "
                  f"{args.min_detections} frames.")
        ids = sorted(m for m, _ in kept)

    n_markers = len(ids)
    row_of = {marker_id: row for row, marker_id in enumerate(ids)}
    x_abs = np.full((n_markers, count), np.nan)
    y_abs = np.full((n_markers, count), np.nan)
    quads = np.full((n_markers, count, 4, 2), np.nan, np.float32)
    for frame_index, record in enumerate(records):
        for marker_id, (x, y, points) in record.items():
            row = row_of.get(marker_id)
            if row is None:
                continue
            x_abs[row, frame_index] = x
            y_abs[row, frame_index] = y
            quads[row, frame_index] = points
    del records

    found = np.isfinite(x_abs)
    dx = np.full_like(x_abs, np.nan)
    dy = np.full_like(y_abs, np.nan)
    path = np.zeros((n_markers, count))
    last_valid = np.full((n_markers, count), -1, dtype=np.int64)
    for row in range(n_markers):
        valid_idx = np.flatnonzero(found[row])
        if not valid_idx.size:
            continue
        if args.absolute:
            dx[row], dy[row] = x_abs[row], y_abs[row]
        else:
            first = valid_idx[0]
            dx[row] = x_abs[row] - x_abs[row, first]
            dy[row] = y_abs[row] - y_abs[row, first]
        last_valid[row] = np.maximum.accumulate(
            np.where(found[row], np.arange(count), -1))
        steps = np.hypot(np.diff(dx[row, valid_idx]), np.diff(dy[row, valid_idx]))
        cumulative = np.zeros(count)
        cumulative[valid_idx] = np.r_[0.0, np.cumsum(steps)]
        path[row] = np.where(last_valid[row] >= 0,
                             cumulative[np.maximum(last_valid[row], 0)], 0.0)

    include_origin = not args.absolute
    both = np.isfinite(dx) & np.isfinite(dy)
    xy_extent = None
    if both.any():
        xy_extent = (float(dx[both].min()), float(dx[both].max()),
                     float(dy[both].min()), float(dy[both].max()))
    return {
        "fps": fps, "n": count, "width": width, "height": height,
        "ids": ids, "colors": [PALETTE[i % len(PALETTE)] for i in range(n_markers)],
        "dx": dx, "dy": dy, "found": found, "quads": quads,
        "last_valid": last_valid, "path": path, "xy_extent": xy_extent,
        "range_x": padded_range(dx, include_origin),
        "range_y": padded_range(dy, include_origin),
        "include_origin": include_origin, "absolute": bool(args.absolute),
        "unit": "mm" if use_mm else "px", "calibrated": calibrated,
        "strip_cache": {},
    }


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


def nice_step(span, target=4):
    raw = span / max(1, target)
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def format_tick(value, step):
    decimals = 0 if step >= 1 else min(3, int(math.ceil(-math.log10(step))))
    return f"{value:.{decimals}f}"


def axis_names(data):
    return ("X", "Y") if data["absolute"] else ("dX", "dY")


# --------------------------------------------------------------------------
# Dashboard parts
# --------------------------------------------------------------------------
def draw_header(canvas, args, data, rect, index, paused, k):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), BG, -1)
    family = "APRILTAG" if is_apriltag(args.dictionary) else "ARUCO"
    title = os.path.basename(args.video)
    put(canvas, title, (x0 + int(18 * k), y0 + int(27 * k)), 0.62 * k, TEXT, 1)

    total = len(data["ids"])
    now = int(data["found"][:, index].sum()) if total else 0
    id_list = ", ".join(str(m) for m in data["ids"]) or "none"
    info = f"{family} | {args.dictionary} | {total} markers: IDs {id_list}"
    badge_text = f"DETECTED {now}/{total}" if total else "NO MARKERS"
    badge_w = text_width(badge_text, 0.45 * k) + int(30 * k)
    info_scale = fit_scale(info, (x1 - x0) - badge_w - int(60 * k), 0.42 * k)
    put(canvas, info, (x0 + int(18 * k), y0 + int(51 * k)), info_scale, MUTED, 1)

    if total == 0 or now == 0:
        color = RED
    elif now < total:
        color = AMBER
    else:
        color = GREEN
    cv2.rectangle(canvas, (x1 - badge_w - int(18 * k), y0 + int(13 * k)),
                  (x1 - int(18 * k), y0 + int(48 * k)), color, -1)
    put(canvas, badge_text, (x1 - badge_w - int(3 * k), y0 + int(37 * k)),
        0.45 * k, (20, 23, 28), 1)

    state = "PAUSED" if paused else "PLAYING"
    timestamp = index / data["fps"]
    right = f"{state}  |  {timestamp:.2f} s  |  frame {index + 1}/{data['n']}"
    put(canvas, right, (x1 - text_width(right, 0.42 * k) - int(18 * k),
                        y1 - int(8 * k)), 0.42 * k, MUTED, 1)


def draw_big_video(canvas, frame, data, index, rect, k):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (7, 9, 12), -1)
    if frame is None:
        put(canvas, "No frame", (x0 + int(20 * k), y0 + int(35 * k)),
            0.6 * k, MUTED)
        return
    height, width = frame.shape[:2]
    area_w, area_h = x1 - x0, y1 - y0
    scale = min(area_w / width, area_h / height)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    shown = cv2.resize(frame, (new_w, new_h),
                       interpolation=(cv2.INTER_AREA if scale < 1
                                      else cv2.INTER_LINEAR))
    for row, marker_id in enumerate(data["ids"]):
        quad = data["quads"][row, index]
        if not np.isfinite(quad).all():
            continue
        color = data["colors"][row]
        points = (quad * scale).astype(np.int32)
        cv2.polylines(shown, [points.reshape(-1, 1, 2)], True, color,
                      max(2, int(3 * k)), cv2.LINE_AA)
        center = points.mean(axis=0).astype(int)
        cv2.circle(shown, tuple(center), max(3, int(4 * k)), color, -1,
                   cv2.LINE_AA)
        anchor = points[np.argmin(points[:, 1])]
        put_outlined(shown, f"ID {marker_id}",
                     (int(anchor[0]), max(int(14 * k), int(anchor[1]) - int(8 * k))),
                     0.55 * k, color)
    ox = x0 + (area_w - new_w) // 2
    oy = y0 + (area_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = shown
    cv2.rectangle(canvas, (x0, y0), (x1, y1), BORDER, 1)


def draw_xy_plane(canvas, data, rect, index, k):
    """Accumulated XY trajectory of every marker up to the current frame.

    The scale is fixed by the whole video (equal units on both axes), so the
    plane does not rescale while playing. The part of each path still to come
    is shown faintly.
    """
    x0, y0, x1, y1 = rect
    panel(canvas, rect)
    unit = data["unit"]
    title = ("XY TRAJECTORIES (POSITION)" if data["absolute"]
             else "ACCUMULATED XY DISPLACEMENT")
    title_scale = fit_scale(title, (x1 - x0) - int(24 * k), 0.48 * k)
    put(canvas, title, (x0 + int(14 * k), y0 + int(24 * k)), title_scale, MUTED)

    left, right = x0 + int(50 * k), x1 - int(16 * k)
    top, bottom = y0 + int(40 * k), y1 - int(46 * k)
    if right - left < 40 or bottom - top < 40:
        return
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_2, -1)

    extent = data["xy_extent"]
    if extent is None:
        put(canvas, "no detections", (left + int(12 * k), (top + bottom) // 2),
            0.5 * k, MUTED)
        return

    xmin, xmax, ymin, ymax = extent
    if data["include_origin"]:
        xmin, xmax = min(xmin, 0.0), max(xmax, 0.0)
        ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
    span_x, span_y = xmax - xmin, ymax - ymin
    if span_x < 1e-9 and span_y < 1e-9:
        span_x = span_y = 1.0
    span_x, span_y = max(span_x, 1e-9), max(span_y, 1e-9)
    avail_w, avail_h = right - left, bottom - top
    scale = min(avail_w / (span_x * 1.16), avail_h / (span_y * 1.16))
    mid_x, mid_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0

    def to_px(values):
        return cx + (np.asarray(values) - mid_x) * scale

    def to_py(values):  # +Y points down, like the video
        return cy + (np.asarray(values) - mid_y) * scale

    x_lo, x_hi = mid_x - (avail_w / 2) / scale, mid_x + (avail_w / 2) / scale
    y_lo, y_hi = mid_y - (avail_h / 2) / scale, mid_y + (avail_h / 2) / scale
    step = nice_step(max(x_hi - x_lo, y_hi - y_lo), 5)
    tick_scale = 0.34 * k
    for value in np.arange(math.ceil(x_lo / step) * step, x_hi, step):
        xx = int(to_px(value))
        cv2.line(canvas, (xx, top), (xx, bottom), BORDER, 1)
        label = format_tick(value, step)
        put(canvas, label, (xx - text_width(label, tick_scale) // 2,
                            bottom + int(14 * k)), tick_scale, MUTED)
    for value in np.arange(math.ceil(y_lo / step) * step, y_hi, step):
        yy = int(to_py(value))
        cv2.line(canvas, (left, yy), (right, yy), BORDER, 1)
        label = format_tick(value, step)
        put(canvas, label, (left - text_width(label, tick_scale) - int(5 * k),
                            yy + int(4 * k)), tick_scale, MUTED)

    if data["include_origin"]:
        ox, oy = int(to_px(0.0)), int(to_py(0.0))
        if left <= ox <= right and top <= oy <= bottom:
            cv2.line(canvas, (left, oy), (right, oy), MUTED, 1)
            cv2.line(canvas, (ox, top), (ox, bottom), MUTED, 1)
            cv2.circle(canvas, (ox, oy), max(4, int(5 * k)), MUTED, 1, cv2.LINE_AA)
    put(canvas, "+X", (right - text_width("+X", tick_scale * 1.2) - int(4 * k),
                       top + int(14 * k)), tick_scale * 1.2, MUTED)
    put(canvas, "+Y", (left + int(5 * k), bottom - int(5 * k)),
        tick_scale * 1.2, MUTED)

    total = data["n"]
    stride_all = max(1, total // 4000)
    all_idx = np.arange(0, total, stride_all)
    stride = max(1, (index + 1) // 4000)
    idx = np.unique(np.r_[np.arange(0, index + 1, stride), index])
    dx, dy = data["dx"], data["dy"]
    both = np.isfinite(dx) & np.isfinite(dy)

    # Faint full paths first, then the accumulated ones on top
    for row in range(len(data["ids"])):
        draw_runs(canvas, to_px(dx[row, all_idx]), to_py(dy[row, all_idx]),
                  both[row, all_idx], BORDER, 1)
    for row in range(len(data["ids"])):
        color = data["colors"][row]
        draw_runs(canvas, to_px(dx[row, idx]), to_py(dy[row, idx]),
                  both[row, idx], color, max(2, int(2 * k)))

    # Current positions (last known one if the marker is not detected now)
    for row, marker_id in enumerate(data["ids"]):
        last = int(data["last_valid"][row, index])
        if last < 0:
            continue
        color = data["colors"][row]
        detected_now = bool(data["found"][row, index])
        px, py = int(to_px(dx[row, last])), int(to_py(dy[row, last]))
        cv2.circle(canvas, (px, py), max(5, int(7 * k)),
                   WHITE if detected_now else RED, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), max(3, int(4 * k)), color, -1, cv2.LINE_AA)
        put_outlined(canvas, str(marker_id),
                     (px + int(9 * k), py - int(7 * k)), 0.42 * k, color)

    total_markers = len(data["ids"])
    now = int(data["found"][:, index].sum())
    line1 = f"detected now: {now}/{total_markers}"
    line2 = f"units: {unit}  |  equal axes"
    put(canvas, line1, (x0 + int(14 * k), y1 - int(24 * k)),
        fit_scale(line1, (x1 - x0) - int(24 * k), 0.46 * k), TEXT)
    put(canvas, line2, (x0 + int(14 * k), y1 - int(8 * k)),
        fit_scale(line2, (x1 - x0) - int(24 * k), 0.4 * k), MUTED)


def draw_time_plot(canvas, data, rect, index, k, axis):
    """dX(t) or dY(t) of every marker over the whole video."""
    x0, y0, x1, y1 = rect
    panel(canvas, rect)
    names = axis_names(data)
    name = names[0] if axis == "x" else names[1]
    values = data["dx"] if axis == "x" else data["dy"]
    ymin, ymax = data["range_x"] if axis == "x" else data["range_y"]
    unit = data["unit"]
    fps, count = data["fps"], data["n"]
    duration = max((count - 1) / fps, 1e-6)

    put(canvas, f"{name} OVER TIME ({unit})", (x0 + int(16 * k), y0 + int(24 * k)),
        0.48 * k, MUTED)

    left, right = x0 + int(60 * k), x1 - int(14 * k)
    top, bottom = y0 + int(36 * k), y1 - int(26 * k)
    if right - left < 40 or bottom - top < 30:
        return

    def to_x(seconds):
        return left + np.asarray(seconds) / duration * (right - left)

    def to_y(vals):
        return bottom - (np.asarray(vals) - ymin) / (ymax - ymin) * (bottom - top)

    for value in np.linspace(ymin, ymax, 5):
        yy = int(to_y(value))
        cv2.line(canvas, (left, yy), (right, yy), BORDER, 1)
        put(canvas, f"{value:.1f}", (x0 + int(6 * k), yy + int(4 * k)),
            0.34 * k, MUTED)
    for seconds in np.linspace(0, duration, 6):
        xx = int(to_x(seconds))
        cv2.line(canvas, (xx, top), (xx, bottom), BORDER, 1)
        label = f"{seconds:.1f}s"
        put(canvas, label, (xx - text_width(label, 0.33 * k) // 2,
                            y1 - int(8 * k)), 0.33 * k, MUTED)
    if data["include_origin"]:
        zero = int(to_y(0.0))
        cv2.line(canvas, (left, zero), (right, zero), MUTED, 1)

    times = np.arange(count) / fps
    stride = max(1, count // max(1, 2 * (right - left)))
    sampled_times = times[::stride]
    thickness = max(1, int(round(2 * k)))
    for row in range(len(data["ids"])):
        sampled = values[row, ::stride]
        valid = np.isfinite(sampled)
        draw_runs(canvas, to_x(sampled_times),
                  to_y(np.where(valid, sampled, 0.0)), valid,
                  data["colors"][row], thickness)

    cursor_x = int(to_x(index / fps))
    cv2.line(canvas, (cursor_x, top), (cursor_x, bottom), WHITE,
             max(1, int(round(2 * k))), cv2.LINE_AA)


def strip_fraction(data, row, width):
    """Fraction of detected frames in each pixel column (cached)."""
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
    """One detection strip per marker, labeled in its color, with live values."""
    x0, y0, x1, y1 = rect
    total = len(data["ids"])
    label_w = text_width("ID 000", 0.4 * k) + int(16 * k)
    values_w = int(min(250 * k, (x1 - x0) * 0.30))
    sx0, sx1 = x0 + label_w, x1 - values_w - int(8 * k)
    if total == 0 or sx1 - sx0 < 40:
        return
    width = sx1 - sx0
    names = axis_names(data)
    unit = data["unit"]
    base = np.array(BORDER, float)

    for row, marker_id in enumerate(data["ids"]):
        ry0 = y0 + row * (row_h + row_gap)
        ry1 = ry0 + row_h
        color = data["colors"][row]
        fraction = strip_fraction(data, row, width)
        strip = (base[None, :] * (1 - fraction[:, None]) +
                 np.array(color, float)[None, :] * fraction[:, None])
        canvas[ry0:ry1, sx0:sx1] = np.repeat(strip[None, :, :], row_h,
                                             axis=0).astype(np.uint8)
        put(canvas, f"ID {marker_id}", (x0, ry1 - max(1, row_h // 5)),
            0.4 * k, color, 1)

        last = int(data["last_valid"][row, index])
        detected_now = bool(data["found"][row, index])
        if last >= 0:
            text = (f"{names[0]} {data['dx'][row, last]:+.1f}  "
                    f"{names[1]} {data['dy'][row, last]:+.1f}  "
                    f"path {data['path'][row, index]:.1f}")
        else:
            text = "--"
        put(canvas, text, (sx1 + int(8 * k), ry1 - max(1, row_h // 5)),
            fit_scale(text, values_w, 0.38 * k), TEXT if detected_now else MUTED)

    cursor = sx0 + int(index / max(1, data["n"] - 1) * (width - 1))
    block_h = total * (row_h + row_gap) - row_gap
    cv2.line(canvas, (cursor, y0 - 3), (cursor, y0 + block_h + 3), WHITE,
             max(1, int(round(2 * k))), cv2.LINE_AA)
    put(canvas, f"Detection per marker  ({unit} | path = accumulated distance)",
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

    # The XY plane is roughly square; the video takes everything else.
    xy_w = int(min(max(body_h * 0.95, width * 0.22), width * 0.40))
    xy_x1 = width - margin
    xy_x0 = xy_x1 - xy_w
    video_rect = (margin, body_top, xy_x0 - margin, body_bottom)
    xy_rect = (xy_x0, body_top, xy_x1, body_bottom)

    plot_w = (width - 3 * margin) // 2
    plot_left = (margin, plots_top, margin + plot_w, plots_bottom)
    plot_right = (margin * 2 + plot_w, plots_top, width - margin, plots_bottom)

    draw_header(canvas, args, data, (0, 0, width, header_h), index, paused, k)
    draw_big_video(canvas, frame, data, index, video_rect, k)
    draw_xy_plane(canvas, data, xy_rect, index, k)
    draw_time_plot(canvas, data, plot_left, index, k, "x")
    draw_time_plot(canvas, data, plot_right, index, k, "y")
    draw_detection_rows(canvas, data, (margin, block_top, width - margin, height - margin),
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
    stem = os.path.splitext(video_path)[0] + "_multi_dashboard"
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
    print(f"[OK] {data['n']} frames @ {data['fps']:.2f} fps | "
          f"{len(data['ids'])} marker(s) | {data['unit']}"
          f"{' (absolute position)' if data['absolute'] else ''}")
    for row, marker_id in enumerate(data["ids"]):
        rate = 100.0 * float(data["found"][row].mean())
        print(f"  ID {marker_id:>3}: detected in {rate:5.1f}% of frames | "
              f"path {data['path'][row, -1]:.2f} {data['unit']}")
    if not data["ids"]:
        print("[WARNING] No marker found. Check the dictionary and the IDs; "
              "for small markers, try --detect-scale 1.0.")
    if data["unit"] == "mm" and not data["calibrated"]:
        print("[WARNING] Approximate intrinsics: values in mm are not metrological.")

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
