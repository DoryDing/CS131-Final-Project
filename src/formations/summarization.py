# Take the formation segments from segmentation.py and the top-down positions from homography.py,
# and for each stable segment compute a single canonical (x, y) per member

import json
import numpy as np
from pathlib import Path


def recenter_formation(positions, stage_cx=50.0, stage_cy=50.0):
    """Translate all member positions so their centroid sits at (stage_cx, stage_cy)."""
    members = sorted(positions.keys())
    pts = np.array([[positions[m]["x"], positions[m]["y"]] for m in members], dtype=float)
    shift = np.array([stage_cx, stage_cy]) - pts.mean(axis=0)
    pts += shift
    return {m: {"x": round(float(pts[i, 0]), 3), "y": round(float(pts[i, 1]), 3)}
            for i, m in enumerate(members)}


def snap_to_grid(positions, grid=10, offset=0):
    """
    Snap each member's (x, y) to the nearest grid point.
    """
    def snap(v):
        return round((v - offset) / grid) * grid + offset

    return {m: {"x": snap(pos["x"]), "y": snap(pos["y"])}
            for m, pos in positions.items()}


def summarize_formation(by_frame, segment):
    start = segment["start_frame"]
    end = segment["end_frame"]
    total = end - start + 1

    #only use the middle 50% of each stable segment for the median
    trim = total // 4
    start = start + trim
    end = end - trim
    # collect all (x, y) observations per member across the segment
    member_positions = {}
    for f in range(start, end + 1):
        if f not in by_frame:
            continue
        for member_id, (x, y) in by_frame[f].items():
            if member_id not in member_positions:
                member_positions[member_id] = {"xs": [], "ys": []}
            member_positions[member_id]["xs"].append(x)
            member_positions[member_id]["ys"].append(y)

    # compute median x and y for each member
    positions = {}
    for member_id, coords in member_positions.items():
        positions[member_id] = {
            "x": round(float(np.median(coords["xs"])), 3),
            "y": round(float(np.median(coords["ys"])), 3),
        }

    # copy segment metadata and attach positions
    result = {
        "formation_id": segment["formation_id"],
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "n_frames": segment["n_frames"],
        "positions": positions,
    }

    # attach timestamps only if the segment has them
    if "start_time_s" in segment:
        result["start_time_s"] = segment["start_time_s"]
        result["end_time_s"] = segment["end_time_s"]

    return result

def summarize_all(top_down, segments):
    # re-index once
    by_frame = {}
    for entry in top_down:
        f = int(entry["frame"])
        m = int(entry["member_id"])
        if f not in by_frame:
            by_frame[f] = {}
        by_frame[f][m] = (entry["x"], entry["y"])

    results = []
    for segment in segments:
        summary = summarize_formation(by_frame, segment)
        summary["positions"] = recenter_formation(summary["positions"])
        summary["positions"] = snap_to_grid(summary["positions"])
        results.append(summary)

    return results


if __name__ == "__main__":
    from homography import top_down_positions
    from segmentation import run_segmentation

    project_root = Path(__file__).resolve().parents[2]
    tracks_path = project_root / "data" / "processed" / "synthetic_tracks.json"
    calibration_path = project_root / "data" / "processed" / "calibration.json"

    with open(tracks_path) as f:
        tracks_data = json.load(f)
    fps = tracks_data["video"]["fps"]

    top_down = top_down_positions(str(tracks_path), str(calibration_path))
    segments = run_segmentation(top_down, fps=fps)
    formations = summarize_all(top_down, segments)

    print(f"summarized {len(formations)} formation(s):\n")
    for formation in formations:
        print(f"formation {formation['formation_id']}  frames {formation['start_frame']}–{formation['end_frame']}")
        for member_id, pos in formation["positions"].items():
            print(f"  member {member_id}: x={pos['x']}, y={pos['y']}")
        print()