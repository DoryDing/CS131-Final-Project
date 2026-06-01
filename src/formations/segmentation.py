import json
import math
import numpy as np
from pathlib import Path
from typing import Optional

def compute_motion_energy(top_down, min_members_for_energy=2):
    # reorganize the flat list into {frame: {member_id: (x, y)}}
    # so its easier to look up positions by frame
    by_frame = {}
    # reshape  the top_down flat list into a nested dictionary
    for entry in top_down:
        f = int(entry["frame"])
        m = int(entry["member_id"])
        if f not in by_frame:
            by_frame[f] = {}
        by_frame[f][m] = (entry["x"], entry["y"])

    frames = sorted(by_frame.keys())
    energy = {}
    energy[frames[0]] = 0.0  # no previous frame to compare to

    for i in range(1, len(frames)):
        prev_f = frames[i - 1]
        curr_f = frames[i]

        # only look at members we can actually track between both frames
        shared_members = set(by_frame[prev_f].keys()) & set(by_frame[curr_f].keys())

        if len(shared_members) < min_members_for_energy:
            # If fewer than 2 members are trackable between these two frames, we can't compute a meaningful average
            energy[curr_f] = float("nan")
            continue

        total_dist = 0
        for m in shared_members:
            # For each shared member, compute their Euclidean distance between the two frames
            # Sum all distances, then divide by the number of members to get the average displacement, which is the
            # total energy of the frame. A high number means people are moving a lot;
            # a low number means they're holding still.
            x1, y1 = by_frame[prev_f][m]
            x2, y2 = by_frame[curr_f][m]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_dist += dist

        energy[curr_f] = total_dist / len(shared_members)

    return energy

def segment_formations(energy, threshold=2.0, min_stable_frames=15):
    frames = sorted(energy.keys())

    # Two conditions must both be true for a frame to count as stable:
    # its energy must not be nan, it must be below the threshold
    stable = []
    for f in frames:
        if not math.isnan(energy[f]) and energy[f] < threshold:
            stable.append(f)

    if len(stable) == 0:
        return []

    segments = []
    seg_start = stable[0]
    seg_prev = stable[0]

    # keep two pointers in this algorithm:
    # seg_start marks where the current run began, seg_prev marks the last stable frame
    # As long as each new frame is exactly seg_prev + 1, the run continues and we advance seg_prev.
    # The moment there's a gap, we save the completed run and reset both pointers to the new frame.
    for f in stable[1:]:
        if f == seg_prev + 1:
            seg_prev = f
        else:
            segments.append((seg_start, seg_prev))
            seg_start = f
            seg_prev = f

    segments.append((seg_start, seg_prev))  # append the last frame

    # filters out segments that are too short
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_stable_frames]

    # format into a clean list of dicts
    result = []
    for i, (s, e) in enumerate(segments):
        result.append({
            "formation_id": i + 1,
            "start_frame": s,
            "end_frame": e,
            "n_frames": e - s + 1,
        })

    return result

# sanity check of the segmentation.py file
def run_segmentation(top_down, threshold=2.0, min_stable_frames=15, min_members_for_energy=2, fps=None):
    energy = compute_motion_energy(top_down, min_members_for_energy)
    segments = segment_formations(energy, threshold, min_stable_frames)

    # add timestamps if we know the fps
    if fps is not None:
        for seg in segments:
            seg["start_time_s"] = round(seg["start_frame"] / fps, 3)
            seg["end_time_s"] = round(seg["end_frame"] / fps, 3)

    return segments

if __name__ == "__main__":
    from homography import top_down_positions

    project_root = Path(__file__).resolve().parents[2]
    tracks_path = project_root / "data" / "processed" / "synthetic_tracks.json"
    calibration_path = project_root / "data" / "processed" / "calibration.json"

    with open(tracks_path) as f:
        tracks_data = json.load(f)
    fps = tracks_data["video"]["fps"]

    top_down = top_down_positions(str(tracks_path), str(calibration_path))

    # print some energy values first to see if the threshold makes sense
    energy = compute_motion_energy(top_down)
    print("first 10 energy values:")
    for f in sorted(energy.keys())[:10]:
        print(f"  frame {f}: {energy[f]}")

    segments = run_segmentation(top_down, fps=fps)
    print(f"\nfound {len(segments)} formations")
    for seg in segments:
        print(seg)
