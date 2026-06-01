import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def lerp(a, b, t):
    """Linear interpolation from a to b."""
    return a + (b - a) * t


def interpolate_bbox(bbox_a, bbox_b, t):
    """Linearly interpolate two bounding boxes."""
    return [
        lerp(bbox_a[0], bbox_b[0], t),
        lerp(bbox_a[1], bbox_b[1], t),
        lerp(bbox_a[2], bbox_b[2], t),
        lerp(bbox_a[3], bbox_b[3], t),
    ]


def group_tracks_by_member_and_frame(tracks):
    """
    Returns:
    {
        member_id: {
            frame: item
        }
    }

    If there are multiple detections for the same member in one frame,
    we keep the one with the highest confidence.
    """
    grouped = defaultdict(dict)

    duplicate_count = 0

    for item in tracks:
        member_id = item.get("member_id")
        frame = item.get("frame")

        if member_id is None or frame is None:
            continue

        member_id = int(member_id)
        frame = int(frame)

        existing = grouped[member_id].get(frame)

        if existing is None:
            grouped[member_id][frame] = item
        else:
            duplicate_count += 1
            existing_conf = existing.get("conf", 0.0)
            new_conf = item.get("conf", 0.0)

            if new_conf > existing_conf:
                grouped[member_id][frame] = item

    return grouped, duplicate_count


def interpolate_member_tracks(data, max_gap=None):
    """
    Fill missing frames for each member using linear interpolation.

    If max_gap is None, interpolate all gaps.
    If max_gap is an int, only interpolate gaps with length <= max_gap.

    Output includes both real and interpolated detections.
    """

    video_info = data.get("video", {})
    n_frames = video_info.get("n_frames")

    if n_frames is None:
        raise ValueError("Input JSON must contain video.n_frames")

    members = data.get("members", [])
    tracks = data.get("tracks", [])

    by_member, duplicate_count = group_tracks_by_member_and_frame(tracks)

    output_tracks = []
    interpolation_summary = {
        "duplicate_member_frame_detections": duplicate_count,
        "members": {}
    }

    for member_id in members:
        member_id = int(member_id)

        frame_to_item = by_member.get(member_id, {})
        real_frames = sorted(frame_to_item.keys())

        member_summary = {
            "real_detections": len(real_frames),
            "interpolated_detections": 0,
            "unfilled_missing_frames": 0,
            "gaps_interpolated": [],
            "gaps_not_interpolated": []
        }

        if not real_frames:
            # No detections at all for this member. Cannot interpolate.
            member_summary["unfilled_missing_frames"] = n_frames
            interpolation_summary["members"][str(member_id)] = member_summary
            continue

        # First, add all real detections.
        for frame in real_frames:
            item = dict(frame_to_item[frame])
            item["is_interpolated"] = False
            item["interp_gap"] = 0
            item["interp_source"] = None
            output_tracks.append(item)

        # Fill gaps between consecutive real detections.
        for prev_frame, next_frame in zip(real_frames[:-1], real_frames[1:]):
            gap_length = next_frame - prev_frame - 1

            if gap_length <= 0:
                continue

            should_interpolate = (
                max_gap is None or gap_length <= max_gap
            )

            prev_item = frame_to_item[prev_frame]
            next_item = frame_to_item[next_frame]

            if should_interpolate:
                member_summary["gaps_interpolated"].append({
                    "start_missing_frame": prev_frame + 1,
                    "end_missing_frame": next_frame - 1,
                    "gap_length": gap_length,
                    "prev_real_frame": prev_frame,
                    "next_real_frame": next_frame
                })

                for frame in range(prev_frame + 1, next_frame):
                    t = (frame - prev_frame) / (next_frame - prev_frame)

                    interp_bbox = interpolate_bbox(
                        prev_item["bbox"],
                        next_item["bbox"],
                        t
                    )

                    interp_item = {
                        "frame": frame,
                        "member_id": member_id,
                        "track_id": prev_item.get("track_id"),
                        "bbox": interp_bbox,
                        "conf": 0.0,
                        "is_interpolated": True,
                        "interp_gap": gap_length,
                        "interp_source": "linear_between_member_detections",
                        "prev_real_frame": prev_frame,
                        "next_real_frame": next_frame,
                        "prev_track_id": prev_item.get("track_id"),
                        "next_track_id": next_item.get("track_id")
                    }

                    # Preserve original_track_id if available, but mark it as unreliable.
                    if "original_track_id" in prev_item:
                        interp_item["original_track_id"] = prev_item.get("original_track_id")

                    output_tracks.append(interp_item)
                    member_summary["interpolated_detections"] += 1

            else:
                member_summary["gaps_not_interpolated"].append({
                    "start_missing_frame": prev_frame + 1,
                    "end_missing_frame": next_frame - 1,
                    "gap_length": gap_length,
                    "prev_real_frame": prev_frame,
                    "next_real_frame": next_frame
                })
                member_summary["unfilled_missing_frames"] += gap_length

        # Missing frames before first real detection and after last real detection
        # cannot be interpolated because we do not have both endpoints.
        first_real = real_frames[0]
        last_real = real_frames[-1]

        if first_real > 0:
            member_summary["unfilled_missing_frames"] += first_real
            member_summary["gaps_not_interpolated"].append({
                "start_missing_frame": 0,
                "end_missing_frame": first_real - 1,
                "gap_length": first_real,
                "reason": "before_first_detection"
            })

        if last_real < n_frames - 1:
            trailing_gap = n_frames - 1 - last_real
            member_summary["unfilled_missing_frames"] += trailing_gap
            member_summary["gaps_not_interpolated"].append({
                "start_missing_frame": last_real + 1,
                "end_missing_frame": n_frames - 1,
                "gap_length": trailing_gap,
                "reason": "after_last_detection"
            })

        interpolation_summary["members"][str(member_id)] = member_summary

    # Sort by frame, then member_id for clean downstream use.
    output_tracks.sort(key=lambda x: (x["frame"], int(x["member_id"])))

    output_data = dict(data)
    output_data["tracks"] = output_tracks

    existing_experiment = output_data.get("experiment", {})
    output_data["experiment"] = dict(existing_experiment)
    output_data["experiment"]["interpolation"] = {
        "enabled": True,
        "method": "linear_between_member_detections",
        "max_gap": max_gap
    }

    return output_data, interpolation_summary


def print_summary(summary):
    print("\n=== Interpolation Summary ===")
    print(f"Duplicate member-frame detections resolved: {summary['duplicate_member_frame_detections']}")

    total_real = 0
    total_interp = 0
    total_unfilled = 0

    for member_id, member_summary in sorted(summary["members"].items(), key=lambda x: int(x[0])):
        real = member_summary["real_detections"]
        interp = member_summary["interpolated_detections"]
        unfilled = member_summary["unfilled_missing_frames"]

        total_real += real
        total_interp += interp
        total_unfilled += unfilled

        print(f"\nmember_id {member_id}:")
        print(f"  real detections: {real}")
        print(f"  interpolated detections: {interp}")
        print(f"  unfilled missing frames: {unfilled}")
        print(f"  interpolated gaps: {len(member_summary['gaps_interpolated'])}")
        print(f"  not interpolated gaps: {len(member_summary['gaps_not_interpolated'])}")

        # Print the largest few interpolated gaps.
        largest_gaps = sorted(
            member_summary["gaps_interpolated"],
            key=lambda g: g["gap_length"],
            reverse=True
        )[:5]

        if largest_gaps:
            print("  largest interpolated gaps:")
            for gap in largest_gaps:
                print(
                    f"    frames {gap['start_missing_frame']}–{gap['end_missing_frame']} "
                    f"length={gap['gap_length']}"
                )

    print("\n--- Total ---")
    print(f"Real detections: {total_real}")
    print(f"Interpolated detections: {total_interp}")
    print(f"Unfilled missing frames: {total_unfilled}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate missing member detections after manual member labeling."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_member_labeled.json",
        help="Input member-labeled tracks JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_member_interpolated.json",
        help="Output tracks JSON with interpolated missing member detections."
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/interpolation_summary.json",
        help="Output interpolation summary JSON."
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=None,
        help=(
            "Maximum missing gap length to interpolate. "
            "If omitted, interpolates all internal gaps."
        )
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    data = load_json(input_path)

    output_data, summary = interpolate_member_tracks(
        data,
        max_gap=args.max_gap
    )

    save_json(output_path, output_data)
    save_json(summary_output_path, summary)

    print_summary(summary)

    print("\nSaved files:")
    print(f"- Interpolated tracks: {output_path}")
    print(f"- Interpolation summary: {summary_output_path}")


if __name__ == "__main__":
    main()