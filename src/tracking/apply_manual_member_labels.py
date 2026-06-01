import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_range_lookup(label_config):
    """
    Build a dictionary from track_id to all manual ranges for that track.

    Example:
    {
        2: [
            {"member_id": 2, "start_frame": 0, "end_frame": 1613},
            {"member_id": 4, "start_frame": 1614, "end_frame": 4553}
        ]
    }
    """
    lookup = defaultdict(list)

    for rule in label_config["ranges"]:
        track_id = int(rule["track_id"])

        lookup[track_id].append({
            "member_id": rule["member_id"],
            "start_frame": int(rule["start_frame"]),
            "end_frame": int(rule["end_frame"])
        })

    # Sort each track's rules by start frame for easier debugging.
    for track_id in lookup:
        lookup[track_id].sort(key=lambda r: r["start_frame"])

    return lookup


def find_member_id_for_item(item, range_lookup):
    """
    Given one tracked detection item, find its manually assigned member_id.

    Matching condition:
    - same track_id
    - frame is inside [start_frame, end_frame]
    """
    track_id = item.get("track_id")
    frame = item.get("frame")

    if track_id is None or frame is None:
        return None

    track_id = int(track_id)
    frame = int(frame)

    if track_id not in range_lookup:
        return None

    matched_member_ids = []

    for rule in range_lookup[track_id]:
        if rule["start_frame"] <= frame <= rule["end_frame"]:
            matched_member_ids.append(rule["member_id"])

    if len(matched_member_ids) == 0:
        return None

    if len(matched_member_ids) > 1:
        print(
            f"Warning: multiple member labels found for "
            f"track_id={track_id}, frame={frame}: {matched_member_ids}. "
            f"Using the first one."
        )

    return matched_member_ids[0]


def apply_manual_labels(tracks_data, label_config):
    """
    Apply manual member labels to a tracking JSON.

    Keeps original track_id unchanged.
    Replaces member_id from None to manually assigned member ID.
    """

    range_lookup = build_range_lookup(label_config)

    output_data = dict(tracks_data)
    output_tracks = []

    assigned_count = 0
    unassigned_count = 0

    assigned_by_member = Counter()
    unassigned_track_ids = Counter()

    for item in tracks_data.get("tracks", []):
        new_item = dict(item)

        member_id = find_member_id_for_item(new_item, range_lookup)

        if member_id is None:
            unassigned_count += 1
            unassigned_track_ids[new_item.get("track_id")] += 1
            new_item["member_id"] = None
        else:
            assigned_count += 1
            assigned_by_member[member_id] += 1
            new_item["member_id"] = member_id

        output_tracks.append(new_item)

    output_data["tracks"] = output_tracks
    output_data["members"] = label_config.get("members", [])
    output_data["member_legend"] = label_config.get("member_legend", {})

    existing_experiment = output_data.get("experiment", {})
    output_data["experiment"] = dict(existing_experiment)
    output_data["experiment"]["manual_member_labels"] = True
    output_data["experiment"]["member_label_source"] = "configs/manual_track_merges_exp04.json"

    summary = {
        "assigned_count": assigned_count,
        "unassigned_count": unassigned_count,
        "assigned_by_member": dict(assigned_by_member),
        "unassigned_track_ids": dict(unassigned_track_ids),
    }

    return output_data, summary


def print_summary(summary):
    print("\n=== Manual Member Labeling Summary ===")
    print(f"Assigned detections: {summary['assigned_count']}")
    print(f"Unassigned detections: {summary['unassigned_count']}")

    print("\n--- Assigned by member_id ---")
    for member_id, count in sorted(summary["assigned_by_member"].items(), key=lambda x: int(x[0])):
        print(f"member_id {member_id}: {count} detections")

    if summary["unassigned_track_ids"]:
        print("\n--- Unassigned track IDs ---")
        for track_id, count in sorted(
            summary["unassigned_track_ids"].items(),
            key=lambda x: (-x[1], str(x[0]))
        ):
            print(f"track_id {track_id}: {count} detections")
    else:
        print("\nNo unassigned track IDs.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply manual member labels to tracking results using frame ranges."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_merged_gap180_dist450.json",
        help="Input tracks JSON."
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="configs/manual_track_merges_exp04.json",
        help="Manual member label config JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_member_labeled.json",
        help="Output tracks JSON with member_id assigned."
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/member_label_summary.json",
        help="Output summary JSON."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    labels_path = Path(args.labels)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find label config file: {labels_path}")

    tracks_data = load_json(input_path)
    label_config = load_json(labels_path)

    labeled_data, summary = apply_manual_labels(tracks_data, label_config)

    save_json(output_path, labeled_data)
    save_json(summary_output_path, summary)

    print_summary(summary)

    print("\nSaved files:")
    print(f"- Labeled tracks: {output_path}")
    print(f"- Label summary: {summary_output_path}")


if __name__ == "__main__":
    main()