import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def bbox_center(bbox):
    """Return bbox center (cx, cy) for bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_bottom_center(bbox):
    """
    Return bottom-center point of bbox.

    For dancer tracking, bottom-center is often closer to the dancer's stage position
    than the geometric center of the box.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def group_by_track(tracks):
    """Group track items by their track_id."""
    grouped = defaultdict(list)

    for item in tracks:
        track_id = item.get("track_id")

        if track_id is not None:
            grouped[track_id].append(item)

    for track_id in grouped:
        grouped[track_id] = sorted(grouped[track_id], key=lambda x: x["frame"])

    return grouped


def summarize_track(track_id, items, point_type="bottom_center"):
    """Create a summary record for one track ID."""
    frames = [item["frame"] for item in items]

    first_item = items[0]
    last_item = items[-1]

    if point_type == "center":
        first_point = bbox_center(first_item["bbox"])
        last_point = bbox_center(last_item["bbox"])
    elif point_type == "bottom_center":
        first_point = bbox_bottom_center(first_item["bbox"])
        last_point = bbox_bottom_center(last_item["bbox"])
    else:
        raise ValueError(f"Unknown point_type: {point_type}")

    avg_conf = sum(item.get("conf", 0.0) for item in items) / len(items)

    return {
        "track_id": track_id,
        "first_frame": min(frames),
        "last_frame": max(frames),
        "num_detections": len(items),
        "first_point": first_point,
        "last_point": last_point,
        "avg_conf": avg_conf,
    }


def build_track_summaries(grouped_tracks, point_type="bottom_center"):
    summaries = {}

    for track_id, items in grouped_tracks.items():
        summaries[track_id] = summarize_track(
            track_id,
            items,
            point_type=point_type
        )

    return summaries


def find_best_successor(
    source_id,
    summaries,
    used_as_successor,
    max_gap,
    max_distance,
):
    """
    Find the best track fragment that should follow source_id.

    We only consider target tracks that:
    - start after source track ends
    - have gap <= max_gap
    - are spatially close enough
    - have not already been used as someone else's successor

    Returns:
        best_target_id or None
    """

    source = summaries[source_id]

    best_target_id = None
    best_score = None

    for target_id, target in summaries.items():
        if target_id == source_id:
            continue

        if target_id in used_as_successor:
            continue

        # Target must start after source ends.
        gap = target["first_frame"] - source["last_frame"]

        if gap <= 0:
            continue

        if gap > max_gap:
            continue

        distance = euclidean_distance(source["last_point"], target["first_point"])

        if distance > max_distance:
            continue

        # Lower score is better.
        # We care mostly about distance, then time gap.
        score = distance + 0.5 * gap

        if best_score is None or score < best_score:
            best_score = score
            best_target_id = target_id

    return best_target_id


def build_merge_mapping(summaries, max_gap, max_distance, min_track_length):
    """
    Build mapping from old track IDs to merged track IDs.

    Strategy:
    - Sort tracks by first frame.
    - For each track, try to connect it to one later track.
    - Use union-find so chains like A -> B -> C become one merged ID.
    """

    parent = {track_id: track_id for track_id in summaries.keys()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            # Keep the earlier-starting track ID as the representative.
            if summaries[root_a]["first_frame"] <= summaries[root_b]["first_frame"]:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    # Process tracks in time order.
    track_ids_sorted = sorted(
        summaries.keys(),
        key=lambda tid: (summaries[tid]["first_frame"], summaries[tid]["last_frame"])
    )

    used_as_successor = set()
    merge_pairs = []

    for source_id in track_ids_sorted:
        source_summary = summaries[source_id]

        # Very tiny tracks are often noise. Do not let them drive merges.
        if source_summary["num_detections"] < min_track_length:
            continue

        target_id = find_best_successor(
            source_id=source_id,
            summaries=summaries,
            used_as_successor=used_as_successor,
            max_gap=max_gap,
            max_distance=max_distance,
        )

        if target_id is not None:
            target_summary = summaries[target_id]

            if target_summary["num_detections"] < min_track_length:
                continue

            union(source_id, target_id)
            used_as_successor.add(target_id)

            merge_pairs.append({
                "source_track": source_id,
                "target_track": target_id,
                "source_first_frame": source_summary["first_frame"],
                "source_last_frame": source_summary["last_frame"],
                "target_first_frame": target_summary["first_frame"],
                "target_last_frame": target_summary["last_frame"],
                "gap": target_summary["first_frame"] - source_summary["last_frame"],
                "distance": euclidean_distance(
                    source_summary["last_point"],
                    target_summary["first_point"]
                )
            })

    # Convert old track IDs to representative merged IDs.
    old_to_root = {track_id: find(track_id) for track_id in summaries.keys()}

    # Re-number merged tracks to compact IDs: 1, 2, 3, ...
    roots_sorted = sorted(
        set(old_to_root.values()),
        key=lambda rid: summaries[rid]["first_frame"]
    )

    root_to_new_id = {
        root: new_id
        for new_id, root in enumerate(roots_sorted, start=1)
    }

    old_to_new = {
        old_id: root_to_new_id[root]
        for old_id, root in old_to_root.items()
    }

    return old_to_new, merge_pairs


def apply_merge_mapping(data, old_to_new):
    """Return a new JSON object with track_id replaced by merged track IDs."""
    merged_data = dict(data)

    old_tracks = data.get("tracks", [])
    new_tracks = []

    for item in old_tracks:
        new_item = dict(item)
        old_id = new_item.get("track_id")

        if old_id in old_to_new:
            new_item["original_track_id"] = old_id
            new_item["track_id"] = old_to_new[old_id]

        new_tracks.append(new_item)

    merged_data["tracks"] = new_tracks

    # Save metadata.
    existing_experiment = merged_data.get("experiment", {})
    merged_data["experiment"] = dict(existing_experiment)
    merged_data["experiment"]["postprocess"] = "merge_track_fragments"

    return merged_data


def print_merge_summary(grouped_before, grouped_after, merge_pairs):
    print("\n=== Merge Track Fragments Summary ===")
    print(f"Original unique track IDs: {len(grouped_before)}")
    print(f"Merged unique track IDs: {len(grouped_after)}")
    print(f"Number of merge pairs applied: {len(merge_pairs)}")

    print("\n--- Merge pairs ---")

    if not merge_pairs:
        print("No merges were applied.")
        return

    for pair in merge_pairs[:30]:
        print(
            f"{pair['source_track']} -> {pair['target_track']} "
            f"(source frames {pair['source_first_frame']}–{pair['source_last_frame']}, "
            f"target frames {pair['target_first_frame']}–{pair['target_last_frame']}, "
            f"gap={pair['gap']} frames, distance={pair['distance']:.1f}px)"
    )

    if len(merge_pairs) > 30:
        print(f"... and {len(merge_pairs) - 30} more")


def main():
    parser = argparse.ArgumentParser(
        description="Merge fragmented ByteTrack IDs using simple temporal/spatial rules."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_anonymous.json",
        help="Path to input tracks JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments/exp04_yolov8s_buffer90_high04/tracks_merged.json",
        help="Path to output merged tracks JSON."
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=90,
        help="Maximum frame gap allowed between two track fragments."
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=250.0,
        help="Maximum pixel distance allowed between track fragments."
    )

    parser.add_argument(
        "--min-track-length",
        type=int,
        default=30,
        help="Ignore tracks shorter than this when proposing merges."
    )

    parser.add_argument(
        "--point-type",
        type=str,
        default="bottom_center",
        choices=["center", "bottom_center"],
        help="Which bbox point to use for spatial matching."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    data = load_json(input_path)
    tracks = data.get("tracks", [])

    grouped_before = group_by_track(tracks)
    summaries = build_track_summaries(
        grouped_before,
        point_type=args.point_type
    )

    old_to_new, merge_pairs = build_merge_mapping(
        summaries=summaries,
        max_gap=args.max_gap,
        max_distance=args.max_distance,
        min_track_length=args.min_track_length
    )

    merged_data = apply_merge_mapping(data, old_to_new)

    grouped_after = group_by_track(merged_data["tracks"])

    save_json(output_path, merged_data)

    print_merge_summary(grouped_before, grouped_after, merge_pairs)
    print(f"\nSaved merged tracks to {output_path}")
    print("\nNext:")
    print("1. Visualize tracks_merged.json.")
    print("2. Analyze tracks_merged.json.")
    print("3. If there are obvious wrong merges, lower --max-distance or --max-gap.")
    print("4. If too few tracks merge, raise --max-distance or --max-gap.")


if __name__ == "__main__":
    main()