import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def bbox_center(bbox):
    """Return the center point (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def load_tracks(path):
    """Load tracks JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_tracks(data, expected_dancers=4, short_track_threshold=30):
    """
    Analyze tracking output.

    This is meant for files like:
    outputs/tracks/tracks_anonymous.json

    It computes:
    - number of frames
    - number of tracked boxes
    - number of unique track IDs
    - average detections per frame
    - frames with fewer than expected dancers
    - track length statistics
    - top longest tracks
    """

    video_info = data.get("video", {})
    tracks = data.get("tracks", [])

    fps = video_info.get("fps", None)
    n_frames = video_info.get("n_frames", None)

    if n_frames is None:
        # Fallback: infer total frames from max frame index
        n_frames = max(item["frame"] for item in tracks) + 1 if tracks else 0

    # Group detections by frame and track ID
    detections_by_frame = defaultdict(list)
    boxes_by_track = defaultdict(list)

    for item in tracks:
        frame = item["frame"]
        track_id = item.get("track_id", None)

        detections_by_frame[frame].append(item)

        if track_id is not None:
            boxes_by_track[track_id].append(item)

    # Frame-level stats
    detections_per_frame = []

    for frame_idx in range(n_frames):
        detections_per_frame.append(len(detections_by_frame.get(frame_idx, [])))

    frame_count_distribution = Counter(detections_per_frame)

    frames_with_fewer_than_expected = [
        frame_idx
        for frame_idx, count in enumerate(detections_per_frame)
        if count < expected_dancers
    ]

    frames_with_more_than_expected = [
        frame_idx
        for frame_idx, count in enumerate(detections_per_frame)
        if count > expected_dancers
    ]

    # Track-level stats
    track_summaries = []

    for track_id, items in boxes_by_track.items():
        items_sorted = sorted(items, key=lambda x: x["frame"])
        frames = [item["frame"] for item in items_sorted]

        first_frame = min(frames)
        last_frame = max(frames)
        num_detections = len(items_sorted)
        duration_frames = last_frame - first_frame + 1

        centers = [bbox_center(item["bbox"]) for item in items_sorted]
        avg_cx = mean([c[0] for c in centers])
        avg_cy = mean([c[1] for c in centers])

        confs = [item.get("conf", 0.0) for item in items_sorted]
        avg_conf = mean(confs) if confs else 0.0

        # Count gaps inside this track
        # Example: frames [1, 2, 3, 10] has one internal gap.
        gaps = []
        for a, b in zip(frames[:-1], frames[1:]):
            if b - a > 1:
                gaps.append(b - a - 1)

        track_summaries.append({
            "track_id": track_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "num_detections": num_detections,
            "duration_frames": duration_frames,
            "duration_seconds": duration_frames / fps if fps else None,
            "avg_conf": avg_conf,
            "avg_center_x": avg_cx,
            "avg_center_y": avg_cy,
            "num_internal_gaps": len(gaps),
            "max_internal_gap": max(gaps) if gaps else 0,
        })

    track_summaries.sort(key=lambda x: x["num_detections"], reverse=True)

    track_lengths = [t["num_detections"] for t in track_summaries]

    short_tracks = [
        t for t in track_summaries
        if t["num_detections"] <= short_track_threshold
    ]

    summary = {
        "video": video_info,
        "expected_dancers": expected_dancers,
        "total_frames": n_frames,
        "total_tracked_boxes": len(tracks),
        "unique_track_ids": len(boxes_by_track),
        "average_detections_per_frame": mean(detections_per_frame) if detections_per_frame else 0,
        "median_detections_per_frame": median(detections_per_frame) if detections_per_frame else 0,
        "frame_count_distribution": dict(sorted(frame_count_distribution.items())),
        "num_frames_with_fewer_than_expected_dancers": len(frames_with_fewer_than_expected),
        "percent_frames_with_fewer_than_expected_dancers": (
            100 * len(frames_with_fewer_than_expected) / n_frames if n_frames else 0
        ),
        "num_frames_with_more_than_expected_dancers": len(frames_with_more_than_expected),
        "percent_frames_with_more_than_expected_dancers": (
            100 * len(frames_with_more_than_expected) / n_frames if n_frames else 0
        ),
        "track_length_min": min(track_lengths) if track_lengths else 0,
        "track_length_max": max(track_lengths) if track_lengths else 0,
        "track_length_mean": mean(track_lengths) if track_lengths else 0,
        "track_length_median": median(track_lengths) if track_lengths else 0,
        "short_track_threshold": short_track_threshold,
        "num_short_tracks": len(short_tracks),
        "top_10_longest_tracks": track_summaries[:10],
    }

    return summary, track_summaries, frames_with_fewer_than_expected


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_track_csv(path, track_summaries):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not track_summaries:
        return

    fieldnames = list(track_summaries[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(track_summaries)


def save_missing_frame_csv(path, missing_frames, fps=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "timestamp_seconds"])
        writer.writeheader()

        for frame in missing_frames:
            writer.writerow({
                "frame": frame,
                "timestamp_seconds": frame / fps if fps else None
            })


def print_summary(summary):
    print("\n=== Tracking Analysis Summary ===")

    video = summary.get("video", {})
    print(f"Video path: {video.get('path', 'unknown')}")
    print(f"FPS: {video.get('fps', 'unknown')}")
    print(f"Resolution: {video.get('width', 'unknown')} x {video.get('height', 'unknown')}")
    print(f"Total frames: {summary['total_frames']}")

    print("\n--- Detection / Tracking Counts ---")
    print(f"Total tracked boxes: {summary['total_tracked_boxes']}")
    print(f"Unique track IDs: {summary['unique_track_ids']}")
    print(f"Expected dancers: {summary['expected_dancers']}")
    print(f"Average detections per frame: {summary['average_detections_per_frame']:.2f}")
    print(f"Median detections per frame: {summary['median_detections_per_frame']}")

    print("\n--- Detections Per Frame Distribution ---")
    for count, num_frames in summary["frame_count_distribution"].items():
        print(f"Frames with {count} detections: {num_frames}")

    print("\n--- Missing / Extra Dancer Frames ---")
    print(
        "Frames with fewer than expected dancers: "
        f"{summary['num_frames_with_fewer_than_expected_dancers']} "
        f"({summary['percent_frames_with_fewer_than_expected_dancers']:.2f}%)"
    )
    print(
        "Frames with more than expected dancers: "
        f"{summary['num_frames_with_more_than_expected_dancers']} "
        f"({summary['percent_frames_with_more_than_expected_dancers']:.2f}%)"
    )

    print("\n--- Track Length Stats ---")
    print(f"Shortest track length: {summary['track_length_min']} detections")
    print(f"Longest track length: {summary['track_length_max']} detections")
    print(f"Mean track length: {summary['track_length_mean']:.2f} detections")
    print(f"Median track length: {summary['track_length_median']} detections")
    print(
        f"Short tracks <= {summary['short_track_threshold']} detections: "
        f"{summary['num_short_tracks']}"
    )

    print("\n--- Top 10 Longest Tracks ---")
    for track in summary["top_10_longest_tracks"]:
        duration = track["duration_seconds"]
        duration_str = f"{duration:.2f}s" if duration is not None else "unknown"

        print(
            f"Track {track['track_id']}: "
            f"{track['num_detections']} detections, "
            f"frames {track['first_frame']}–{track['last_frame']}, "
            f"duration {duration_str}, "
            f"avg conf {track['avg_conf']:.3f}"
        )

    print("\nInterpretation:")
    print("- If unique_track_ids is much larger than the number of dancers, tracking is fragmented.")
    print("- If many frames have fewer than expected dancers, detection is missing people.")
    print("- If many tracks are very short, the tracker is repeatedly creating new IDs.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze YOLO/ByteTrack tracking output."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/tracks/tracks_anonymous.json",
        help="Path to tracking JSON file."
    )

    parser.add_argument(
        "--expected-dancers",
        type=int,
        default=4,
        help="Expected number of dancers in the video."
    )

    parser.add_argument(
        "--short-track-threshold",
        type=int,
        default=30,
        help="Tracks with this many detections or fewer are counted as short tracks."
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default="outputs/tracks/track_analysis_summary.json",
        help="Where to save summary JSON."
    )

    parser.add_argument(
        "--track-csv-output",
        type=str,
        default="outputs/tracks/track_lengths.csv",
        help="Where to save per-track CSV."
    )

    parser.add_argument(
        "--missing-frames-output",
        type=str,
        default="outputs/tracks/missing_dancer_frames.csv",
        help="Where to save frames with fewer than expected dancers."
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    data = load_tracks(input_path)

    summary, track_summaries, missing_frames = analyze_tracks(
        data,
        expected_dancers=args.expected_dancers,
        short_track_threshold=args.short_track_threshold
    )

    print_summary(summary)

    save_json(Path(args.summary_output), summary)
    save_track_csv(Path(args.track_csv_output), track_summaries)

    fps = summary.get("video", {}).get("fps", None)
    save_missing_frame_csv(Path(args.missing_frames_output), missing_frames, fps=fps)

    print("\nSaved files:")
    print(f"- Summary JSON: {args.summary_output}")
    print(f"- Track CSV: {args.track_csv_output}")
    print(f"- Missing dancer frame CSV: {args.missing_frames_output}")


if __name__ == "__main__":
    main()