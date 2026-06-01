import argparse
import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
TRACKS_PATH = "outputs/tracks/tracks_anonymous.json"
OUTPUT_VIDEO_PATH = "outputs/debug/tracking_debug.mp4"


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking IDs on video.")

    parser.add_argument(
        "--tracks",
        type=str,
        default=TRACKS_PATH,
        help="Path to tracks JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_VIDEO_PATH,
        help="Path to save tracking debug video."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name. If provided, reads/writes under outputs/experiments/<experiment-name>/."
    )

    args = parser.parse_args()

    # Preserve original behavior unless --experiment-name is provided.
    if args.experiment_name is not None:
        tracks_path = Path("outputs") / "experiments" / args.experiment_name / "tracks_anonymous.json"
        output_video_path = Path("outputs") / "experiments" / args.experiment_name / "tracking_debug.mp4"
    else:
        tracks_path = Path(args.tracks)
        output_video_path = Path(args.output)

    with open(tracks_path, "r") as f:
        data = json.load(f)

    tracks_by_frame = {}

    for item in data["tracks"]:
        frame = item["frame"]
        tracks_by_frame.setdefault(frame, []).append(item)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in tracks_by_frame:
            for item in tracks_by_frame[frame_idx]:
                x1, y1, x2, y2 = map(int, item["bbox"])
                track_id = item["track_id"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"track {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Read tracks from {tracks_path}")
    print(f"Saved tracking debug video to {output_video_path}")


if __name__ == "__main__":
    main()