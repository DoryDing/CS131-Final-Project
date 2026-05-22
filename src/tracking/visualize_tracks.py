import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
TRACKS_PATH = "outputs/tracks/tracks_anonymous.json"
OUTPUT_VIDEO_PATH = "outputs/debug/tracking_debug.mp4"


def main():
    with open(TRACKS_PATH, "r") as f:
        data = json.load(f)

    tracks_by_frame = {}

    for item in data["tracks"]:
        frame = item["frame"]
        tracks_by_frame.setdefault(frame, []).append(item)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

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

    print(f"Saved tracking debug video to {output_path}")


if __name__ == "__main__":
    main()