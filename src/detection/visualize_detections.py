import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
DETECTIONS_PATH = "outputs/tracks/detections_only.json"
OUTPUT_DIR = "outputs/debug/detection_frames"


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DETECTIONS_PATH, "r") as f:
        data = json.load(f)

    detections_by_frame = {}

    for item in data["tracks"]:
        frame = item["frame"]
        detections_by_frame.setdefault(frame, []).append(item)

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections_by_frame:
            for det in detections_by_frame[frame_idx]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                conf = det["conf"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        frame_idx += 1

    cap.release()
    print(f"Saved debug frames to {output_dir}")


if __name__ == "__main__":
    main()