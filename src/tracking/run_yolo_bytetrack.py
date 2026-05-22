from ultralytics import YOLO
import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
OUTPUT_PATH = "outputs/tracks/tracks_anonymous.json"


def main():
    video_path = Path(VIDEO_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    tracks = []

    results = model.track(
        source=str(video_path),
        classes=[0],          # person only
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False
    )

    for frame_idx, result in enumerate(results):
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            track_id = int(box.id[0])

            tracks.append({
                "frame": frame_idx,
                "member_id": None,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

    output = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames
        },
        "members": [],
        "tracks": tracks
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved anonymous tracks to {output_path}")
    print(f"Number of tracked boxes: {len(tracks)}")


if __name__ == "__main__":
    main()