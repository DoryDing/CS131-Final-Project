from ultralytics import YOLO
import argparse
import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
OUTPUT_PATH = "outputs/tracks/tracks_anonymous.json"


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 + ByteTrack on a video.")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights, e.g. yolov8n.pt or yolov8s.pt."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Path to save tracking JSON."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name. If provided, saves output under outputs/experiments/<experiment-name>/."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Optional YOLO confidence threshold."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional YOLO inference image size."
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config path, e.g. bytetrack.yaml or configs/bytetrack_buffer90.yaml."
    )

    args = parser.parse_args()

    video_path = Path(VIDEO_PATH)

    # Preserve original behavior unless --experiment-name is provided.
    if args.experiment_name is not None:
        output_path = Path("outputs") / "experiments" / args.experiment_name / "tracks_anonymous.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    tracks = []

    track_kwargs = {
        "source": str(video_path),
        "classes": [0],          # person only
        "tracker": args.tracker,
        "persist": True,
        "stream": True,
        "verbose": False
    }

    # These are optional. If not provided, original model.track(...) behavior is preserved.
    if args.conf is not None:
        track_kwargs["conf"] = args.conf

    if args.imgsz is not None:
        track_kwargs["imgsz"] = args.imgsz

    results = model.track(**track_kwargs)

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
        "experiment": {
            "name": args.experiment_name,
            "model": args.model,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "tracker": args.tracker
        },
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