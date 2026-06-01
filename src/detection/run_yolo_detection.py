from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from tqdm import tqdm

# for convenience: /Users/haorangao/Desktop/Stanford/CS 131/CS131-Final-Project

VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"
OUTPUT_PATH = "outputs/tracks/detections_only.json"

SAMPLE_EVERY_N_FRAMES = 5  # start simple: process every 5 frames


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

    all_tracks = []

    frame_idx = 0

    pbar = tqdm(total=n_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
            # YOLO class 0 = person
            results = model(frame, classes=[0], verbose=False)

            for result in results:
                boxes = result.boxes

                for det_id, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    all_tracks.append({
                        "frame": frame_idx,
                        "member_id": None,
                        "track_id": None,
                        "det_id": det_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    output = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": n_frames
        },
        "members": [],
        "tracks": all_tracks
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved detections to {output_path}")
    print(f"Number of detections: {len(all_tracks)}")


if __name__ == "__main__":
    main()