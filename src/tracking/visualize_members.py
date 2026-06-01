import argparse
import cv2
import json
from pathlib import Path


VIDEO_PATH = "data/raw/test_playing_with_fire_short.mp4"

DEFAULT_TRACKS_PATH = (
    "outputs/experiments/exp04_yolov8s_buffer90_high04/"
    "tracks_member_labeled.json"
)

DEFAULT_OUTPUT_VIDEO_PATH = (
    "outputs/experiments/exp04_yolov8s_buffer90_high04/"
    "member_debug.mp4"
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_member_label(item, member_legend):
    """
    Return display label for one detection.

    Example:
    member_id = 1
    member_legend["1"] = "red_hair"

    Output:
    "member 1: red_hair"
    """
    member_id = item.get("member_id", None)

    if member_id is None:
        return "member ?"

    member_key = str(member_id)
    member_name = member_legend.get(member_key, None)

    if member_name is None:
        return f"member {member_id}"

    return f"member {member_id}: {member_name}"


def get_member_color(member_id):
    """
    Return a consistent BGR color for each member.

    OpenCV uses BGR, not RGB.
    These colors are just for visualization.
    """
    colors = {
        1: (0, 0, 255),       # red
        2: (255, 255, 255),   # white
        3: (40, 40, 40),      # dark gray / black
        4: (255, 0, 0),       # blue
        None: (0, 255, 255)   # yellow for unknown
    }

    return colors.get(member_id, (0, 255, 255))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize manually assigned member IDs on video."
    )

    parser.add_argument(
        "--tracks",
        type=str,
        default=DEFAULT_TRACKS_PATH,
        help="Path to member-labeled tracks JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_VIDEO_PATH,
        help="Path to save member debug video."
    )

    parser.add_argument(
        "--video",
        type=str,
        default=VIDEO_PATH,
        help="Path to input video."
    )

    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    output_video_path = Path(args.output)
    video_path = Path(args.video)

    if not tracks_path.exists():
        raise FileNotFoundError(f"Could not find tracks file: {tracks_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video file: {video_path}")

    data = load_json(tracks_path)

    member_legend = data.get("member_legend", {})

    tracks_by_frame = {}

    for item in data["tracks"]:
        frame = item["frame"]
        tracks_by_frame.setdefault(frame, []).append(item)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

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
                member_id = item.get("member_id", None)

                color = get_member_color(member_id)
                label = get_member_label(item, member_legend)

                is_interpolated = item.get("is_interpolated", False)
                thickness = 1 if is_interpolated else 3

                if is_interpolated:
                    label = label + " interp"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Text background for readability
                text_x = x1
                text_y = max(y1 - 10, 20)

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Read member-labeled tracks from {tracks_path}")
    print(f"Saved member debug video to {output_video_path}")


if __name__ == "__main__":
    main()