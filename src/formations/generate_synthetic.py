import json
import math
import random
import numpy as np
from pathlib import Path

VIDEO_META = {
    "path": "data/raw/playing_with_fire_short_test.mov",
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "n_frames": 300
}

N_MEMBERS = 4
BBOX_W = 160
BBOX_H = 320

def get_dancer_positions(center_x, center_y, n_frames, amplitude=120, phase=0.0):
    """
        Return where the dancer is standing at each frame

        Args:
            center_x, center_y: dancer's starting position on the screen
            n_frames: frames extracted from the video
            amplitude: how wide dancer's movements are
            phase: offset

        Returns:
            a list of (x,y) positions
    """
    positions = []
    for f in range(n_frames):
        # convert the frame number to a value between 0.0 and 1.0 so the math and calculations are
        # not depended on the number of frames
        converted_frame_number = f / n_frames

        #the cosine and sine waves below simulates the smooth moves of dancers in dance videos so that they dont
        #"teleport" between positions in frames

        #dancer's x position oscillates left and right using a sine wave
        x = center_x + amplitude * math.sin(2 * math.pi * converted_frame_number + phase)

        #dancer's y position oscillates left and right using a cosine wave
        #the 0.5 makes the dancer moves less vertically, which is common in KPOP dances
        y = center_y + amplitude * 0.5 * math.cos(2 * math.pi * converted_frame_number + phase)

        #mimics the noise in real world dance  move
        x += random.uniform(-5, 5)
        y += random.uniform(-5, 5)

        positions.append((x, y))
    return positions

def center_to_bbox(cx, cy, w=BBOX_W, h=BBOX_H):
    """
    Converts center-positions format of a box to corner-positions format of a box

    Args:
        cx, cy: the center position of the box
        w,h = width and height of box (constant)

    Returns:
        a list that contains two corners of the box (top left (x1, y1) and bottom right (x2, y2))

    """
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    x1 = max(0, x1) #x1 never goes below 0
    y1 = max(0, y1) #y1 never goes above the top edge
    x2 = min(VIDEO_META["width"],  x2)
    y2 = min(VIDEO_META["height"], y2)

    return [x1, y1, x2, y2]

def generate_synthetic_tracks(output_path):
    """
        Generate a synthetic tracks.json file simulating 4 dancers moving
        across a 1920x1080 stage over n_frames frames.

        Args:
            output_path: a string that is the file path to write the JSON output to

        Output: a JSON file
    """

    n_frames = VIDEO_META["n_frames"]

    start_positions = [
        (400, 540),
        (730, 520),
        (1060, 520),
        (1390, 540),
    ] #placeholders for the starting positions for now, wait for the real starting position from partner

    dancers_path = {}
    #get dancers' positions at different frames
    for member_id, (cx, cy) in enumerate(start_positions, start=1):
        phase = (member_id - 1) * (math.pi / 2) #ensure the dancers are 90 degress apart in their movements
        dancers_path[member_id] = get_dancer_positions(cx, cy, n_frames, phase=phase)

    temp_output = []
    for frame in range(n_frames):
        for member_id in range(1, N_MEMBERS + 1):
            cx, cy = dancers_path[member_id][frame] #get dancers' center position at the curernt frame
            bbox = center_to_bbox(cx, cy) #convert the center-position format to corner-positions format

            temp_output.append({
                "frame": frame,
                "member_id": member_id,
                "track_id": member_id,
                "bbox": bbox,
            })
    output = {
        "video":   VIDEO_META,
        "members": list(range(1, N_MEMBERS + 1)),
        "tracks":  temp_output
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(temp_output)} track entries to {output_path}")

if __name__ == "__main__":
    output_path = "data/processed/synthetic_tracks.json"
    generate_synthetic_tracks(output_path)

