import cv2
import json
import numpy as np
from pathlib import Path

class HomographyCalibrator:
    """
    Interactive tool for computing a homography matrix by clicking 4 floor corners manually on the first frame of
    a video. The 4 points must be clicked in clockwise order starting from top-left.

    The resulting homography matrix H is saved to a JSON calibration file.
    """

    def __init__(self, video_path, stage_width=100, stage_height=100):
        self.video_path = video_path
        self.stage_width = stage_width
        self.stage_height = stage_height

        self.clicked_points = []  # list of (u, v) image-space points the user clicks
        self.frame = None  # the first frame image, stored so _mouse_click can draw on it
        self.H = None  # the computed 3x3 homography matrix

    def _mouse_click(self, event, x, y, flags, param):
        """
            Called automatically by OpenCV on every mouse event in the window.
            Only responds to left button clicks. Collects up to 4 points.
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.clicked_points) >= 4:
            return

        self.clicked_points.append((x, y))
        point_number = len(self.clicked_points)

        cv2.circle(self.frame, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
        cv2.putText(
            self.frame,
            str(point_number),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2
        )

        cv2.imshow("Calibration", self.frame)
        print(f"  Point {point_number} recorded: ({x}, {y})")

    def run(self, calibration_output_path="data/processed/calibration.json"):
        """
        Opens the first frame of the video in an interactive window. Waits for the user to click 4 floor corners,
        then computes and saves H.

        Args:
            calibration_output_path: where to write the calibration JSON file
        """

        #open the first frame
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError("Could not read first frame from video.")

        self.frame = frame.copy()

        # --- print instructions for users---
        print("\n=== Homography Calibration ===")
        print("Click exactly 4 floor corners IN THIS ORDER:")
        print("  1. Top-left")
        print("  2. Top-right")
        print("  3. Bottom-right")
        print("  4. Bottom-left")
        print("Press 'q' to quit without saving.\n")

        # --- open window and attach mouse ---
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._mouse_click)
        cv2.imshow("Calibration", self.frame)

        # --- wait loop ---
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("Calibration cancelled.")
                break

            if len(self.clicked_points) == 4:
                self._compute_homography(calibration_output_path)
                break

        cv2.destroyAllWindows()

    def _compute_homography(self, calibration_output_path):
        """
        Computes H from the 4 clicked points and saves it.

        """
        source = np.float32(self.clicked_points)
        destination = np.float32([
            [0, 0],  # top-left
            [self.stage_width, 0],  # top-right
            [self.stage_width, self.stage_height],  # bottom-right
            [0, self.stage_height],  # bottom-left
        ])

        self.H = cv2.getPerspectiveTransform(source, destination)
        print("\nHomography matrix computed successfully.")

        save_calibration(
            H=self.H,
            stage_w=self.stage_width,
            stage_h=self.stage_height,
            src_points=self.clicked_points,
            output_path=calibration_output_path
        )

def save_calibration(H, stage_w, stage_h, src_points, output_path):
    """
    Saves the homography matrix and calibration metadata to a JSON file.

    Args:
        H: (3x3) numpy float32 array that represents the homography matrix
        stage_w: stage width in stage coordinate units
        stage_h: stage height in stage coordinate units
        src_points: list of 4 (u, v) image-space points that the user clicked
        output_path: file path to write the JSON to
    """
    data = {
        "homography_matrix": H.tolist(),
        "stage_width": stage_w,
        "stage_height": stage_h,
        "src_points": src_points
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {output_path}")

def load_calibration(calibration_path):
    """
    Loads a saved calibration JSON and returns the homography matrix and stage dimensions.

    Args:
        calibration_path: path to the calibration JSON file

    Returns:
        H: (3x3) numpy float32 array
        stage_w: stage width
        stage_h: stage height
    """
    with open(calibration_path, "r") as f:
        data = json.load(f)

    H = np.array(data["homography_matrix"], dtype=np.float32)
    stage_w = data["stage_width"]
    stage_h = data["stage_height"]

    return H, stage_w, stage_h

def top_down_positions(tracks_path, calibration_path):
    """
    Converts per-frame bounding box positions from image space to top-down stage coordinates using a
    precomputed homography matrix.

    For each track entry, the footpoint (center of the bottom edge of the bounding box) is computed and transformed
    using H. The footpoint represents where the dancer's feet touch the floor.

    Args:
        tracks_path: path to the tracks.json file
        calibration_path: path to the calibration.json file produced by HomographyCalibrator

    Returns:
        A list of dicts, one per track entry.
    """

    with open(tracks_path, "r") as f:
        tracks_data = json.load(f)

    H, stage_w, stage_h = load_calibration(calibration_path)

    #compute footpoints and transform
    results = []
    for entry in tracks_data["tracks"]:
        x1, y1, x2, y2 = entry["bbox"]

        # footpoint is calculated as the center of the bottom edge of the bounding box
        foot_u = (x1 + x2) / 2.0
        foot_v = float(y2)

        point = np.array([[[foot_u, foot_v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)

        stage_x = float(transformed[0][0][0])
        stage_y = float(transformed[0][0][1])

        results.append({
            "frame":     entry["frame"],
            "member_id": entry["member_id"],
            "x":         round(stage_x, 3),
            "y":         round(stage_y, 3),
        })

    return results

if __name__ == "__main__":
    # resolve path relative to this file's location so it works from any directory
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "data" / "raw" / "playing_with_fire_short_test.mov"

    calibrator = HomographyCalibrator(video_path=str(video_path))
    calibrator.run(
        calibration_output_path=str(project_root / "data" / "processed" / "calibration.json")
    )


