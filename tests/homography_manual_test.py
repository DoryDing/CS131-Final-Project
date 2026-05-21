"""
homography_manual_test.py
--------------------------
Sanity check for the homography pipeline.
Runs the full pipeline without needing the interactive calibration window:
  1. Generate synthetic tracks.json
  2. Create a fake calibration.json using hardcoded floor corners
  3. Run top_down_positions() and print sample output
  4. Plot top-down dancer positions for frame 0 and save to outputs/
"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# make sure Python can find src/ regardless of where this script is run from
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.formations.homography import save_calibration, top_down_positions
from src.formations.generate_synthetic import generate_synthetic_tracks

TRACKS_PATH      = str(project_root / "data" / "processed" / "synthetic_tracks.json")
CALIBRATION_PATH = str(project_root / "data" / "processed" / "test_calibration.json")
OUTPUT_PLOT_PATH = str(project_root / "outputs" / "sanity_check_frame0.png")


def main():
    print("\n=== Sanity Check: Homography Pipeline ===\n")

    # ------------------------------------------------------------------
    # Step 1 — generate synthetic tracks
    # ------------------------------------------------------------------
    print("Step 1: Generating synthetic tracks...")
    generate_synthetic_tracks(TRACKS_PATH)

    # ------------------------------------------------------------------
    # Step 2 — create a fake calibration without the interactive window
    # 4 hardcoded floor corners that cover the synthetic stage area
    # order: top-left, top-right, bottom-right, bottom-left (clockwise)
    # ------------------------------------------------------------------
    print("\nStep 2: Creating fake calibration (no video window needed)...")

    src_points = [
        [100,  200],   # top-left
        [1820, 200],   # top-right
        [1820, 950],   # bottom-right
        [100,  950],   # bottom-left
    ]

    dst_points = np.float32([
        [0,   0  ],    # top-left     → stage (0, 0)
        [100, 0  ],    # top-right    → stage (100, 0)
        [100, 100],    # bottom-right → stage (100, 100)
        [0,   100],    # bottom-left  → stage (0, 100)
    ])

    H = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)

    save_calibration(
        H=H,
        stage_w=100,
        stage_h=100,
        src_points=src_points,
        output_path=CALIBRATION_PATH
    )

    # ------------------------------------------------------------------
    # Step 3 — run top_down_positions() and print sample output
    # ------------------------------------------------------------------
    print("\nStep 3: Running top_down_positions()...")
    positions = top_down_positions(TRACKS_PATH, CALIBRATION_PATH)

    print(f"  Total entries returned: {len(positions)}")
    print("  First 8 entries:")
    for p in positions[:8]:
        print(f"    frame={p['frame']:3d}  member={p['member_id']}  "
              f"x={p['x']:6.2f}  y={p['y']:6.2f}")

    # quick sanity check — all x and y values should be between 0 and 100
    out_of_bounds = [
        p for p in positions
        if not (0 <= p["x"] <= 100 and 0 <= p["y"] <= 100)
    ]
    if out_of_bounds:
        print(f"\n  WARNING: {len(out_of_bounds)} entries are outside stage bounds (0-100).")
        print("  This may indicate a mismatch between synthetic dancer positions and calibration corners.")
    else:
        print("\n  All positions are within stage bounds (0-100). Calibration looks good.")

    # ------------------------------------------------------------------
    # Step 4 — plot top-down positions for frame 0
    # ------------------------------------------------------------------
    print("\nStep 4: Plotting top-down positions for frame 0...")

    frame0 = [p for p in positions if p["frame"] == 0]
    colors = {1: "red", 2: "blue", 3: "green", 4: "orange"}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Stage X (0 = left, 100 = right)")
    ax.set_ylabel("Stage Y (0 = top, 100 = bottom)")
    ax.set_title("Top-down view — Frame 0 (synthetic data)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # y=0 at top, matching image coordinate convention

    # draw stage boundary box
    ax.add_patch(mpatches.Rectangle(
        (0, 0), 100, 100,
        linewidth=2, edgecolor="black", facecolor="lightyellow", zorder=1
    ))

    # plot each dancer as a colored dot with label
    for p in frame0:
        ax.scatter(p["x"], p["y"], s=250, c=colors[p["member_id"]], zorder=5)
        ax.annotate(
            f"  #{p['member_id']}",
            (p["x"], p["y"]),
            fontsize=12,
            fontweight="bold"
        )

    # legend
    legend_handles = [
        mpatches.Patch(color=c, label=f"Member {m}")
        for m, c in colors.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()

    Path(OUTPUT_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"  Plot saved to outputs/sanity_check_frame0.png")
    plt.show()

    print("\n=== Sanity check complete ===")


if __name__ == "__main__":
    main()
