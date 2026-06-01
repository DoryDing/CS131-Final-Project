import argparse
import json
from pathlib import Path

from src.formations.homography import top_down_positions
from src.formations.segmentation import run_segmentation
from src.formations.summarization import summarize_all

from src.output.pdf_generate import build_pdf


def main():
    parser = argparse.ArgumentParser(description="K-pop formation sheet generator")
    parser.add_argument("--tracks",  required=True,  help="Path to tracks_member_interpolated JSON")
    parser.add_argument("--calib",   required=True,  help="Path to calibration JSON")
    parser.add_argument("--output",  default="outputs/formation_sheet.pdf", help="Output PDF path")
    parser.add_argument("--title",   default="Formation Sheet", help="Title shown on PDF cover")
    parser.add_argument("--threshold",      type=float, default=2.0,  help="Motion energy threshold for stable formation")
    parser.add_argument("--min-frames",     type=int,   default=15,   help="Minimum stable frames to count as a formation")
    args = parser.parse_args()

    # load tracks for fps + member_legend
    with open(args.tracks) as f:
        tracks_data = json.load(f)

    fps = tracks_data["video"]["fps"]
    member_legend = tracks_data.get("member_legend")  # e.g. {"1": "red_hair", ...}

    print(f"Video: {tracks_data['video']['path']}")
    print(f"FPS: {fps}  |  Total frames: {tracks_data['video']['n_frames']}")
    if member_legend:
        print(f"Members: {member_legend}")

    # step 1: homography → top-down positions
    print("\n[1/3] Computing top-down positions...")
    top_down = top_down_positions(args.tracks, args.calib)
    print(f"      {len(top_down)} position entries")

    # step 2: segmentation → formation segments
    print("[2/3] Segmenting formations...")
    segments = run_segmentation(
        top_down,
        threshold=args.threshold,
        min_stable_frames=args.min_frames,
        fps=fps,
    )
    print(f"      {len(segments)} formations found")
    for seg in segments:
        print(f"      Formation {seg['formation_id']}: "
              f"frames {seg['start_frame']}–{seg['end_frame']}  "
              f"({seg.get('start_time_s', '?')}s – {seg.get('end_time_s', '?')}s)")

    if not segments:
        print("No stable formations found. Try lowering --threshold or --min-frames.")
        return

    # step 3: summarization → canonical positions per formation
    print("[3/3] Summarizing formations...")
    formations = summarize_all(top_down, segments)

    # generate PDF
    print(f"\nBuilding PDF → {args.output}")
    build_pdf(
        formations=formations,
        output_path=args.output,
        title=args.title,
        member_legend=member_legend,
    )
    print("Done.")


if __name__ == "__main__":
    main()