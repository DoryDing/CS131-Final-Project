# Task Division — CS131 Final Project

Two partners, similar skill levels, 3–4 week timeline, split by module independence.

The repo already has four module folders (`src/detection`, `src/tracking`, `src/formations`, `src/output`). The cleanest module-independent split is to draw the line down the middle of the pipeline: one person owns "who is where in each frame," the other owns "what is the formation and how do we show it." These two halves talk to each other through a single well-defined data file, so both people can develop in parallel from day one.

## The split

### Partner A — Perception (Detection + Tracking)

Owns `src/detection/` and `src/tracking/`. The goal is to turn a raw video into a clean, per-frame record of where each dancer is and which member they are.

Responsibilities:
- Video I/O: reading frames, handling framerate, writing intermediate artifacts.
- Person detection per frame (YOLOv8 or similar off-the-shelf model; this is integration, not training).
- Multi-object tracking across frames (ByteTrack or DeepSORT) so each dancer gets a persistent track ID.
- Member identification: mapping anonymous track IDs to actual member names. This is the hardest part on this side — options include manual seeding on a reference frame, appearance-based re-ID (outfit color histograms / embeddings), or face crops against a small reference gallery. Plan to ship the simplest version that works and document the limitation.
- Handling re-entry and ID swaps (when a tracker loses someone and reassigns a new ID).
- Producing the handoff artifact: a `tracks.json` (or parquet) file with one row per (frame, member) — see interface below.

### Partner B — Spatial analysis + Output (Formations + PDF)

Owns `src/formations/` and `src/output/`. The goal is to turn the per-frame tracks into a structured formation sheet.

Responsibilities:
- Homography / perspective transform: mapping image-space bounding-box footpoints to a normalized top-down stage coordinate system. Manual 4-point calibration on the first frame is fine for v1.
- Formation segmentation in time: deciding *when* the group is in a stable formation vs. transitioning. Approach: compute per-frame "motion energy" (e.g., mean dancer velocity in top-down space) and threshold; stable runs become formations.
- Formation summarization: for each stable segment, compute a single canonical (x, y) per member (e.g., median position over the segment).
- Top-down diagram rendering: a clean 2D stage with labeled dots per member.
- PDF assembly: structured formation sheet with timestamps, diagrams, and member legend (uses the `pdf` skill / reportlab).

### Shared (both partners, paired)

These are not module-divisible and should be done together to avoid mismatched assumptions:

- **The interface contract** (week 1, day 1–2). Lock the schema of `tracks.json` before either side starts building. See draft below.
- **A test video set** (3–5 short K-pop dance practice clips with known ground-truth formations) — used for evaluation by both sides.
- **Integration** at the week 2/3 boundary.
- **Written report** (split sections, see below).
- **Poster / presentation** (split sections, see below).

## The handoff interface

This is the single most important thing to agree on in week 1. Suggested schema for `tracks.json`:

```json
{
  "video": {"path": "data/raw/song_x.mp4", "fps": 30, "width": 1920, "height": 1080, "n_frames": 5400},
  "members": ["Jisoo", "Jennie", "Rosé", "Lisa"],
  "tracks": [
    {"frame": 0, "member": "Jisoo", "bbox": [x1, y1, x2, y2], "conf": 0.92, "track_id": 3},
    {"frame": 0, "member": "Jennie", "bbox": [...], "conf": 0.88, "track_id": 1},
    ...
  ]
}
```

Partner B should be able to build and test the formation/PDF pipeline end-to-end against a hand-written `tracks.json` while Partner A is still building the real detector. That's the whole point of the independent-modules split.

## Week-by-week plan (4 weeks)

### Week 1 — Foundations (paired start, then split)

Together (days 1–2):
- Agree on the `tracks.json` schema above.
- Pick 3–5 test videos and clip them to 30–60 seconds each.
- Hand-label 2 of them with rough ground-truth formations (timestamps + member positions on a top-down sketch) — this is your eval set.
- Partner B writes a tiny synthetic `tracks.json` generator so they can start without waiting on Partner A.

Then in parallel (days 3–7):
- **A:** get YOLO detection + ByteTrack working end-to-end on one test video, output a `tracks.json` with anonymous track IDs (no member names yet).
- **B:** implement the homography calibration tool (click 4 points on first frame) and a `top_down_positions()` function that consumes `tracks.json` and outputs per-frame (x, y) in stage coords.

### Week 2 — Core implementation (parallel)

- **A:** member identification — start with manual seeding on frame 0 (user clicks each member, you propagate via track_id), then add appearance-based re-ID for re-entry. Handle the obvious ID-swap cases.
- **B:** formation segmentation (motion-energy thresholding), formation summarization (median position per stable run), and a first-pass top-down diagram renderer (matplotlib is fine).

Mid-week 2: 30-minute sync. A delivers a real `tracks.json` for one full test video. B confirms it plugs into the pipeline.

### Week 3 — Integration + iteration

- Both: run the full pipeline on all test videos. Find what breaks.
- **A:** fix the worst tracking failures (long occlusions, members crossing).
- **B:** fix the worst formation-segmentation failures (tune the motion threshold; handle slow transitions); polish the PDF layout.
- End of week 3: end-to-end PDF for at least 2 videos, even if rough.

### Week 4 — Polish + deliverables

- Mon–Tue: final tuning + run on all eval videos; compute quantitative results (formation-detection precision/recall against your hand labels).
- Wed–Thu: written report (split, see below).
- Fri: poster + practice presentation.

## Splitting the report and poster

**Report sections:**
- A writes: Detection, Tracking, Member identification, related work on multi-object tracking.
- B writes: Homography, Formation segmentation, PDF generation, related work on spatial/formation analysis.
- Together: Intro, Problem statement, Evaluation methodology, Results, Limitations, Conclusion.

**Poster:** same split — A owns the left half (perception), B owns the right half (formations + output), shared center for the example formation sheet. One person leads the talk-through, the other answers questions.

## Risks to watch

- **Member identification is the riskiest piece** on A's side. Have a manual-fallback path so the rest of the pipeline isn't blocked by it.
- **Homography quality** is the riskiest on B's side — bad calibration ruins the top-down view. Build a quick visual sanity-check (overlay the top-down points back onto the source frame).
- **Don't over-engineer either half.** A working rough end-to-end pipeline by end of week 3 is worth more than a polished half.

## What to do right now

1. Decide who takes A vs. B (both halves are roughly equal in difficulty and originality).
2. Schedule a 1-hour kickoff to lock the `tracks.json` schema and pick the test videos.
3. Partner B can start the synthetic-data generator the same day so neither side blocks the other.
