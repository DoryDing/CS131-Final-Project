# K-pop Dance Formation Detection

A computer vision pipeline that automatically extracts dance formations from K-pop dance practice videos and generates a structured PDF formation sheet.

CS131 Final Project — Stanford University, Spring 2026.

## Problem

K-pop choreographies rely heavily on dynamic group formations and spatial storytelling. Dance cover groups currently study practice videos frame-by-frame to recreate these formations, or manually annotate them using tools like ArrangeUs. This project automates that process.

## Pipeline Overview

1. **Person detection** — locate dancers in each frame
2. **Multi-person tracking** — assign consistent IDs to each dancer across frames
3. **Formation analysis** — detect when the group settles into stable formations
4. **Top-down projection** — map dancer positions to a 2D stage diagram
5. **PDF generation** — output a formation sheet with timestamps and diagrams

## Project Structure

```
src/
├── detection/     # Person detection
├── tracking/      # Multi-person tracking
├── formations/    # Spatial clustering, formation segmentation
└── output/        # PDF formation sheet generation

data/
├── raw/           # Original dance practice videos (not committed)
└── processed/     # Extracted frames, intermediate outputs (not committed)

notebooks/         # Jupyter notebooks for experimentation
outputs/           # Generated formation sheets (not committed)
tests/             # Unit tests
```

## Setup

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Status

In development — early setup phase.
