"""
formations package

Modules
-------
homography: perspective calibration + top-down coordinate transform
generate_synthetic: synthetic tracks.json generator for testing
"""

from .homography import (
    HomographyCalibrator,
    save_calibration,
    load_calibration,
    top_down_positions,
)

__all__ = [
    "HomographyCalibrator",
    "save_calibration",
    "load_calibration",
    "top_down_positions",
]