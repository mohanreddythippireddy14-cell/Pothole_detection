"""Severity engine rules and mask geometry (scripts/10_full_pipeline.py)."""

import numpy as np
import pytest

from conftest import load_script

pipeline = load_script("10_full_pipeline.py")
SeverityPipeline = pipeline.SeverityPipeline
AREA_LO, AREA_HI = pipeline.SEVERITY_AREA_THRESHOLDS
DEPTH_T = pipeline.DEPTH_STEEPNESS_ESCALATION_THRESHOLD
severity = SeverityPipeline._severity_from_features


def test_calibration_file_is_loaded():
    # the tracked context/severity_calibration.json must be found -- if it weren't,
    # depth escalation would silently be disabled (threshold = inf)
    assert np.isfinite(DEPTH_T)
    assert 0 < AREA_LO < AREA_HI < 1


def test_no_mask_gives_unknown_not_low():
    assert severity(False, 0.5, None) == "UNKNOWN"


@pytest.mark.parametrize(
    "relative_area, expected",
    [(AREA_LO / 2, "LOW"), ((AREA_LO + AREA_HI) / 2, "MEDIUM"), (AREA_HI * 2, "HIGH")],
)
def test_area_bins_without_depth(relative_area, expected):
    assert severity(True, relative_area, None) == expected


def test_area_bin_edges_are_lower_inclusive():
    assert severity(True, AREA_LO, None) == "MEDIUM"
    assert severity(True, AREA_HI, None) == "HIGH"


@pytest.mark.parametrize(
    "relative_area, expected",
    [(AREA_LO / 2, "MEDIUM"), ((AREA_LO + AREA_HI) / 2, "HIGH"), (AREA_HI * 2, "HIGH")],
)
def test_steep_pothole_escalates_one_level_capped_at_high(relative_area, expected):
    assert severity(True, relative_area, DEPTH_T * 1.5) == expected


def test_shallow_pothole_does_not_escalate():
    assert severity(True, AREA_LO / 2, DEPTH_T * 0.5) == "LOW"


def test_geometry_from_mask_uses_full_image_for_relative_area():
    mask = np.zeros((50, 80), dtype=np.uint8)  # crop-sized mask
    mask[10:20, 30:60] = 1  # 10 rows x 30 cols
    geometry = SeverityPipeline._geometry_from_mask(None, mask, (100, 200))

    assert geometry["mask_found"] is True
    assert (geometry["width_px"], geometry["height_px"]) == (30, 10)
    assert geometry["area_px"] == 300
    assert geometry["relative_area"] == pytest.approx(300 / (100 * 200))


def test_geometry_from_empty_mask():
    geometry = SeverityPipeline._geometry_from_mask(None, np.zeros((5, 5), np.uint8), (10, 10))
    assert geometry["mask_found"] is False
    assert geometry["relative_area"] == 0.0
