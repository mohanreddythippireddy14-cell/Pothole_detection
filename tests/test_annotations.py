"""VOC -> YOLO conversion and dataset-merge helpers (scripts/02 and scripts/11)."""

from pathlib import Path

import pytest

from conftest import load_script

VOC_XML = """<annotation>
  <filename>potholes1.png</filename>
  <size><width>400</width><height>200</height><depth>3</depth></size>
  <object>
    <name>pothole</name>
    <bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox>
  </object>
  <object>
    <name>pothole</name>
    <bndbox><xmin>30</xmin><ymin>10</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
  </object>
</annotation>
"""


@pytest.fixture
def voc_file(tmp_path: Path) -> Path:
    path = tmp_path / "potholes1.xml"
    path.write_text(VOC_XML)
    return path


def test_convert_voc_to_yolo_normalizes_and_drops_degenerate_box(voc_file):
    convert = load_script("02_convert_voc_to_yolo.py").convert_voc_to_yolo
    lines, n_skipped = convert(voc_file)

    assert n_skipped == 1  # the zero-width box (xmin == xmax)
    assert lines == ["0 0.375000 0.500000 0.250000 0.500000"]


def test_parse_voc_skips_degenerate_box(voc_file):
    parse_voc = load_script("11_build_combined_kfold_dataset.py").parse_voc
    w, h, boxes = parse_voc(voc_file)

    assert (w, h) == (400, 200)
    assert boxes == [(100.0, 50.0, 200.0, 150.0)]


def test_voc_boxes_to_yolo_lines_matches_scripts_02_format():
    to_lines = load_script("11_build_combined_kfold_dataset.py").voc_boxes_to_yolo_lines
    assert to_lines(400, 200, [(100, 50, 200, 150)]) == ["0 0.375000 0.500000 0.250000 0.500000"]
    assert to_lines(400, 200, []) == []


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("img-105_jpg.rf.3fe9dff3d1631e79ecb480ff403bcb86.jpg", "img-105"),
        ("img-105_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg", "img-105"),
        ("plain_name.jpg", "plain_name"),
    ],
)
def test_roboflow_base_stem_groups_augmented_variants(filename, expected):
    base_stem = load_script("11_build_combined_kfold_dataset.py").roboflow_base_stem
    assert base_stem(Path(filename)) == expected
