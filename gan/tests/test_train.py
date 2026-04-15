import pytest
import os
import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the function under test
sys.modules['numpy'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.optimizers'] = MagicMock()

from pathlib import Path
from gan.src.train import get_image_lists

def test_get_image_lists_happy_path(tmp_path):
    # Setup
    base_path = tmp_path / "data"
    rgb_dir = base_path / "rgb"
    thermal_dir = base_path / "thermal"
    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir(parents=True)

    rgb_file = rgb_dir / "img1.jpg"
    rgb_file.write_text("dummy content")

    thermal_file = thermal_dir / "img1.jpg"
    thermal_file.write_text("dummy content")

    # Execute
    rgbs, thermals = get_image_lists(base_path)

    # Verify
    assert len(rgbs) == 1
    assert len(thermals) == 1
    assert str(rgb_file) in rgbs
    assert str(thermal_file) in thermals

def test_get_image_lists_empty_directories(tmp_path):
    # Setup
    base_path = tmp_path / "data"
    (base_path / "rgb").mkdir(parents=True)
    (base_path / "thermal").mkdir(parents=True)

    # Execute
    rgbs, thermals = get_image_lists(base_path)

    # Verify
    assert rgbs == []
    assert thermals == []

def test_get_image_lists_missing_directories(tmp_path):
    # Setup
    base_path = tmp_path / "non_existent"

    # Execute
    rgbs, thermals = get_image_lists(base_path)

    # Verify
    assert rgbs == []
    assert thermals == []

def test_get_image_lists_zero_sized_files(tmp_path):
    # Setup
    base_path = tmp_path / "data"
    rgb_dir = base_path / "rgb"
    thermal_dir = base_path / "thermal"
    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir(parents=True)

    rgb_file = rgb_dir / "empty.jpg"
    rgb_file.write_text("") # Zero size

    thermal_file = thermal_dir / "valid.jpg"
    thermal_file.write_text("not empty")

    # Execute
    rgbs, thermals = get_image_lists(base_path)

    # Verify
    assert rgbs == []
    assert len(thermals) == 1
