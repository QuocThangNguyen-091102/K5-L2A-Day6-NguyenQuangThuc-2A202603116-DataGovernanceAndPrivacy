import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.common.context import RunContext
from src.pipeline import process


def _context(tmp_path, input_dir=None):
    return RunContext(
        track="t02",
        input_dir=Path(input_dir or tmp_path),
        output_dir=tmp_path / "out",
        config={"params": {"budget": 20}},
        seed=42,
        rng=np.random.default_rng(42),
    )


def test_t02_process_scores_images(tmp_path):
    source = "fixtures/t02_raw_qc"
    images = pd.read_csv(f"{source}/images.csv").head(3).copy()
    result = process(_context(tmp_path, source), {"images": images})
    assert len(result) == 3
    assert {"image_id", "score", "reason"}.issubset(result.columns)
    assert result["score"].between(0, 1).all()


def test_t02_process_fails_on_missing_image(tmp_path):
    images = pd.DataFrame(
        [{"image_id": "missing", "path": "images/nope.jpg", "camera_id": "cam_A", "light": "day", "split": "dev"}]
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        process(_context(tmp_path), {"images": images})
