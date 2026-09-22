"""Reproducibility tests — verifies the Phase 7 history artifacts exist with
sane epoch counts (early stopping engaged) and that the notebook contains the
two reviewer-facing notes (SEED constant + non-determinism caveat).

These six assertions cover: LSTM/GRU history files exist, both ran for a
plausible number of epochs (10-100, bounded above by the 50-epoch fit cap and
below by sanity), and the Phase 7 narrative documents the GPU non-determinism
caveat that a commission member is likely to ask about.
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "outputs" / "results"
LSTM_HIST = RESULTS / "lstm_history.json"
GRU_HIST = RESULTS / "gru_history.json"
NOTEBOOK = PROJECT_ROOT / "diploma_pipeline.ipynb"


@pytest.fixture(scope="module")
def notebook():
    assert NOTEBOOK.exists(), f"missing notebook {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _epoch_count(history_path: Path) -> int:
    h = json.loads(history_path.read_text(encoding="utf-8"))
    assert "loss" in h, f"{history_path.name} missing 'loss' key"
    return len(h["loss"])


def test_lstm_history_exists():
    assert LSTM_HIST.exists(), f"Phase 7 LSTM history missing: {LSTM_HIST}"


def test_gru_history_exists():
    assert GRU_HIST.exists(), f"Phase 7 GRU history missing: {GRU_HIST}"


def test_lstm_epoch_count_in_range():
    n = _epoch_count(LSTM_HIST)
    assert 10 <= n <= 100, f"LSTM ran {n} epochs (expected EarlyStopping in [10, 100])"


def test_gru_epoch_count_in_range():
    n = _epoch_count(GRU_HIST)
    assert 10 <= n <= 100, f"GRU ran {n} epochs (expected EarlyStopping in [10, 100])"


def test_notebook_phase7_mentions_non_determinism(notebook):
    """The Phase 7 narrative must document the GPU non-determinism caveat."""
    in_phase7 = False
    for cell in notebook["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            if "Phase 7" in src and src.lstrip().startswith("#"):
                in_phase7 = True
            elif src.lstrip().startswith("# Phase 8") or src.lstrip().startswith("## Phase 8"):
                in_phase7 = False
        if in_phase7 and "non-determinism" in src.lower():
            return
    raise AssertionError(
        "no Phase 7 markdown cell mentions 'non-determinism' — "
        "add the reviewer-facing reproducibility note"
    )


def test_notebook_contains_seed_constant(notebook):
    for cell in notebook["cells"]:
        src = "".join(cell.get("source", []))
        if "SEED = 42" in src:
            return
    raise AssertionError("notebook does not contain the literal 'SEED = 42'")
