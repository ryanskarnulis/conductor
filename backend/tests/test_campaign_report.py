"""The campaign report pairs samples by seed across arms (#53). CI, no GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "campaign_report.py"


def _report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("campaign_report", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(out: Path, block: int, arm: str, passes: list[bool], seed: int | None) -> None:
    samples = [
        {"seed": None if seed is None else seed + j, "pass": ok} for j, ok in enumerate(passes)
    ]
    record = {"scenario": "S2", "samples": samples}
    (out / f"block-{block}-{arm}.jsonl").write_text(json.dumps(record) + "\n")


def test_pairs_by_seed_and_reads_the_flips(tmp_path: Path) -> None:
    _block(tmp_path, 1, "a", [True, True, False, False, True], 1053)
    _block(tmp_path, 1, "b", [True, True, True, True, True], 1053)
    lines = _report().table(_report().load(tmp_path))
    assert lines[1].startswith("S2")
    assert "3/5" in lines[1] and "5/5" in lines[1]
    assert "pairs 5 · A-only 0 · B-only 2" in lines[1]


def test_unseeded_blocks_are_counted_not_paired(tmp_path: Path) -> None:
    _block(tmp_path, 1, "a", [True, False], None)
    _block(tmp_path, 1, "b", [True, True], None)
    assert "unpaired" in _report().table(_report().load(tmp_path))[1]


def test_plan_eval_rows_count_per_case_unpaired(tmp_path: Path) -> None:
    # The plan suite (#75) writes one row per sample: a case and its verdict.
    rows = [{"case": "chess-resign", "group": "golden", "correct": ok} for ok in (True, False)]
    (tmp_path / "block-1-a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "block-1-b.jsonl").write_text(json.dumps({**rows[0]}) + "\n")
    (line,) = _report().table(_report().load(tmp_path))[1:]
    assert line.startswith("chess-resign") and "1/2" in line and "1/1" in line
    assert "unpaired" in line
