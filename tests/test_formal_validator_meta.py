import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _import_log() -> str:
    """Import the validator in a fresh interpreter and return its log output."""
    code = (
        "import logging; logging.basicConfig(level=logging.INFO);"
        " import audit.formal_structure.validator"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_load_logs_revision_meta():
    """Модуль пишет revised_at, число правил и самую старую дату сверки при загрузке."""
    text = _import_log()
    assert "revised_at=2026-08-21" in text
    assert "rules=42" in text
    assert "oldest verified_at=2026-08-19" in text
