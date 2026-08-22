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
    """Модуль пишет revised_at, число правил и самую старую дату сверки при загрузке.

    Значения сверяются с самим rules.json, а не с датой в тексте теста: проверяется
    проводка лога, а каждая ревизия правил иначе роняет тест на ровном месте.
    """
    import json

    doc = json.loads((ROOT / "src" / "audit" / "formal_structure" / "rules.json").read_text("utf-8"))
    oldest = min(r["verified_at"] for r in doc["rules"] if r.get("verified_at"))

    text = _import_log()
    assert f"revised_at={doc['revised_at']}" in text
    assert f"rules={len(doc['rules'])}" in text
    assert f"oldest verified_at={oldest}" in text
