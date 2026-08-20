import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_load_logs_revision_meta(caplog):
    """Модуль пишет revised_at и число правил при загрузке."""
    import audit.formal_structure.validator as v

    with caplog.at_level("INFO", logger="audit.formal_structure.validator"):
        importlib.reload(v)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "revised_at=2026-08-19" in text
    assert f"rules={len(v._RULES)}" in text
    assert "oldest verified_at=" in text
