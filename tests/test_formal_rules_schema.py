import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_RULES_PATH = ROOT / "src" / "audit" / "formal_structure" / "rules.json"

_ISO = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def _doc() -> dict:
    return json.loads(_RULES_PATH.read_text(encoding="utf-8"))


def test_file_is_object_with_meta():
    doc = _doc()
    assert isinstance(doc, dict), "rules.json должен быть объектом, а не списком"
    assert _ISO.match(doc["revised_at"]), f"revised_at не ISO: {doc['revised_at']!r}"
    assert doc["sources_doc"] == "docs/formal-rules-sources.md"
    assert isinstance(doc["rules"], list) and doc["rules"]
