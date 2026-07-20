from pathlib import Path

from RAG.ingestion.reingest_planner import base_id, classify, sha256_file, stale_revision_ids
from storage.models.guideline import Guideline


def _g(file_id="F1", name="A", mkb=None):
    return Guideline(file_id=file_id, name=name, mkb=mkb or ["I10"])


# --- classify: full-reingest triggers ---
def test_no_row_is_full():
    assert classify(status=None, stored_hash=None, current_hash="h1",
                    stored_guideline=None, new_guideline=_g()) == "full"


def test_pending_is_full():
    assert classify(status="pending", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_failed_is_full():
    assert classify(status="failed", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_hash_changed_is_full_even_if_metadata_same():
    assert classify(status="done", stored_hash="old", current_hash="new",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


def test_rollback_hash_differs_from_last_done_is_full():
    # PDF rolled back to an older version → current hash != last-done hash → full
    assert classify(status="done", stored_hash="hB", current_hash="hA",
                    stored_guideline=_g(), new_guideline=_g()) == "full"


# --- classify: metadata-only ---
def test_done_same_hash_metadata_diff_is_metadata_only():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(name="Old"), new_guideline=_g(name="New")) == "metadata_only"


def test_done_same_hash_mkb_diff_is_metadata_only():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(mkb=["I10"]), new_guideline=_g(mkb=["I11"])) == "metadata_only"


# --- classify: skip ---
def test_done_same_hash_same_metadata_is_skip():
    assert classify(status="done", stored_hash="h1", current_hash="h1",
                    stored_guideline=_g(), new_guideline=_g()) == "skip"


# --- sha256_file ---
def test_sha256_file_deterministic_and_sensitive(tmp_path: Path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"hello")
    first = sha256_file(p)
    assert first == sha256_file(p)  # deterministic
    p.write_bytes(b"hello!")
    assert sha256_file(p) != first  # sensitive to content


# --- build_worklist ---
# Stored guidelines are built via from_manifest_row so equality with the "new"
# guideline holds by construction — the skip/metadata_only split is unambiguous.
def test_build_worklist_mixed():
    from RAG.ingestion.reingest_planner import build_worklist

    rows = {
        "A": {"ID": "A", "Наименование": "A"},        # done, hash same, meta same -> skip
        "B": {"ID": "B", "Наименование": "B-new"},    # done, hash same, meta diff -> metadata_only
        "C": {"ID": "C", "Наименование": "C"},        # failed                     -> full
        "D": {"ID": "D", "Наименование": "D"},        # no ingest_runs row          -> full
    }
    runs = {"A": ("done", "hA"), "B": ("done", "hB"), "C": ("failed", "hC")}
    guidelines_by_id = {
        "A": Guideline.from_manifest_row(rows["A"]),                              # == new -> skip
        "B": Guideline.from_manifest_row({"ID": "B", "Наименование": "B-old"}),   # != new -> metadata_only
        "C": Guideline.from_manifest_row(rows["C"]),
    }
    hash_of = {"A": "hA", "B": "hB", "C": "hC", "D": "hD"}.get  # dict.get -> None if missing

    wl = dict(build_worklist(rows, runs, guidelines_by_id, hash_of))
    assert wl == {"A": "skip", "B": "metadata_only", "C": "full", "D": "full"}


def test_build_worklist_skips_missing_pdf():
    from RAG.ingestion.reingest_planner import build_worklist

    rows = {"A": {"ID": "A", "Наименование": "A"}}
    wl = build_worklist(rows, {}, {}, lambda fid: None)  # PDF missing on disk
    assert wl == []


# --- base_id ---
def test_base_id_strips_revision_suffix():
    assert base_id("318_3") == "318"
    assert base_id("1027_1") == "1027"


def test_base_id_none_for_non_numeric():
    assert base_id("A") is None
    assert base_id("") is None


# --- stale_revision_ids ---
def test_stale_revision_ids_finds_superseded_revision():
    # manifest moved from 318_2 to 318_3; 318_2 still known in the DB
    assert stale_revision_ids(["318_3"], ["318_2", "318_3"]) == ["318_2"]


def test_stale_revision_ids_ignores_unrelated_bases():
    assert stale_revision_ids(["318_3"], ["999_1"]) == []


def test_stale_revision_ids_ignores_non_numeric_ids():
    assert stale_revision_ids(["A"], ["A_old"]) == []


def test_stale_revision_ids_empty_when_no_stale_data():
    assert stale_revision_ids(["318_3", "999_1"], ["318_3", "999_1"]) == []
