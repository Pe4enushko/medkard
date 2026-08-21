"""Static assertions on the GRLS registry migration SQL (no DB required)."""
import re
from pathlib import Path

SQL = (Path(__file__).resolve().parent.parent
       / "migrations" / "028_grls_registry.sql").read_text(encoding="utf-8")

STATUSES = (
    "Действующий",
    "Выдано по правилам ЕАЭС",
    "Действует, на подтверждении государственной регистрации",
    "Действует, в иностранных упаковках",
    "Приостановлено применение",
    "Истёкший",
    "Исключённый",
)

REGISTRY_COLUMNS = (
    "id", "status", "reg_number", "registered_at", "expires_at", "annulled_at",
    "holder", "holder_country", "trade_name", "inn_name", "forms", "forms_raw",
    "dosage_forms", "dispensing", "is_substance", "production_stages",
    "normative_docs", "pharm_group", "is_vital", "narcotic_list", "is_orphan",
    "row_hash", "imported_at",
)


def test_creates_both_tables():
    assert "CREATE TABLE IF NOT EXISTS grls_registry" in SQL
    assert "CREATE TABLE IF NOT EXISTS grls_imports" in SQL


def test_status_check_lists_all_seven_verbatim():
    for s in STATUSES:
        assert f"'{s}'" in SQL, s
    assert "CHECK (status IN (" in SQL


def test_row_hash_unique():
    assert re.search(r"row_hash\s+TEXT\s+NOT NULL\s+UNIQUE", SQL)


def test_grls_norm_function_is_immutable():
    assert "CREATE OR REPLACE FUNCTION grls_norm(" in SQL
    assert "IMMUTABLE" in SQL


def test_functional_trgm_indexes():
    assert "USING GIN (grls_norm(trade_name) gin_trgm_ops)" in SQL
    assert "USING GIN (grls_norm(inn_name) gin_trgm_ops)" in SQL


def test_every_registry_column_has_comment():
    for col in REGISTRY_COLUMNS:
        assert f"COMMENT ON COLUMN grls_registry.{col} IS" in SQL, col


def test_drops_drugs_table():
    assert "DROP TABLE IF EXISTS drugs" in SQL
