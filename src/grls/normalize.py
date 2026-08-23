"""Pure normalization helpers for the GRLS import (no I/O)."""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime

logger = logging.getLogger(__name__)

_NULL_MARKERS = {"", "~"}
_XLSX_CR = "_x000D_"
_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
_HASH_SEP = "\x1f"
_SUBSTANCE_PREFIX = "ФС-"
_SUBSTANCE_FORM = "субстанция"

# Keep in sync with SQL grls_norm() in migrations/028_grls_registry.sql.
_DROP_CHARS = "\"«»„“”‘’'®™©~"
_QUERY_TABLE = str.maketrans({"ё": "е", **{c: None for c in _DROP_CHARS}})
_PLUS_RE = re.compile(r"\s*\+\s*")


def clean_cell(value: object) -> str | None:
    """Cell → stripped text; '' and '~' → None; xlsx CR artefact removed."""
    if value is None:
        return None
    text = str(value).replace(_XLSX_CR, "").strip()
    return None if text in _NULL_MARKERS else text


def parse_date(value: object) -> date | None:
    """'DD.MM.YYYY[ HH:MM[:SS]]' | datetime | date → date; junk → warning + None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text in _NULL_MARKERS:
        return None
    m = _DATE_RE.match(text)
    if not m:
        logger.warning("GRLS: unparsable date %r", text)
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        logger.warning("GRLS: invalid calendar date %r", text)
        return None


def split_forms(forms_raw: str | None) -> list[str]:
    if not forms_raw:
        return []
    return [p.strip() for p in forms_raw.split(";") if p.strip()]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def derive_dosage_forms(forms: list[str]) -> list[str]:
    """First comma-segment of each element; fragments starting with '-' skipped."""
    return _unique([el.split(",", 1)[0].strip() for el in forms if not el.startswith("-")])


def derive_dispensing(forms: list[str]) -> list[str]:
    """Last ' - '-segment of each element; elements without the separator skipped."""
    return _unique([el.rsplit(" - ", 1)[1].strip() for el in forms if " - " in el])


def is_substance(reg_number: str, dosage_forms: list[str]) -> bool:
    return reg_number.startswith(_SUBSTANCE_PREFIX) or any(
        f.lower().startswith(_SUBSTANCE_FORM) for f in dosage_forms)


def parse_yes_no(value: object) -> bool | None:
    text = clean_cell(value)
    if text is None:
        return None
    low = text.lower()
    return True if low == "да" else False if low == "нет" else None


def parse_narcotic(value: object) -> str | None:
    text = clean_cell(value)
    if text is None or text.lower() == "нет":
        return None
    return text


def _hash_part(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def row_hash(*, status, reg_number, registered_at, expires_at, annulled_at, holder,
             holder_country, trade_name, inn_name, forms_raw, production_stages,
             normative_docs, pharm_group, is_vital, narcotic_list, is_orphan) -> str:
    """sha256 over the fixed-order source tuple (spec §4.3). Sync contract with engine."""
    parts = (status, reg_number, registered_at, expires_at, annulled_at, holder,
             holder_country, trade_name, inn_name, forms_raw, production_stages,
             normative_docs, pharm_group, is_vital, narcotic_list, is_orphan)
    payload = _HASH_SEP.join(_hash_part(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_query(text: str) -> str:
    """Python mirror of SQL grls_norm(): lower, drop quotes/®™©/~, ё→е, collapse
    whitespace, убрать пробелы вокруг «+».

    Про «+»: врач пишет «Амоксициллин + Клавулановая кислота», реестр хранит
    «Амоксициллин+клавулановая кислота». Без общей формы это разные строки, и
    составное МНН опознаётся как «похожее».
    """
    collapsed = " ".join(text.lower().translate(_QUERY_TABLE).split())
    return _PLUS_RE.sub("+", collapsed)
