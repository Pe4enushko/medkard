from __future__ import annotations

from dataclasses import dataclass, field

_ID = "ID"
_NAME = "Наименование"
_MKB = "МКБ-10"
_AGE = "Возрастная категория"
_DEVELOPER = "Разработчик"
_NPS = "Статус одобрения НПС"
_PUBLISHED = "Дата размещения"
_USAGE = "Статус применения"


def _split_csv_cell(cell: str, *, upper: bool = False) -> list[str]:
    """Разбить ячейку манифеста по запятой; strip; опционально upper. Пусто → []."""
    parts = [p.strip() for p in (cell or "").split(",")]
    parts = [p for p in parts if p]
    return [p.upper() for p in parts] if upper else parts


def name_embed_input(name: str | None, age_category: list[str] | None) -> str:
    """Passage string embedded for the guideline registry: labeled title + age category.

    CROSS-PROJECT CONTRACT: engine (integrations/clinrec/mapping.py) rebuilds this
    byte-for-byte for its fallback re-embed. Do not change form without updating both.
    Passage mode — bare embed, no instruct prefix.
    """
    base = f"Название: {(name or '').strip()}"
    ages = [a.strip() for a in (age_category or []) if a and a.strip()]
    return f"{base}\nВозрастная группа: [{', '.join(ages)}]" if ages else base


@dataclass
class Guideline:
    """Строка справочника клинреков (зеркало строки manifest.csv)."""

    file_id: str
    name: str | None = None
    mkb: list[str] = field(default_factory=list)
    age_category: list[str] = field(default_factory=list)
    developer: str | None = None
    nps_status: str | None = None
    published_at: str | None = None
    usage_status: str | None = None
    name_embedding: list[float] | None = None

    @classmethod
    def from_manifest_row(cls, row: dict[str, str]) -> "Guideline":
        return cls(
            file_id=(row.get(_ID) or "").strip(),
            name=row.get(_NAME) or None,
            mkb=_split_csv_cell(row.get(_MKB, ""), upper=True),
            age_category=_split_csv_cell(row.get(_AGE, "")),
            developer=row.get(_DEVELOPER) or None,
            nps_status=row.get(_NPS) or None,
            published_at=row.get(_PUBLISHED) or None,
            usage_status=row.get(_USAGE) or None,
        )
