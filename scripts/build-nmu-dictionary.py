#!/usr/bin/env python3
"""Собрать словарь типов амбулаторного приёма из приказа 804н.

Вход — PDF номенклатуры медицинских услуг (приказ Минздрава России от
13.10.2017 № 804н). Выход — ``src/audit/formal_structure/nmu_visit_types.json``,
который читает ``FormalValidator``.

Берём только то, что бывает в обычной поликлинике:

* ``B01.*`` вида «Прием (осмотр, консультация) … первичный/повторный»,
  «Прием (тестирование, консультация) … первичный/повторный» и
  «Осмотр (консультация) врачом-… первичный/повторный» → PRIMARY/REPEAT;
* ``B04.*`` вида «Диспансерный прием …» и «Профилактический прием …» →
  PROPHYLACTIC.

Всё остальное из 804н намеренно НЕ включается: ежедневные осмотры и суточное
наблюдение (стационар), ведение родов, анестезиологические пособия, патронаж
выездной бригадой, медицинское освидетельствование, школы и консультирования,
сестринский уход (``B02``), комплексы исследований (``B03``), реабилитация
(``B05``). Такой код словарь не найдёт, услуга уйдёт в разбор по наименованию,
а затем в ``OTHER`` — то есть в сегодняшнее поведение.

Лабораторные, инструментальные и вмешательства (``A*``) в словаре не нужны:
у них тип определяется префиксом, а конкретную услугу правила отбирают через
``applies_to.service_code_prefixes``.

Запуск::

    python scripts/build-nmu-dictionary.py ~/projects/minzdrav/804_N_MZ.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "src" / "audit" / "formal_structure" / "nmu_visit_types.json"

SOURCE = (
    "приказ Минздрава России от 13.10.2017 № 804н «Об утверждении номенклатуры "
    "медицинских услуг», раздел B"
)

# В PDF часть кодов набрана кириллической «В» — нормализуем обе раскладки.
_CODE_RE = re.compile(r"[BВ]0[1-5]\.\d{3}\.\d{3}")
# Хвост записи в PDF цепляет сноски и ссылки на приказы — режем по ним.
_NAME_TAIL_RE = re.compile(r"Приказ Министерства|Утратил[аи]? силу|<\d")

# Одна и та же сущность записана в 804н двумя способами: «Прием (осмотр,
# консультация) врача-невролога первичный» и «Осмотр (консультация) врачом-
# радиологом первичный».
_APPOINTMENT = (
    r"(?:Прием \((?:осмотр, консультация|тестирование, консультация)\)"
    r"|Осмотр \(консультация\))"
)
_PRIMARY_RE = re.compile(rf"^{_APPOINTMENT}.+первичный$")
_REPEAT_RE = re.compile(rf"^{_APPOINTMENT}.+повторный$")
_PROPHYLACTIC_RE = re.compile(r"^(Диспансерный|Профилактический) прием \(")


def _extract_entries(pdf_path: Path) -> list[tuple[str, str]]:
    """Вернуть (код, наименование) для каждой записи раздела B по порядку PDF."""
    import fitz  # noqa: PLC0415 — тяжёлый импорт нужен только этому скрипту

    with fitz.open(pdf_path) as doc:
        flat = re.sub(r"\s+", " ", "\n".join(page.get_text() for page in doc))

    matches = list(_CODE_RE.finditer(flat))
    entries: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(flat)
        name = _NAME_TAIL_RE.split(flat[match.end() : end])[0].strip(" .;·—-")
        entries.append((match.group(0).replace("В", "B"), name))
    return entries


def _classify(name: str) -> str | None:
    if _PRIMARY_RE.match(name):
        return "PRIMARY"
    if _REPEAT_RE.match(name):
        return "REPEAT"
    if _PROPHYLACTIC_RE.match(name):
        return "PROPHYLACTIC"
    return None


def build(pdf_path: Path) -> dict:
    codes: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []

    for code, name in _extract_entries(pdf_path):
        visit_type = _classify(name)
        if visit_type is None:
            continue
        existing = codes.get(code)
        if existing is None:
            codes[code] = {"visit_type": visit_type, "name": name}
        elif existing["visit_type"] != visit_type:
            conflicts.append(f"{code}: {existing['name']!r} vs {name!r}")

    if conflicts:
        raise SystemExit(
            "804н дал противоречивые типы для одного кода:\n  " + "\n  ".join(conflicts)
        )
    if len(codes) < 150:
        raise SystemExit(
            f"извлечено всего {len(codes)} кодов — похоже, PDF распознан не полностью"
        )

    return {
        "source": SOURCE,
        "verified_at": date.today().isoformat(),
        "scope": (
            "Только амбулаторные приёмы: B01 «Прием … первичный/повторный» и "
            "B04 «Диспансерный/Профилактический прием». Остальные записи 804н "
            "(ежедневный осмотр, ведение родов, анестезия, патронаж, "
            "освидетельствование, школы, B02/B03/B05) намеренно не включены — "
            "такой код уходит в разбор по наименованию услуги, затем в OTHER."
        ),
        "generated_by": "scripts/build-nmu-dictionary.py",
        "codes": dict(sorted(codes.items())),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="путь к PDF приказа 804н")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        parser.error(f"файл не найден: {args.pdf}")

    document = build(args.pdf)
    args.out.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for entry in document["codes"].values():
        counts[entry["visit_type"]] = counts.get(entry["visit_type"], 0) + 1
    print(f"{args.out}: {len(document['codes'])} кодов {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
