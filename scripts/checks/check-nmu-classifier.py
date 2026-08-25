#!/usr/bin/env python3
"""Сверить таблицу `_CODE_RULES` с приказом 804н.

Классификатор типа визита живёт в
``audit.formal_structure.validator._CODE_RULES`` — маленькая таблица совпадений
по началу, середине и концу кода. Приказ здесь не источник данных, а способ её
проверить: скрипт читает PDF номенклатуры, для каждой записи раздела B берёт вид
услуги из наименования и сравнивает с тем, что говорит таблица.

Три исхода:

* **противоречие** — таблица и приказ называют разные виды приёма. Это ошибка
  таблицы, скрипт завершается ненулевым кодом;
* **лишнее** — таблица выносит вердикт там, где приказ не видит приёма
  (ежедневный осмотр, ведение родов, патронаж). Тоже ошибка;
* **не покрыто** — приказ видит приём, таблица молчит. Это ожидаемо и не
  ошибка: устойчиво разбираются только окончания .001/.002, остальные пары
  (участковый, подростковый, «беременной») распознаются по наименованию услуги.
  Скрипт печатает их числом и примерами, чтобы решение оставалось осознанным.

Запуск::

    python scripts/checks/check-nmu-classifier.py ~/projects/minzdrav/804_N_MZ.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import VisitType, classify_code  # noqa: E402

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
_BY_NAME: tuple[tuple[re.Pattern[str], VisitType], ...] = (
    (re.compile(rf"^{_APPOINTMENT}.+первичный$"), VisitType.PRIMARY),
    (re.compile(rf"^{_APPOINTMENT}.+повторный$"), VisitType.REPEAT),
    (re.compile(r"^Диспансерный прием \("), VisitType.DISPENSARY),
    (re.compile(r"^Профилактический прием \("), VisitType.PROPHYLACTIC),
)


def _entries(pdf_path: Path) -> list[tuple[str, str]]:
    """(код, наименование) для каждой записи раздела B, в порядке PDF."""
    import fitz  # noqa: PLC0415 — тяжёлый импорт нужен только этому скрипту

    with fitz.open(pdf_path) as doc:
        flat = re.sub(r"\s+", " ", "\n".join(page.get_text() for page in doc))

    matches = list(_CODE_RE.finditer(flat))
    out: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(flat)
        name = _NAME_TAIL_RE.split(flat[match.end() : end])[0].strip(" .;·—-")
        out.append((match.group(0).replace("В", "B"), name))
    return out


def _by_name(name: str) -> VisitType | None:
    for pattern, visit_type in _BY_NAME:
        if pattern.match(name):
            return visit_type
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="путь к PDF приказа 804н")
    parser.add_argument("--show", type=int, default=8, help="сколько примеров печатать")
    args = parser.parse_args(argv)
    if not args.pdf.exists():
        parser.error(f"файл не найден: {args.pdf}")

    entries = _entries(args.pdf)
    if len(entries) < 500:
        print(f"извлечено всего {len(entries)} записей — PDF распознан не полностью", file=sys.stderr)
        return 2

    contradictions: list[str] = []
    extra: list[str] = []
    uncovered: list[str] = []
    agreed = Counter()

    for code, name in entries:
        from_code = classify_code(code)
        from_name = _by_name(name)
        if from_code is None and from_name is None:
            continue
        if from_code is None:
            uncovered.append(f"{code} — {name}")
        elif from_name is None:
            extra.append(f"{code} → {from_code.name}, но по приказу это не приём: {name}")
        elif from_code is not from_name:
            contradictions.append(f"{code} → {from_code.name}, а по приказу {from_name.name}: {name}")
        else:
            agreed[from_code.name] += 1

    print(f"записей раздела B: {len(entries)}")
    print(f"совпало: {sum(agreed.values())} {dict(agreed)}")
    print(f"не покрыто таблицей (разбирается по наименованию): {len(uncovered)}")
    for line in uncovered[: args.show]:
        print(f"    {line[:110]}")
    if len(uncovered) > args.show:
        print(f"    … ещё {len(uncovered) - args.show}")

    for label, items in (("ПРОТИВОРЕЧИЕ", contradictions), ("ЛИШНЕЕ", extra)):
        for line in items:
            print(f"{label}: {line[:130]}", file=sys.stderr)

    failed = len(contradictions) + len(extra)
    print(f"\nошибок таблицы: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
