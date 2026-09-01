#!/usr/bin/env python3
"""Сверить таблицу `_CODE_RULES` с приказом 804н.

Классификатор типа визита живёт в
``audit.formal_structure.validator._CODE_RULES`` — маленькая таблица совпадений
по началу, середине и концу кода. Приказ здесь не источник данных, а способ её
проверить: скрипт читает номенклатуру, для каждой записи раздела B берёт вид
услуги из наименования и сравнивает с тем, что говорит таблица.

Источник по умолчанию — ``resources/nomenclature-804n.csv``, выгрузка текстового
слоя ``804_N_MZ.pdf`` (экспорт ГАРАНТ от 06.09.2023, ред. с изм. 24.09.2020),
10 441 код. Она лежит в репозитории ради того, чтобы ту же сверку гонял обычный
тест, а не только человек с корпусом НПА под рукой. Флаг ``--pdf`` читает сам
приказ — так CSV и пересобирают, когда выходит новая редакция.

**Чего проверка не ловит.** Выгрузка заморожена на одной редакции: если Минздрав
переставит окончания кодов, CSV об этом не узнает и тест останется зелёным.
Сторожит она регрессию классификатора, а не устаревание номенклатуры.

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

    python scripts/checks/check-nmu-classifier.py
    python scripts/checks/check-nmu-classifier.py --pdf ~/projects/minzdrav/804_N_MZ.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.formal_structure.validator import (  # noqa: E402
    NO_GUESS,
    VisitType,
    classify_code,
    classify_name,
)

CSV_PATH = ROOT / "resources" / "nomenclature-804n.csv"

# В PDF часть кодов набрана кириллической «В» — нормализуем обе раскладки.
_CODE_RE = re.compile(r"[BВ]0[1-5]\.\d{3}\.\d{3}")
# В CSV код лежит отдельным полем целиком, поэтому четвёртая группа не теряется:
# именно на ней проверяется запрет читать окончание у длинного кода.
_CSV_CODE_RE = re.compile(r"[BВ]0[1-5]\.\d{3}\.\d{3}(?:\.\d{3})?")
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


def entries_from_csv(path: Path = CSV_PATH) -> list[tuple[str, str]]:
    """(код, наименование) для записей раздела B из выгрузки номенклатуры."""
    with open(path, encoding="utf-8", newline="") as f:
        return [
            (row["code"], row["name"])
            for row in csv.DictReader(f)
            if _CSV_CODE_RE.fullmatch(row["code"]) and row["name"]
        ]


def _by_name(name: str) -> VisitType | None:
    for pattern, visit_type in _BY_NAME:
        if pattern.match(name):
            return visit_type
    return None


class Report(NamedTuple):
    """Итог сверки. Ошибками считаются contradictions и extra."""

    contradictions: list[str]
    extra: list[str]
    uncovered: list[str]
    barred: list[str]
    by_section: Counter
    agreed: Counter


def compare(entries: list[tuple[str, str]]) -> Report:
    """Сверить вердикт классификатора с наименованием по каждой записи."""
    report = Report([], [], [], [], Counter(), Counter())

    for code, name in entries:
        from_code = classify_code(code)
        from_name = _by_name(name)
        if code[:3] in ("B02", "B03", "B05"):
            # Вердикт вынесен разделом кода — это цитата из п. 5.1 приказа.
            # Сверять его с наименованием не с чем: _by_name знает только
            # шаблоны приёмов, а тут уход, диагностические комплексы и
            # реабилитация. Исключения из раздела проверяются ниже.
            if from_code is not NO_GUESS:
                report.by_section[code[:3]] += 1
                continue
        if from_code is NO_GUESS:
            # Ради этих строк _NAME_DESCRIBES_SERVICE и заведён: разбор
            # наименования выносит вердикт, а услуга приёмом не является.
            loose = classify_name(name)
            if loose is not None:
                report.barred.append(f"{code} → наименование дало бы {loose.name}: {name}")
            continue
        if from_code is None and from_name is None:
            continue
        if from_code is None:
            report.uncovered.append(f"{code} — {name}")
        elif from_name is None:
            report.extra.append(f"{code} → {from_code.name}, но по приказу это не приём: {name}")
        elif from_code is not from_name:
            report.contradictions.append(
                f"{code} → {from_code.name}, а по приказу {from_name.name}: {name}"
            )
        else:
            report.agreed[from_code.name] += 1

    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="перечитать приказ из PDF вместо resources/nomenclature-804n.csv",
    )
    parser.add_argument("--show", type=int, default=8, help="сколько примеров печатать")
    args = parser.parse_args(argv)
    source = args.pdf or CSV_PATH
    if not source.exists():
        parser.error(f"файл не найден: {source}")

    entries = _entries(args.pdf) if args.pdf else entries_from_csv()
    if len(entries) < 500:
        print(f"извлечено всего {len(entries)} записей — источник прочитан не полностью", file=sys.stderr)
        return 2

    r = compare(entries)

    print(f"источник: {source}")
    print(f"записей раздела B: {len(entries)}")
    print(f"вердикт по разделу кода (п. 5.1 приказа): {sum(r.by_section.values())} {dict(sorted(r.by_section.items()))}")
    print(f"эвристика окончания совпала с наименованием: {sum(r.agreed.values())} {dict(r.agreed)}")
    print(f"не покрыто таблицей (разбирается по наименованию): {len(r.uncovered)}")
    for line in r.uncovered[: args.show]:
        print(f"    {line[:110]}")
    if len(r.uncovered) > args.show:
        print(f"    … ещё {len(r.uncovered) - args.show}")

    print(f"остановлено списками исключений (иначе наименование соврало бы): {len(r.barred)}")
    for line in r.barred[: args.show]:
        print(f"    {line[:110]}")
    if len(r.barred) > args.show:
        print(f"    … ещё {len(r.barred) - args.show}")

    for label, items in (("ПРОТИВОРЕЧИЕ", r.contradictions), ("ЛИШНЕЕ", r.extra)):
        for line in items:
            print(f"{label}: {line[:130]}", file=sys.stderr)

    failed = len(r.contradictions) + len(r.extra)
    # Ноль здесь значит «эвристика окончания не разошлась с наименованиями на
    # текущей редакции 804н», а не «правило верное»: приказ окончания не
    # расшифровывает, см. комментарий к _CODE_RULES.
    print(f"\nрасхождений эвристики окончания с наименованиями 804н: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
