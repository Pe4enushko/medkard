#!/usr/bin/env python3
"""
Развернуть guideline_file_id в снимок редакции у карт, проверенных до снимков.

WHY THIS EXISTS: аудит теперь кладёт в diag_result поле guideline_meta —
{name, date, age_group} той редакции клинрека, против которой шла проверка
(src/audit/diagnosis/validator.py). У карт, проаудированных раньше, в строке
остался голый file_id, а развернуть его на чтении можно не всегда: редакции
меняются, и file_id пропадает из манифеста вместе со старой. Скрипт проходит
по таким картам один раз и проставляет снимок из текущего справочника.

Заодно вынимает из строки наши errors: diag_result уезжает в реплику движка и
попадает агенту медчека как есть, а он прочитает нашу деградацию как факт о
карте и понесёт врачу. Аварии переезжают в done_cards.diag_errors (миграция 030),
откуда наружу не выходят.

Ничего не выдумывает: file_id, которого в справочнике уже нет, пропускается —
карта остаётся как была, и в отчёте у неё, как и сейчас, будет один номер
вместо названия. Карту со снимком скрипт не трогает: там записана та редакция,
против которой её проверяли, и текущая ей не замена.

Запускать из корня проекта (dry-run без -y):

    python scripts/hacks/backfill-guideline-meta.py [--limit 100] [--batch 500] [-y]

Options:
    --limit    Обработать не больше стольких карт; 0 — все
    --batch    Размер пачки на один SELECT (по умолчанию 500)
    -y         Писать. Без него скрипт только считает, что сделал бы.

UPDATE поднимает updated_at (триггер done_cards_set_updated_at, миграция 022),
поэтому реплика движка заберёт изменённые карты обычным инкрементальным синком.
На полной перезаливке (MEDCHECK_INCREMENTAL_SYNC выключен) это ничего не меняет.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reporting.result_parser import build_manifest_meta
from storage.done_cards_storage import DoneCardsStorage
from storage.guidelines_storage import GuidelinesStorage


def expand(
    diag_result: list[dict], manifest: dict[str, dict],
) -> tuple[list[dict] | None, list[str] | None, Counter]:
    """Перебрать записи диагнозов: (новый diag_result | None, аварии | None, счётчики).

    Делает две вещи разом, потому что обе переписывают одну и ту же строку:
    проставляет снимок редакции и вынимает из строки наши errors — diag_result
    уезжает в реплику движка и попадает агенту медчека как есть, а он прочитает
    нашу деградацию как факт о карте. Аварии уходят в done_cards.diag_errors
    в том же виде, что их пишет аудит: «<код МКБ>: <что упало>».

    Первый элемент — None, когда строку писать незачем. Счётчики считают записи,
    а не карты: expanded — развёрнутые, already — уже со снимком, missing —
    file_id, которого нет в справочнике (редакция ушла), no_guideline — диагнозы
    без клинрека, degraded — записи, из которых вынули аварию.
    """
    counts: Counter = Counter()
    entries = []
    degradation: list[str] = []
    changed = False
    for entry in diag_result or []:
        entry = dict(entry)
        file_id = entry.get("guideline_file_id")
        if not file_id:
            counts["no_guideline"] += 1
        elif entry.get("guideline_meta"):
            counts["already"] += 1
        elif file_id in manifest:
            entry["guideline_meta"] = manifest[file_id]
            counts["expanded"] += 1
            changed = True
        else:
            counts["missing"] += 1
        errors = entry.pop("errors", None)
        if errors:
            code = entry.get("icd_code") or "—"
            degradation.extend(f"{code}: {error}" for error in errors)
            counts["degraded"] += 1
            changed = True
        entries.append(entry)
    return (entries if changed else None), (degradation or None), counts


async def _run(storage, *, manifest: dict[str, dict], limit: int, batch: int,
               apply: bool) -> Counter:
    totals: Counter = Counter()
    after_id = ""
    seen = 0
    while True:
        size = batch if not limit else min(batch, limit - seen)
        if size <= 0:
            break
        rows = await storage.list_diag_results_to_backfill(limit=size, after_id=after_id)
        if not rows:
            break
        after_id = rows[-1]["id"]
        seen += len(rows)
        for row in rows:
            entries, degradation, counts = expand(row["diag_result"], manifest)
            totals += counts
            if entries is None:
                continue
            totals["cards"] += 1
            if apply:
                await storage.set_diag_result(
                    card_id=row["id"],
                    diag_json=json.dumps(entries, ensure_ascii=False),
                    diag_errors=degradation)
        print(f"  просмотрено карт: {seen}, к записи: {totals['cards']}")
    return totals


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Развернуть guideline_file_id в снимок редакции (dry-run без -y)")
    parser.add_argument("--limit", type=int, default=0,
                        help="обработать не больше стольких карт; 0 — все")
    parser.add_argument("--batch", type=int, default=500,
                        help="размер пачки на один SELECT")
    parser.add_argument("-y", dest="apply", action="store_true",
                        help="писать; без него ничего не меняется")
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit(f"--limit must not be negative, got {args.limit}")
    if args.batch <= 0:
        raise SystemExit(f"--batch must be positive, got {args.batch}")

    async with GuidelinesStorage() as guidelines:
        manifest = build_manifest_meta(await guidelines.all())
    if not manifest:
        raise SystemExit("справочник клинреков пуст — разворачивать нечем")
    print(f"Клинреков в справочнике: {len(manifest)}")

    async with DoneCardsStorage() as storage:
        totals = await _run(storage, manifest=manifest, limit=args.limit,
                            batch=args.batch, apply=args.apply)

    print(f"\nЗаписей диагнозов: развёрнуто {totals['expanded']}, "
          f"редакция ушла из справочника {totals['missing']}, "
          f"уже со снимком {totals['already']}, "
          f"без клинрека {totals['no_guideline']}, "
          f"вынуто аварий {totals['degraded']}")
    if args.apply:
        print(f"Карт изменено: {totals['cards']}")
    else:
        print(f"DRY-RUN: ничего не записано, изменилось бы карт: {totals['cards']}. "
              "Повторите с -y.")


if __name__ == "__main__":
    asyncio.run(main())
