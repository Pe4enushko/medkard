#!/usr/bin/env python3
"""
Привести уже лежащие карты Алёнки к нашей форме врача.

WHY THIS EXISTS: Алёнка прислала врача одним объектом в верхнем блоке
`Врач = {GUID, FIO, SPECIALIZATION}` вместо `Прием.Врач` + `Прием.Врач_код` из
технички. Живой поток теперь перекладывает его на входе (parsers/doctor.py),
а карты, попавшие в done_cards раньше, лежат как пришли, и для pull API,
персональных отчётов и doctor_user_map движка у них врача нет.

Скрипт проходит по таким картам один раз и прогоняет card_data через тот же
normalize_doctor, что и живой поток — один код на оба пути. Выдуманный врач
от костыля демо-врачей в блоке Прием при этом перезаписывается настоящим:
штамп верхний блок не трогал, и настоящий врач в карте цел.

Запускать из корня проекта (dry-run без -y):

    python scripts/hacks/backfill-alenka-doctors.py [--limit 100] [--batch 500] [-y]

Options:
    --limit    Обработать не больше стольких карт; 0 — все
    --batch    Размер пачки на один SELECT (по умолчанию 500)
    -y         Писать. Без него скрипт только считает, что сделал бы.

UPDATE поднимает updated_at (триггер done_cards_set_updated_at, миграция 022),
поэтому реплика движка заберёт изменённые карты обычным инкрементальным синком.
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

from parsers.doctor import normalize_doctor
from storage.done_cards_storage import DoneCardsStorage


async def _run(storage, *, limit: int, batch: int, apply: bool) -> Counter:
    totals: Counter = Counter()
    after_id = ""
    seen = 0
    while True:
        size = batch if not limit else min(batch, limit - seen)
        if size <= 0:
            break
        rows = await storage.list_cards_with_top_level_doctor(limit=size, after_id=after_id)
        if not rows:
            break
        after_id = rows[-1]["id"]
        seen += len(rows)
        for row in rows:
            card = normalize_doctor(row["card_data"])
            totals["cards"] += 1
            if apply:
                await storage.set_card_data(
                    card_id=row["id"], card_json=json.dumps(card, ensure_ascii=False))
        print(f"  просмотрено карт: {seen}, к записи: {totals['cards']}")
    return totals


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Привести карты Алёнки к нашей форме врача (dry-run без -y)")
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

    async with DoneCardsStorage() as storage:
        totals = await _run(storage, limit=args.limit, batch=args.batch, apply=args.apply)

    if args.apply:
        print(f"Карт изменено: {totals['cards']}")
    else:
        print(f"DRY-RUN: ничего не записано, изменилось бы карт: {totals['cards']}. "
              "Повторите с -y.")


if __name__ == "__main__":
    asyncio.run(main())
