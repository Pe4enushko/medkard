#!/usr/bin/env python3
"""
Проставить слепок правила в формальных замечаниях, записанных до слепков.

WHY THIS EXISTS: аудит теперь кладёт в каждое формальное замечание слепок
правила — rule_id, source, severity, source_ref, expectation, verified_at
(SNAPSHOT_FIELDS в storage/models/result.py). Движок «Искры» читает по ним
эталон для проверки ответов врача. У замечаний, записанных раньше, есть только
flag и короткий ярлык source, и эталона у них нет.

Скрипт проходит по таким замечаниям один раз и берёт слепок по флагу из
ТЕКУЩИХ каталогов — rules.json и deterministic_rules.json. Это текущая редакция правила, а не та, что действовала
при проверке: исторических версий каталога нет, а правила почти не меняются —
в основном добавляются новые.

Флаг ДИСПАНСЕРНОЕ_НАБЛЮДЕНИЕ_НЕ_ОТРАЖЕНО делят два правила — взрослое 168н и
детское 192н. Правило выбирается по возрасту пациента тем же patient_age, что
и в валидаторе; возраст не читается — замечание остаётся без слепка и
считается как ambiguous, чтобы его было видно.

Флаг, которого нет ни в одном каталоге (незаполненные поля шаблона, флаг
ушедшего правила), получает пустой слепок — ту же форму, что живой аудит пишет
синтетическим замечаниям. Противоречие НМУ (NMU_CODE_CONTRADICTION) с
2026-09-13 — правило service_name_matches_code детерминированного каталога.

Запускать из корня проекта (dry-run без -y):

    python scripts/hacks/backfill-rule-snapshot.py [--limit 100] [--batch 500] [-y]

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

from audit.deterministic.validator import _RULES_PATH as _DETERMINISTIC_PATH
from audit.formal_structure.validator import _ADULT_AGE, _RULES, _rule_snapshot
from parsers.json_parser import patient_age
from storage.done_cards_storage import DoneCardsStorage
from storage.models.result import SNAPSHOT_FIELDS

_EMPTY_SNAPSHOT = {key: "" for key in SNAPSHOT_FIELDS}

# Both catalogues write into the same formal_result, so a flag is looked up in both.
CATALOGUE: list[dict] = _RULES + json.loads(
    _DETERMINISTIC_PATH.read_text(encoding="utf-8"))["rules"]


def _pick_rule(candidates: list[dict], patient: dict) -> dict | None:
    """The one rule for a flag; among several, the one matching the patient's
    age group. None when the age is unknown or no candidate fits."""
    if len(candidates) == 1:
        return candidates[0]
    age = patient_age(patient or {})
    if age is None:
        return None
    group = "child" if age < _ADULT_AGE else "adult"
    fitting = [r for r in candidates if r.get("applies_to", {}).get("age_group", "all") in ("all", group)]
    return fitting[0] if len(fitting) == 1 else None


def snapshot(
    formal_result: list[dict], patient: dict, rules: list[dict],
) -> tuple[list[dict] | None, Counter]:
    """(new formal_result | None when nothing changed, counters).

    Counters count findings: snapshotted — got a rule snapshot, no_rule — got
    the empty one, already — carried a snapshot, ambiguous — shared flag and
    no usable age, left as is.
    """
    by_flag: dict[str, list[dict]] = {}
    for rule in rules:
        by_flag.setdefault(rule["flag_code"], []).append(rule)

    counts: Counter = Counter()
    out = []
    changed = False
    for finding in formal_result or []:
        if "rule_id" in finding:
            counts["already"] += 1
            out.append(finding)
            continue
        candidates = by_flag.get(finding.get("flag", ""), [])
        if not candidates:
            counts["no_rule"] += 1
            out.append({**finding, **_EMPTY_SNAPSHOT})
            changed = True
            continue
        rule = _pick_rule(candidates, patient)
        if rule is None:
            counts["ambiguous"] += 1
            out.append(finding)
            continue
        counts["snapshotted"] += 1
        out.append({**finding, **_rule_snapshot(rule)})
        changed = True
    return (out if changed else None), counts


async def _run(storage, *, rules: list[dict], limit: int, batch: int, apply: bool) -> Counter:
    totals: Counter = Counter()
    after_id = ""
    seen = 0
    while True:
        size = batch if not limit else min(batch, limit - seen)
        if size <= 0:
            break
        rows = await storage.list_formal_results_to_backfill(limit=size, after_id=after_id)
        if not rows:
            break
        after_id = rows[-1]["id"]
        seen += len(rows)
        for row in rows:
            findings, counts = snapshot(row["formal_result"], row.get("patient") or {}, rules)
            totals += counts
            if findings is None:
                continue
            totals["cards"] += 1
            if apply:
                await storage.set_formal_result(
                    card_id=row["id"], formal_json=json.dumps(findings, ensure_ascii=False))
        print(f"  просмотрено карт: {seen}, к записи: {totals['cards']}")
    return totals


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проставить слепок правила в старых формальных замечаниях (dry-run без -y)")
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

    print(f"Правил в каталогах: {len(CATALOGUE)}")
    async with DoneCardsStorage() as storage:
        totals = await _run(storage, rules=CATALOGUE, limit=args.limit,
                            batch=args.batch, apply=args.apply)

    print(f"\nЗамечаний: со слепком правила {totals['snapshotted']}, "
          f"без правила в каталоге {totals['no_rule']}, "
          f"уже со слепком {totals['already']}, "
          f"двусмысленный флаг без возраста {totals['ambiguous']}")
    if args.apply:
        print(f"Карт изменено: {totals['cards']}")
    else:
        print(f"DRY-RUN: ничего не записано, изменилось бы карт: {totals['cards']}. "
              "Повторите с -y.")


if __name__ == "__main__":
    asyncio.run(main())
