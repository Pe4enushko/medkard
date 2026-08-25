#!/usr/bin/env python3
"""Снимок детерминированного слоя аудита и сравнение двух снимков.

Зачем отдельно от e2e: у теста есть ожидаемый ответ, здесь его нет. Вопрос не
«прошло или упало», а «что изменилось между двумя состояниями кода».

Почему только детерминированный слой: тип визита, возраст, отбор правил и
сверка кода с наименованием — чистые функции. Их можно сравнивать точно и
бесплатно. Сработал ли чекер на выбранном правиле — вопрос к LLM, и его нельзя
сравнивать, пока не измерен собственный разброс прогонов.

Использование::

    # снимок из фикстуры или файла с картами
    python scripts/checks/deterministic-snapshot.py snapshot e2e/fixtures/eval_broken_cards/cases.json -o before.json

    # снимок из БД (POSTGRES_* из окружения)
    python scripts/checks/deterministic-snapshot.py snapshot --from-db --limit 500 -o after.json

    python scripts/checks/deterministic-snapshot.py diff before.json after.json
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from audit.formal_structure.validator import FormalValidator  # noqa: E402


def _age_reader():
    """Разбор возраста там, где он живёт в этой ревизии кода.

    Снимок обязан сниматься по обе стороны сравнения, включая ревизии, где
    общего `parsers.json_parser.patient_age` ещё не было.
    """
    try:
        from parsers.json_parser import patient_age

        return patient_age
    except ImportError:
        from audit.diagnosis.clinic_recs import _patient_age

        return _patient_age


_patient_age_fn = _age_reader()


def _guid(visit: dict[str, Any]) -> str:
    return str((visit.get("Прием") or {}).get("GUID") or "").lower()


def _cards_from_file(path: Path) -> Iterable[dict[str, Any]]:
    """Список визитов, обёртка `{cases: [{visit}]}` или один визит."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case["visit"] for case in data["cases"] if isinstance(case.get("visit"), dict)]
    if isinstance(data, dict) and isinstance(data.get("Прием"), dict):
        return [data]
    if isinstance(data, list):
        return [v for v in data if isinstance(v, dict)]
    raise SystemExit(f"{path}: не похоже на карты — ожидались список визитов или {{cases: [...]}}")


async def _cards_from_db(limit: int, org: str | None) -> list[dict[str, Any]]:
    from storage.done_cards_storage import DoneCardsStorage

    async with DoneCardsStorage() as store:
        async with store._pool.connection() as conn:  # noqa: SLF001 — снимок, не прод-путь
            sql = ("SELECT card_data FROM done_cards "
                   "WHERE card_data IS NOT NULL AND NOT ignored "
                   + ("AND organization_id = %(org)s " if org else "")
                   + "ORDER BY card_guid LIMIT %(limit)s")
            cur = await conn.execute(sql, {"org": org, "limit": limit})
            rows = await cur.fetchall()
    cards = []
    for row in rows:
        data = row["card_data"]
        cards.append(json.loads(data) if isinstance(data, str) else data)
    return cards


def _provenance(validator: FormalValidator) -> dict[str, Any]:
    """Чем снят снимок. Без этого сравнение двух файлов доказывает не то, что кажется.

    Стоило одного разбирательства: два снимка разошлись по возрасту на 65 картах,
    и объяснить это кодом не вышло — потому что нечем было проверить, на какой
    ревизии снят «до». Снимок обязан свидетельствовать о себе сам.
    """
    revision = "неизвестна"
    try:
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or revision
    except Exception:  # noqa: BLE001 — происхождение не обязано ронять снимок
        pass
    age_fn = _patient_age_fn
    return {
        "revision": revision,
        "src": str(Path(__file__).resolve().parent.parent.parent / "src"),
        "age_reader": f"{age_fn.__module__}.{age_fn.__name__}",
        "get_rules_arity": len(inspect.signature(validator.get_rules).parameters),
    }


async def _profile(visit: dict[str, Any], validator: FormalValidator) -> dict[str, Any]:
    """Всё, что решается кодом до обращения к модели."""
    visit_types = await validator.get_visit_types(visit)
    age = _patient_age_fn(visit.get("Пациент") or {})
    icd_codes = [
        str(d.get("КодМКБ") or "")
        for d in (visit.get("Диагнозы") or [])
        if isinstance(d, dict)
    ]
    # Число параметров get_rules менялось: снимок должен сниматься и на старой
    # ревизии, иначе сравнивать будет не с чем.
    accepted = len(inspect.signature(validator.get_rules).parameters)
    rules = validator.get_rules(*(visit_types, age, icd_codes, visit)[:accepted])
    contradiction = validator._check_nmu_keyword_contradiction(visit)  # noqa: SLF001
    return {
        "visit_types": sorted(t.name for t in visit_types),
        "age": age,
        "icd": sorted(c for c in icd_codes if c),
        "rules": sorted(r["flag_code"] for r in rules),
        "nmu_contradiction": bool(contradiction),
    }


async def _snapshot(cards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validator = FormalValidator()
    out: dict[str, Any] = {"_meta": _provenance(validator)}
    for i, visit in enumerate(cards):
        key = _guid(visit) or f"#{i}"
        out[key] = await _profile(visit, validator)
    return out


def _diff(before: dict[str, Any], after: dict[str, Any]) -> int:
    meta_before = before.pop("_meta", None)
    meta_after = after.pop("_meta", None)
    for label, meta in (("до ", meta_before), ("после", meta_after)):
        if meta is None:
            print(f"{label}: происхождение не записано — снимок снят старой версией скрипта")
        else:
            print(f"{label}: ревизия {meta.get('revision')}, возраст читает "
                  f"{meta.get('age_reader')}, get_rules({meta.get('get_rules_arity')} арг.)")
    if meta_before and meta_after and meta_before.get("revision") == meta_after.get("revision"):
        print("ВНИМАНИЕ: обе стороны сняты одной ревизией — сравнивать нечего\n")
    else:
        print()

    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    shared = sorted(set(before) & set(after))

    changed = []
    for key in shared:
        b, a = before[key], after[key]
        if b == a:
            continue
        entry: dict[str, Any] = {"card": key}
        for field in ("visit_types", "age", "icd", "nmu_contradiction"):
            if b.get(field) != a.get(field):
                entry[field] = f"{b.get(field)} → {a.get(field)}"
        gained = sorted(set(a["rules"]) - set(b["rules"]))
        lost = sorted(set(b["rules"]) - set(a["rules"]))
        if gained:
            entry["правил добавилось"] = gained
        if lost:
            entry["правил ушло"] = lost
        entry["правил всего"] = f"{len(b['rules'])} → {len(a['rules'])}"
        changed.append(entry)

    print(f"карт в снимках: было {len(before)}, стало {len(after)}, общих {len(shared)}")
    if only_before:
        print(f"пропали из снимка ({len(only_before)}): {', '.join(only_before[:10])}")
    if only_after:
        print(f"появились в снимке ({len(only_after)}): {', '.join(only_after[:10])}")
    print(f"карт с изменениями: {len(changed)} из {len(shared)}\n")

    for entry in changed:
        print(f"— {entry.pop('card')}")
        for field, value in entry.items():
            print(f"    {field}: {value}")

    if shared:
        rules_before = sum(len(before[k]["rules"]) for k in shared)
        rules_after = sum(len(after[k]["rules"]) for k in shared)
        print(f"\nправил на карту в среднем: {rules_before / len(shared):.1f} → "
              f"{rules_after / len(shared):.1f}")
    return 1 if changed or only_before or only_after else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="снять профиль карт")
    snap.add_argument("path", nargs="?", type=Path, help="файл с картами")
    snap.add_argument("--from-db", action="store_true", help="взять card_data из done_cards")
    snap.add_argument("--limit", type=int, default=500)
    snap.add_argument("--org", default=None)
    snap.add_argument("-o", "--out", type=Path, required=True)

    dif = sub.add_parser("diff", help="сравнить два снимка")
    dif.add_argument("before", type=Path)
    dif.add_argument("after", type=Path)

    args = parser.parse_args()

    if args.command == "diff":
        return _diff(json.loads(args.before.read_text(encoding="utf-8")),
                     json.loads(args.after.read_text(encoding="utf-8")))

    if args.from_db:
        cards = asyncio.run(_cards_from_db(args.limit, args.org))
    elif args.path:
        cards = list(_cards_from_file(args.path))
    else:
        raise SystemExit("нужен файл с картами или --from-db")

    snapshot = asyncio.run(_snapshot(cards))
    args.out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(f"{len(snapshot)} карт → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
