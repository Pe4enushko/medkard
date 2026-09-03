#!/usr/bin/env python3
"""Замер детерминистичных правил на выгрузках 1С.

Отвечает на вопросы, которые чтением кода не решаются: как часто срабатывает
правило, сколько карт уходит в OTHER и почему, опознаётся ли шаблон записи,
находят ли синонимические ряды 168н показатели в живом тексте осмотра.

Выгрузки лежат вне репозитория (персональный контур), путь передаётся аргументом.
Формат у организаций разный: ``{"appointments": [...]}`` либо голый список.

Запуск::

    python scripts/checks/eval-deterministic.py ~/projects/data_snapshots
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.deterministic import DeterministicValidator  # noqa: E402
from audit.deterministic.indicators import groups_in_text, required_groups  # noqa: E402
from audit.deterministic.validator import _RULES_PATH  # noqa: E402
from audit.formal_structure.required_fields import (  # noqa: E402
    _filled_labels,
    _load,
    _match_template,
)
from audit.formal_structure.validator import (  # noqa: E402
    NMU_RE,
    FormalValidator,
    VisitType,
    _VISIT_TYPE_RULE_KEY,
)
from parsers.json_parser import patient_age  # noqa: E402


def _org(path: str) -> str:
    name = os.path.basename(path)
    return name.split("_")[2] if name.count("_") > 2 else "unknown"


def cards(snapshots: Path):
    for path in sorted(glob.glob(os.path.join(snapshots, "*.json"))):
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"!! не прочитан {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue
        items = doc.get("appointments") if isinstance(doc, dict) else doc
        for card in items or []:
            if isinstance(card, dict):
                yield _org(path), card


def _all_rules_enabled() -> str:
    """Копия файла правил со снятыми выключателями — меряем и отложенные."""
    doc = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    for rule in doc["rules"]:
        # номер медкарты не мерить: поля нет в выгрузке, правило сработает везде
        if rule["rule_id"] != "protocol_card_number_present":
            rule["enabled"] = True
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(doc, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


def _service_codes(service: dict) -> list[str]:
    return [
        token.strip()
        for raw in service.values() if raw
        for token in str(raw).split()
        if NMU_RE.fullmatch(token.strip())
    ]


async def run(snapshots: Path, show: int) -> None:
    formal = FormalValidator()
    deterministic = DeterministicValidator(_all_rules_enabled())

    total = collections.Counter()
    types = collections.Counter()
    template = collections.Counter()
    fired = collections.Counter()
    other_services = collections.Counter()
    other_empty = collections.Counter()
    groups = collections.Counter()
    with_text = 0

    for org, card in cards(snapshots):
        total[org] += 1
        visit_types = await formal.get_visit_types(card)
        keys = {_VISIT_TYPE_RULE_KEY[t] for t in visit_types}
        for key in keys:
            types[(org, key)] += 1

        labels = _filled_labels(card.get("ДанныеОсмотра") or [])
        template[(org, bool(_match_template(labels, _load())))] += 1

        for finding in await deterministic.validate(card):
            fired[(org, finding["flag"])] += 1

        if visit_types == {VisitType.OTHER}:
            other_empty[(org, bool(card.get("ДанныеОсмотра")))] += 1
            for service in (card.get("Услуги") or []):
                if isinstance(service, dict):
                    codes = _service_codes(service)
                    other_services[(org, codes[0] if codes else "без кода",
                                    (service.get("Наименование") or "")[:48])] += 1

        text = " ".join(
            str(item.get("Значение") or "")
            for item in (card.get("ДанныеОсмотра") or []) if isinstance(item, dict)
        )
        if text:
            with_text += 1
            groups.update(groups_in_text(text))

    print(f"карт: {dict(total)}, всего {sum(total.values())}")
    for org in sorted(total):
        n = total[org]
        print(f"\n=== {org} ({n} карт)")
        print("  виды приёма (карта может иметь несколько):")
        for (o, key), c in sorted(types.items(), key=lambda x: -x[1]):
            if o == org:
                print(f"    {key:28} {c:6}  {c * 100 / n:5.1f}%")
        ok = template[(org, True)]
        print(f"  шаблон записи опознан: {ok} ({ok * 100 / n:.1f}%)")
        print("  срабатывания правил:")
        for (o, flag), c in sorted(fired.items(), key=lambda x: -x[1]):
            if o == org:
                print(f"    {flag:38} {c:6}  {c * 100 / n:5.1f}%")
        print(f"  OTHER с пустым осмотром: {other_empty[(org, False)]}, "
              f"с заполненным: {other_empty[(org, True)]}")

    print("\n=== услуги, из-за которых карта уходит в OTHER")
    for (org, code, name), c in other_services.most_common(show):
        print(f"   {org:8} {c:5}  {code:18} {name}")

    print(f"\n=== синонимические ряды 168н на живом тексте ({with_text} записей с осмотром)")
    for group, c in groups.most_common(show):
        print(f"   {group:28} {c:6}  {c * 100 / with_text:5.1f}%")
    print(f"   рядов без единого совпадения: {52 - len(groups)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, help="каталог с выгрузками 1С")
    parser.add_argument("--show", type=int, default=20, help="сколько строк печатать в топах")
    args = parser.parse_args(argv)
    if not args.snapshots.is_dir():
        parser.error(f"каталог не найден: {args.snapshots}")
    asyncio.run(run(args.snapshots, args.show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
