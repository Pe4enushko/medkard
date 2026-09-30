#!/usr/bin/env python3
"""
Сводка сломанных карт за интервал: сколько, у кого и на чём падает.

Сломанная карта — та, где аудит упал с исключением: `done_cards.broken = TRUE`,
текст падения в `stacktrace` (миграция 012, там же вью `broken_cards`).
После выкатки это первое, что надо смотреть: пропавший файл `resources/` или
новое правило с опечаткой проявляются именно так — картами в `broken`, а не
неверными замечаниями.

Группировка по последней строке стектрейса: это класс исключения и сообщение,
то есть причина. Полные стектрейсы — в детальной выгрузке.

Запуск из корня проекта:

    python scripts/stats/stats-broken.py                       # 7 дней
    python scripts/stats/stats-broken.py --days 1
    python scripts/stats/stats-broken.py --from 24.09.2026 --to 29.09.2026 --org MDS
    python scripts/stats/stats-broken.py --days 3 --detailed --send remoteclaude

Опции:
    --days N            интервал назад от сегодня (по умолчанию 7)
    --from / --to       границы интервала, ДД.ММ.ГГГГ или ГГГГ-ММ-ДД
    --org ИМЯ           только эта организация (Alenka, MDS)
    --by audit|visit    по времени аудита (по умолчанию) или по дате приёма
    --detailed          выгрузка карт со стектрейсами в logs/stats-broken-*.csv
    --bucket day|week|month  период в сводке по дням (по умолчанию day)
    --send АЛИАС        отправить выгрузку по scp на этот ssh-алиас
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats_common import (  # noqa: E402
    FINDINGS_JSON, VISIT_DATE, add_interval_arguments, bucket_expression,
    export_path, interval, offer_scp, print_table, target_database, write_csv,
)

from RAG.retrieval.vector_store import close_pool  # noqa: E402
from storage.base import BaseStorage  # noqa: E402

_WHEN = {
    "audit": "coalesce(done_cards.finished_at, done_cards.updated_at)",
    "visit": VISIT_DATE,
}

# Последняя непустая строка стектрейса — «ValueError: ...», то есть причина.
# split_part по обратному порядку строк в SQL не выразить коротко, поэтому режем
# в питоне: строк тут сотни, не миллионы.
def _reason(stacktrace: str | None) -> str:
    if not stacktrace:
        return "— (стектрейс пуст)"
    lines = [line.strip() for line in stacktrace.splitlines() if line.strip()]
    return lines[-1][:120] if lines else "— (стектрейс пуст)"


class _BrokenReader(BaseStorage):
    async def per_period(self, *, when: str, bucket: str, start, end, org: str | None) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT {bucket_expression(_WHEN[when], bucket)}         AS period,
                       coalesce(organizations.name, '—')                AS organization,
                       count(*)                                          AS cards,
                       count(*) FILTER (WHERE done_cards.broken)         AS broken
                FROM done_cards
                LEFT JOIN organizations ON organizations.id = done_cards.organization_id
                WHERE {_WHEN[when]}::date BETWEEN %(start)s AND %(end)s
                  AND done_cards.ignored = FALSE
                  AND (%(org)s = '' OR organizations.name = %(org)s)
                GROUP BY 1, 2
                ORDER BY period, organization
                """,
                {"start": start, "end": end, "org": org or ""},
            )
            return list(await cur.fetchall())

    async def broken_cards(self, *, when: str, start, end, org: str | None) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT done_cards.card_guid                              AS card_guid,
                       coalesce(organizations.name, '—')                 AS organization,
                       {VISIT_DATE}                                  AS visit_date,
                       done_cards.started_at                             AS started_at,
                       done_cards.stacktrace                             AS stacktrace
                FROM done_cards
                LEFT JOIN organizations ON organizations.id = done_cards.organization_id
                WHERE done_cards.broken = TRUE
                  AND {_WHEN[when]}::date BETWEEN %(start)s AND %(end)s
                  AND (%(org)s = '' OR organizations.name = %(org)s)
                ORDER BY done_cards.started_at DESC
                """,
                {"start": start, "end": end, "org": org or ""},
            )
            return list(await cur.fetchall())


_PERIOD_COLUMNS = (
    ("period", "Период", 12),
    ("organization", "Организация", 16),
    ("cards", "Карт", 8),
    ("broken", "Сломано", 8),
    ("share", "%", 7),
)
_REASON_COLUMNS = (
    ("reason", "Причина (последняя строка стектрейса)", 78),
    ("cards", "Карт", 6),
)
_DETAILED_FIELDS = ("card_guid", "organization", "visit_date", "started_at", "reason", "stacktrace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сводка сломанных карт за интервал")
    add_interval_arguments(parser)
    parser.add_argument("--org", default=None, help="только эта организация")
    parser.add_argument("--by", choices=tuple(_WHEN), default="audit",
                        help="по времени аудита или по дате приёма")
    parser.add_argument("--detailed", action="store_true",
                        help="выгрузка карт со стектрейсами в CSV")
    parser.add_argument("--bucket", choices=("day", "week", "month"), default="day",
                        help="период в сводке")
    parser.add_argument("--send", metavar="АЛИАС", default=None,
                        help="отправить выгрузку по scp на этот ssh-алиас")
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    start, end = interval(args)
    if args.send and not args.detailed:
        print("--send без --detailed: отправлять нечего", file=sys.stderr)
        return 2

    try:
        async with _BrokenReader() as reader:
            periods = await reader.per_period(
                when=args.by, bucket=args.bucket, start=start, end=end, org=args.org)
            cards = await reader.broken_cards(
                when=args.by, start=start, end=end, org=args.org)
    finally:
        await close_pool()

    scope = args.org or "все организации"
    print(f"\nБаза: {target_database()}")
    print(f"Сломанные карты за {start:%d.%m.%Y}–{end:%d.%m.%Y} ({scope}), "
          f"по времени {'аудита' if args.by == 'audit' else 'приёма'}")

    total = sum(int(row["cards"]) for row in periods)
    broken = sum(int(row["broken"]) for row in periods)
    if not total:
        print("  карт за интервал не найдено")
        return 0
    print(f"  карт: {total}   сломано: {broken} ({broken / total:.2%})\n")

    for row in periods:
        row["share"] = f"{int(row['broken']) / int(row['cards']):.1%}" if row["cards"] else "—"
    print_table(periods, _PERIOD_COLUMNS)

    if cards:
        reasons: dict[str, int] = {}
        for row in cards:
            row["reason"] = _reason(row.get("stacktrace"))
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        print("\nПричины:")
        print_table(
            [{"reason": reason, "cards": count}
             for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])],
            _REASON_COLUMNS,
        )

    if args.detailed:
        path = write_csv(cards, _DETAILED_FIELDS, export_path("broken", start, end, args.org))
        offer_scp(path, args.send)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
