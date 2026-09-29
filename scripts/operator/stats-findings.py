#!/usr/bin/env python3
"""
Сводка замечаний аудита за интервал: какие флаги, сколько и на скольких картах.

Зачем: после выкатки надо видеть, что именно новые правила выставили на бою и в
каком объёме, не сочиняя запрос каждый раз. `scripts/checks/metrics.py` считает
карты и токены по дням, но про флаги не знает ничего.

Считается по `formal_result` — туда пишут оба валидатора, формальный и
детерминированный, и слепок правила лежит в самой находке, поэтому степень и
код правила берутся из неё, а не подбираются по каталогу.

Запуск из корня проекта:

    python scripts/operator/stats-findings.py                      # 7 дней
    python scripts/operator/stats-findings.py --days 1 --org MDS
    python scripts/operator/stats-findings.py --from 24.09.2026 --to 29.09.2026
    python scripts/operator/stats-findings.py --days 30 --detailed --bucket week
    python scripts/operator/stats-findings.py --days 3 --detailed --send remoteclaude

Опции:
    --days N            интервал назад от сегодня (по умолчанию 7)
    --from / --to       границы интервала, ДД.ММ.ГГГГ или ГГГГ-ММ-ДД
    --org ИМЯ           только эта организация (Alenka, MDS)
    --by audit|visit    по времени аудита (по умолчанию) или по дате приёма
    --detailed          выгрузка по периодам и флагам в logs/stats-findings-*.csv
    --bucket day|week|month  период детальной выгрузки (по умолчанию day)
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
    export_path, interval, offer_scp, print_table, write_csv,
)

from RAG.retrieval.vector_store import close_pool  # noqa: E402
from storage.base import BaseStorage  # noqa: E402

# Время аудита или дата приёма — вопрос не праздный: карту за 1 сентября могли
# проверить 29-го, и после выкатки интересно именно «что выставил новый код», то
# есть время аудита. Для разговора с клиникой наоборот нужна дата приёма.
_WHEN = {
    "audit": "coalesce(done_cards.finished_at, done_cards.updated_at)",
    "visit": VISIT_DATE,
}


class _FindingsReader(BaseStorage):
    async def totals(self, *, when: str, start, end, org: str | None) -> dict:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT count(*)                                              AS cards,
                       count(*) FILTER (WHERE jsonb_array_length(
                           {FINDINGS_JSON}) > 0)        AS cards_with_findings,
                       coalesce(sum(jsonb_array_length(
                           {FINDINGS_JSON})), 0)        AS findings
                FROM done_cards
                LEFT JOIN organizations ON organizations.id = done_cards.organization_id
                WHERE {_WHEN[when]}::date BETWEEN %(start)s AND %(end)s
                  AND done_cards.ignored = FALSE
                  AND (%(org)s = '' OR organizations.name = %(org)s)
                """,
                {"start": start, "end": end, "org": org or ""},
            )
            return await cur.fetchone() or {}

    async def by_flag(self, *, when: str, start, end, org: str | None) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT finding ->> 'flag'                     AS flag,
                       coalesce(finding ->> 'severity', '—')  AS severity,
                       count(*)                               AS findings,
                       count(DISTINCT done_cards.id)          AS cards
                FROM done_cards
                LEFT JOIN organizations ON organizations.id = done_cards.organization_id
                CROSS JOIN LATERAL jsonb_array_elements({FINDINGS_JSON}) AS finding
                WHERE {_WHEN[when]}::date BETWEEN %(start)s AND %(end)s
                  AND done_cards.ignored = FALSE
                  AND (%(org)s = '' OR organizations.name = %(org)s)
                GROUP BY 1, 2
                ORDER BY findings DESC
                """,
                {"start": start, "end": end, "org": org or ""},
            )
            return list(await cur.fetchall())

    async def by_bucket_and_flag(
        self, *, when: str, bucket: str, start, end, org: str | None
    ) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT {bucket_expression(_WHEN[when], bucket)}  AS period,
                       coalesce(organizations.name, '—')         AS organization,
                       finding ->> 'flag'                        AS flag,
                       coalesce(finding ->> 'severity', '—')     AS severity,
                       coalesce(finding ->> 'rule_id', '')       AS rule_id,
                       count(*)                                  AS findings,
                       count(DISTINCT done_cards.id)              AS cards
                FROM done_cards
                LEFT JOIN organizations ON organizations.id = done_cards.organization_id
                CROSS JOIN LATERAL jsonb_array_elements({FINDINGS_JSON}) AS finding
                WHERE {_WHEN[when]}::date BETWEEN %(start)s AND %(end)s
                  AND done_cards.ignored = FALSE
                  AND (%(org)s = '' OR organizations.name = %(org)s)
                GROUP BY 1, 2, 3, 4, 5
                ORDER BY period, findings DESC
                """,
                {"start": start, "end": end, "org": org or ""},
            )
            return list(await cur.fetchall())


_SUMMARY_COLUMNS = (
    ("flag", "Флаг", 42),
    ("severity", "Степень", 14),
    ("findings", "Замечаний", 10),
    ("cards", "Карт", 7),
    ("share", "% карт", 7),
)
_DETAILED_FIELDS = ("period", "organization", "flag", "severity", "rule_id", "findings", "cards")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сводка замечаний аудита за интервал",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_interval_arguments(parser)
    parser.add_argument("--org", default=None, help="только эта организация")
    parser.add_argument("--by", choices=tuple(_WHEN), default="audit",
                        help="по времени аудита или по дате приёма")
    parser.add_argument("--detailed", action="store_true",
                        help="выгрузка по периодам и флагам в CSV")
    parser.add_argument("--bucket", choices=("day", "week", "month"), default="day",
                        help="период детальной выгрузки")
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
        async with _FindingsReader() as reader:
            totals = await reader.totals(when=args.by, start=start, end=end, org=args.org)
            rows = await reader.by_flag(when=args.by, start=start, end=end, org=args.org)
            detailed = (
                await reader.by_bucket_and_flag(
                    when=args.by, bucket=args.bucket, start=start, end=end, org=args.org)
                if args.detailed else []
            )
    finally:
        await close_pool()

    scope = args.org or "все организации"
    print(f"\nЗамечания за {start:%d.%m.%Y}–{end:%d.%m.%Y} ({scope}), "
          f"по времени {'аудита' if args.by == 'audit' else 'приёма'}")
    cards = int(totals.get("cards") or 0)
    if not cards:
        print("  карт за интервал не найдено")
        return 0
    with_findings = int(totals.get("cards_with_findings") or 0)
    findings = int(totals.get("findings") or 0)
    print(f"  карт: {cards}   с замечаниями: {with_findings} ({with_findings / cards:.1%})   "
          f"замечаний: {findings}   на карту: {findings / cards:.2f}\n")

    for row in rows:
        row["share"] = f"{int(row['cards']) / cards:.1%}"
    print_table(rows, _SUMMARY_COLUMNS)

    if args.detailed:
        path = write_csv(detailed, _DETAILED_FIELDS,
                         export_path("findings", start, end, args.org))
        offer_scp(path, args.send)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
