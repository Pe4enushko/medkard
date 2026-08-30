#!/usr/bin/env python3
"""
Diagnose the made-up-doctor crutch: is it on, and which visits still have no doctor.

WHY THIS EXISTS: the stamp lives on ONE code path — `POST /visits/push`
(api/routes/visits.py → api/demo_doctors.stamp). Cards that arrive the other
way, through the nightly 1C pull (scripts/audit-one-c-period.py → AuditPipeline),
never pass it and stay without a doctor. Nothing anywhere says so out loud: the
switch is on, the file is filled in, and the personal reports are still empty.
This script says so, with the dates.

Read-only: it writes nothing, so it is safe on prod at any time.

Run from project root:

    python scripts/hacks/check-demo-doctors.py [ORG] [--days N] [--limit N]

Options:
    ORG        Organization as named in the organizations table. Default: the
               one DEMO_DOCTOR_STAMP_ORG names.
    --days     Only look at visits this many days back from today; 0 — the
               whole history. Same meaning as in audit-one-c-period.py.
    --limit    How many dates to print (default 20; 0 — all of them).

Exit code is 1 when anything is wrong, so cron can shout.

THIS IS A CRUTCH AND IS BUILT TO BE REMOVED, together with the module and the
other scripts/hacks/*-demo-doctors.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

_TEMPLATE_NAME = "Фамилия Имя Отчество"


# ── проверки, не требующие БД ────────────────────────────────────────────────

def check_switch(stamp_org: str, org: str, known_orgs: list[str]) -> list[str]:
    """Проблемы с переключателем DEMO_DOCTOR_STAMP_ORG. Пустой список — всё в порядке."""
    stamp_org = (stamp_org or "").strip()
    if not stamp_org:
        return ["DEMO_DOCTOR_STAMP_ORG пуст — штамп выключен, врача не получит ни одна карта"]
    folded = {o.casefold() for o in known_orgs}
    if stamp_org.casefold() not in folded:
        return [
            f"DEMO_DOCTOR_STAMP_ORG={stamp_org!r} — такой организации нет в базе "
            f"(есть: {', '.join(known_orgs)}). Штамп не сработает ни на одной карте"
        ]
    if stamp_org.casefold() != (org or "").casefold():
        return [
            f"DEMO_DOCTOR_STAMP_ORG={stamp_org!r}, а проверяем {org!r} — "
            f"на карты {org} штамп не встаёт"
        ]
    return []


def check_doctors_file(path: str) -> tuple[list[dict], list[str]]:
    """Список врачей и проблемы с файлом."""
    problems: list[str] = []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"файл врачей {path} не прочитан ({exc}) — штамп молча ничего не делает"]
    if not isinstance(raw, list):
        return [], [f"в файле врачей {path} ожидался массив"]

    doctors = [{"code": str(d["code"]), "name": str(d["name"])} for d in raw
               if isinstance(d, dict) and d.get("code") and d.get("name")]
    if len(doctors) != len(raw):
        problems.append(
            f"в файле врачей {len(raw)} записей, а годных {len(doctors)} — "
            f"остальные без code или name и в штамп не попадут"
        )
    if not doctors:
        problems.append(f"файл врачей {path} пуст — карта уходит без врача, это видно только в логе")
        return doctors, problems

    if any(d["name"] == _TEMPLATE_NAME for d in doctors):
        problems.append(
            f"в файле врачей остался шаблон «{_TEMPLATE_NAME}» — впишите настоящие ФИО: "
            f"движок сопоставляет врачей по ФИО и сведёт одинаковые в одного"
        )
    names = [d["name"] for d in doctors]
    if len(set(names)) != len(names):
        problems.append("в файле врачей повторяются ФИО — движок сопоставляет врачей по ФИО")
    codes = [d["code"] for d in doctors]
    if len(set(codes)) != len(codes):
        problems.append("в файле врачей повторяются коды — на разных врачей встанет один код")
    return doctors, problems


def explain_missing(pushed: int, pulled: int) -> str:
    """Что делать с картами без врача — по тому, каким путём они пришли."""
    if not pushed and not pulled:
        return ""
    lines: list[str] = []
    if pulled:
        lines.append(
            f"{pulled} карт(ы) без врача пришли НЕ через пуш (pushed_at пуст) — это ночной "
            f"пул 1С (scripts/audit-one-c-period.py). Штамп висит только на POST /visits/push, "
            f"этот путь его не проходит, и так будет каждую ночь"
        )
    if pushed:
        lines.append(
            f"{pushed} карт(ы) без врача пришли через пуш — значит в момент пуша штамп был "
            f"выключен (DEMO_DOCTOR_STAMP_ORG пуст или называл другую клинику) либо файл "
            f"врачей не читался"
        )
    lines.append(
        "Дозаполнить: python scripts/hacks/backfill-demo-doctors.py <ORG> --days N -y — "
        "и ставить его в крон следом за ночным аудитом, пока 1С не шлёт врача"
    )
    return "\n".join(lines)


def since_date(days: int | None, today: date) -> date | None:
    """Граница периода, как в backfill-demo-doctors.py: 0/None — вся история."""
    if not days:
        return None
    if days < 0:
        raise SystemExit(f"--days must not be negative, got {days}")
    return today - timedelta(days=days)


# ── чтение базы ──────────────────────────────────────────────────────────────

_CARDS = """
    SELECT
        count(*)                                                        AS total,
        count(*) FILTER (WHERE has_doctor)                              AS with_doctor,
        count(*) FILTER (WHERE NOT has_doctor AND pushed_at IS NOT NULL) AS missing_pushed,
        count(*) FILTER (WHERE NOT has_doctor AND pushed_at IS NULL)     AS missing_pulled
    FROM (
        SELECT COALESCE(card_data -> 'Прием' ->> 'Врач_код', '') <> '' AS has_doctor,
               pushed_at
        FROM done_cards
        WHERE organization_id = %(org_id)s::uuid
          AND card_data IS NOT NULL
          AND jsonb_typeof(card_data -> 'Прием') = 'object'
          AND (%(since)s::date IS NULL
               OR medkard_visit_date(card_data -> 'Прием' ->> 'DATE') >= %(since)s::date)
    ) t
"""

_DATES = """
    SELECT medkard_visit_date(card_data -> 'Прием' ->> 'DATE')  AS visit_date,
           count(*)                                             AS cards,
           count(*) FILTER (WHERE pushed_at IS NOT NULL)         AS pushed
    FROM done_cards
    WHERE organization_id = %(org_id)s::uuid
      AND card_data IS NOT NULL
      AND jsonb_typeof(card_data -> 'Прием') = 'object'
      AND COALESCE(card_data -> 'Прием' ->> 'Врач_код', '') = ''
      AND (%(since)s::date IS NULL
           OR medkard_visit_date(card_data -> 'Прием' ->> 'DATE') >= %(since)s::date)
    GROUP BY 1
    ORDER BY 1 DESC NULLS LAST
"""

_CODES = """
    SELECT card_data -> 'Прием' ->> 'Врач_код' AS code, count(*) AS cards
    FROM done_cards
    WHERE organization_id = %(org_id)s::uuid
      AND card_data IS NOT NULL
      AND COALESCE(card_data -> 'Прием' ->> 'Врач_код', '') <> ''
      AND (%(since)s::date IS NULL
           OR medkard_visit_date(card_data -> 'Прием' ->> 'DATE') >= %(since)s::date)
    GROUP BY 1
    ORDER BY 2 DESC
"""


async def _read(storage_cls, org_id: str, since: date | None) -> dict[str, Any]:
    params = {"org_id": org_id, "since": since}
    async with storage_cls() as store:
        async with store._pool.connection() as conn:      # noqa: SLF001 — hack-скрипт, свой SQL
            cur = await conn.execute(_CARDS, params)
            counts = await cur.fetchone()
            cur = await conn.execute(_DATES, params)
            dates = await cur.fetchall()
            cur = await conn.execute(_CODES, params)
            codes = await cur.fetchall()
    return {"counts": counts, "dates": dates, "codes": codes}


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверить костыль выдуманных врачей: включён ли и где карты без врача")
    parser.add_argument("org", nargs="?", help="организация; по умолчанию — из DEMO_DOCTOR_STAMP_ORG")
    parser.add_argument("--days", type=int, default=0, help="сколько дней назад смотреть; 0 — вся история")
    parser.add_argument("--limit", type=int, default=20, help="сколько дат печатать; 0 — все")
    args = parser.parse_args()

    from storage.base import BaseStorage
    from storage.organizations_storage import OrganizationsStorage
    from api.demo_doctors import doctors_path

    stamp_org = os.getenv("DEMO_DOCTOR_STAMP_ORG", "").strip()
    org = args.org or stamp_org
    if not org:
        print("Не указана организация и DEMO_DOCTOR_STAMP_ORG пуст: "
              "укажите её аргументом или заполните .env", file=sys.stderr)
        return 1

    since = since_date(args.days, date.today())
    problems: list[str] = []

    async with OrganizationsStorage() as orgs:
        async with orgs._pool.connection() as conn:       # noqa: SLF001 — hack-скрипт, свой SQL
            cur = await conn.execute("SELECT name FROM organizations ORDER BY name")
            known_orgs = [row["name"] for row in await cur.fetchall()]
        try:
            org_id = await orgs.get_id_by_name(org)
        except Exception as exc:                          # noqa: BLE001 — диагностика
            raise SystemExit(f"организация {org!r} не найдена: {exc}")

    data = await _read(BaseStorage, org_id, since)

    print(f"Клиника: {org}")
    print(f"Период: {'с ' + since.isoformat() if since else 'вся история'}")
    print()

    print("— переключатель —")
    switch_problems = check_switch(stamp_org, org, known_orgs)
    problems += switch_problems
    for p in switch_problems:
        print(f"  ПРОБЛЕМА: {p}")
    if not switch_problems:
        print(f"  DEMO_DOCTOR_STAMP_ORG={stamp_org} — включён на эту клинику")

    print("— файл врачей —")
    path = doctors_path()
    doctors, file_problems = check_doctors_file(path)
    problems += file_problems
    print(f"  {path}: {len(doctors)} врач(ей)")
    for p in file_problems:
        print(f"  ПРОБЛЕМА: {p}")

    counts = data["counts"]
    missing = counts["missing_pushed"] + counts["missing_pulled"]
    print("— карты —")
    print(f"  всего с блоком «Прием»: {counts['total']}")
    print(f"  с врачом:               {counts['with_doctor']}")
    print(f"  без врача:              {missing}"
          f" (пуш: {counts['missing_pushed']}, ночной пул: {counts['missing_pulled']})")

    our_codes = {d["code"] for d in doctors}
    ours = sum(r["cards"] for r in data["codes"] if r["code"] in our_codes)
    alien = sum(r["cards"] for r in data["codes"] if r["code"] not in our_codes)
    if data["codes"]:
        print(f"  из них с кодом из файла врачей: {ours}, с каким-то другим: {alien}")
        if alien:
            print("  чужие коды — это либо врач от 1С (тогда костыль пора снимать, "
                  "docs/visits-api.md), либо старый файл врачей")

    if missing:
        print("— даты карт без врача —")
        rows = data["dates"] if not args.limit else data["dates"][:args.limit]
        for row in rows:
            when = row["visit_date"].isoformat() if row["visit_date"] else "дата не разобрана"
            print(f"  {when}: {row['cards']} (из них пушем: {row['pushed']})")
        if args.limit and len(data["dates"]) > args.limit:
            print(f"  … ещё {len(data['dates']) - args.limit} дат(ы), покажет --limit 0")
        print()
        print(explain_missing(counts["missing_pushed"], counts["missing_pulled"]))
        problems.append(f"{missing} карт(ы) без врача")

    print()
    if problems:
        print(f"ИТОГ: проблем — {len(problems)}")
        return 1
    print("ИТОГ: костыль включён, врач есть на каждой карте")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
