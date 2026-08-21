"""Интеграционные тесты storage.grls_storage.GrlsStorage.

Требует Postgres (.env) с применённой миграцией 027. Запускается на стенде —
на dev-машине нет доступа к БД. Тесты подменяют содержимое grls_registry
целиком (replace_all) — не гонять на БД с боевыми данными без последующего
повторного импорта.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from grls import status as st
from grls.normalize import normalize_query
from grls.parser import build_record
from tests.grls_fixtures import sample_row
from storage.grls_storage import GrlsStorage
from storage.models.grls_record import GrlsImport


def _fixture_records():
    return [
        build_record(st.STATUS_EXPIRED, sample_row(reg_number="ЛП-000001", trade_name="Амоксиклав®",
                                                   inn_name="амоксициллин+клавулановая кислота", expires_at="31.12.2025")),
        build_record(st.STATUS_ACTIVE, sample_row(reg_number="ЛП-000002", trade_name="АМОКСИКЛАВ",
                                                  inn_name="амоксициллин+клавулановая кислота")),
        build_record(st.STATUS_ANNULLED, sample_row(reg_number="ЛП-000003", trade_name="амоксиклав",
                                                    inn_name="амоксициллин+клавулановая кислота", annulled_at="14.02.2024")),
        build_record(st.STATUS_ACTIVE, sample_row(reg_number="ФС-000001", trade_name="Амоксициллин",
                                                  inn_name="амоксициллин",
                                                  forms_raw="субстанция-порошок, ~, 25 кг - мешки - Не указано;")),
    ]


def _import():
    return GrlsImport(archive_name="test", registry_date=date(2026, 8, 17),
                      status_counts={st.STATUS_ACTIVE: 2, st.STATUS_EXPIRED: 1, st.STATUS_ANNULLED: 1},
                      skipped_files=["Изменённый"])


async def test_replace_all_and_latest_import():
    async with GrlsStorage() as s:
        n = await s.replace_all(_fixture_records(), _import())
        assert n == 4
        assert await s.count() == 4
        imp = await s.latest_import()
        assert imp is not None and imp.registry_date == date(2026, 8, 17)
        # idempotent: second run replaces, not appends
        assert await s.replace_all(_fixture_records(), _import()) == 4
        assert await s.count() == 4


async def test_search_by_trade_name_orders_by_status_and_ignores_case_and_marks():
    async with GrlsStorage() as s:
        await s.replace_all(_fixture_records(), _import())
        got = await s.search_by_trade_name('"амоксиклав®"')
        assert [r.status for r in got][:2] == [st.STATUS_ACTIVE, st.STATUS_EXPIRED]
        assert got[-1].status == st.STATUS_ANNULLED


async def test_search_by_trade_name_orders_by_sim_then_expires_at_within_same_status():
    # Isolate the 2nd/3rd ORDER BY terms: all rows below share one status, so
    # rank alone cannot determine order — sim DESC and expires_at DESC NULLS
    # FIRST must both be doing real work, or this suite stays green on a
    # regression that flips/drops either term. Synthetic names chosen so the
    # two sub-cases don't trigram-overlap with each other (verified sim=0.0
    # between the two roots, so neither query pulls in the other pair's rows).
    exact = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-100001", trade_name="Амизолам", inn_name="тестамол"))
    partial = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-100002", trade_name="Амизолам форте", inn_name="тестамол"))
    perpetual = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-100003", trade_name="Кардиовин", inn_name="тестамол", expires_at=""))
    dated = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-100004", trade_name="Кардиовин", inn_name="тестамол", expires_at="31.12.2030"))
    async with GrlsStorage() as s:
        await s.replace_all([exact, partial, perpetual, dated],
                            GrlsImport(archive_name="test", registry_date=date(2026, 8, 17),
                                      status_counts={st.STATUS_ACTIVE: 4}))
        # sim DESC: an exact normalized match ranks above a longer partial match, same status.
        by_sim = await s.search_by_trade_name("Амизолам", threshold=0.4)
        assert [r.reg_number for r in by_sim][:2] == ["ЛП-100001", "ЛП-100002"]
        # expires_at DESC NULLS FIRST: perpetual (NULL) outranks a dated registration, same status+sim.
        by_expiry = await s.search_by_trade_name("Кардиовин", threshold=0.4)
        assert [r.reg_number for r in by_expiry][:2] == ["ЛП-100003", "ЛП-100004"]


async def test_similarity_threshold_guc_is_compatible_with_inn_fuzzy_threshold():
    # The `%` trigram operator is gated by the session's pg_trgm.similarity_threshold
    # GUC (default 0.3), ANDed with our explicit thresholds. search_by_inn /
    # inn_status_counts hardcode _INN_FUZZY_THRESHOLD=0.6; if a prior stand
    # session raised the GUC above that, fuzzy INN matches silently return
    # nothing (no error) and the two INN tests above would fail confusingly.
    # This test turns that into a diagnosed failure with an explicit message.
    async with GrlsStorage() as s:
        async with s._pool.connection() as conn:
            cur = await conn.execute("SELECT current_setting('pg_trgm.similarity_threshold') AS v")
            row = await cur.fetchone()
            guc = float(row["v"])
        assert guc <= 0.6, (
            f"pg_trgm.similarity_threshold={guc} exceeds GrlsStorage._INN_FUZZY_THRESHOLD=0.6 "
            "on this session/DB — fuzzy INN search will silently drop matches below the GUC "
            "regardless of the explicit threshold. Reset the GUC (session or postgresql.conf) "
            "before trusting the INN search tests below."
        )


async def test_search_by_inn_composite_and_substance_filter():
    async with GrlsStorage() as s:
        await s.replace_all(_fixture_records(), _import())
        got = await s.search_by_inn("амоксициллин + клавулановая кислота")
        assert {r.reg_number for r in got} == {"ЛП-000001", "ЛП-000002", "ЛП-000003"}
        assert await s.search_by_inn("амоксициллин") == []          # substance hidden
        assert len(await s.search_by_inn("амоксициллин", include_substances=True)) >= 1
        counts = await s.inn_status_counts("амоксициллин+клавулановая кислота")
        assert counts == {st.STATUS_ACTIVE: 1, st.STATUS_EXPIRED: 1, st.STATUS_ANNULLED: 1}


async def test_grls_norm_parity_with_python():
    samples = ['  "ЭФКУРИЯ®"  ', "«Кей Джи Пи»", "Ёлкин\tчай", "Аспирин™ 500", "~", "Bayer's", "A B"]
    async with GrlsStorage() as s:
        async with s._pool.connection() as conn:
            for text in samples:
                cur = await conn.execute("SELECT grls_norm(%(t)s) AS v", {"t": text})
                row = await cur.fetchone()
                assert (row["v"] or "") == normalize_query(text), text
