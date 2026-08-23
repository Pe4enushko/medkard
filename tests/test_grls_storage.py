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
from grls.match import MatchKind
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
        got, kind = await s.search_by_trade_name('"амоксиклав®"')
        assert kind is MatchKind.EXACT
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
        # sim DESC внутри одного уровня. Запрос намеренно не равен ни одной
        # записи: точное совпадение забрало бы одну строку и до сравнения
        # порядка дело не дошло бы.
        by_sim, kind = await s.search_by_trade_name("Амизола")
        assert kind is MatchKind.CONTAINS
        assert [r.reg_number for r in by_sim][:2] == ["ЛП-100001", "ЛП-100002"]
        # expires_at DESC NULLS FIRST: бессрочная (NULL) выше датированной при
        # одинаковых статусе и sim.
        by_expiry, kind = await s.search_by_trade_name("Кардиовин")
        assert kind is MatchKind.EXACT
        assert [r.reg_number for r in by_expiry][:2] == ["ЛП-100003", "ЛП-100004"]


async def test_similarity_threshold_guc_is_compatible_with_inn_fuzzy_threshold():
    # The `%` trigram operator is gated by the session's pg_trgm.similarity_threshold
    # GUC (default 0.3), ANDed with our explicit thresholds. The fuzzy tier
    # uses grls.match.FUZZY_THRESHOLD=0.6; if a prior stand
    # session raised the GUC above that, fuzzy INN matches silently return
    # nothing (no error) and the two INN tests above would fail confusingly.
    # This test turns that into a diagnosed failure with an explicit message.
    async with GrlsStorage() as s:
        async with s._pool.connection() as conn:
            cur = await conn.execute("SELECT current_setting('pg_trgm.similarity_threshold') AS v")
            row = await cur.fetchone()
            guc = float(row["v"])
        assert guc <= 0.6, (
            f"pg_trgm.similarity_threshold={guc} exceeds grls.match.FUZZY_THRESHOLD=0.6 "
            "on this session/DB — fuzzy INN search will silently drop matches below the GUC "
            "regardless of the explicit threshold. Reset the GUC (session or postgresql.conf) "
            "before trusting the INN search tests below."
        )


async def test_search_by_inn_composite_and_substance_filter():
    async with GrlsStorage() as s:
        await s.replace_all(_fixture_records(), _import())
        got, kind = await s.search_by_inn("амоксициллин + клавулановая кислота")
        assert kind is MatchKind.EXACT
        assert {r.reg_number for r in got} == {"ЛП-000001", "ЛП-000002", "ЛП-000003"}
        # «Амоксициллин» входит в составное МНН — это уровень CONTAINS, и раньше
        # он терялся: similarity 0.394 ниже порога 0.6, поиск по МНН отдавал
        # пусто, и запрос проваливался в торговые наименования и БАДы.
        part, kind = await s.search_by_inn("амоксициллин")
        assert kind is MatchKind.CONTAINS
        assert {r.reg_number for r in part} == {"ЛП-000001", "ЛП-000002", "ЛП-000003"}
        # Субстанция по-прежнему скрыта, пока её явно не попросили.
        assert all(not r.is_substance for r in part)
        with_substance, kind = await s.search_by_inn("амоксициллин", include_substances=True)
        assert kind is MatchKind.EXACT
        assert {r.reg_number for r in with_substance} == {"ФС-000001"}
        counts, revived = await s.inn_status_counts("амоксициллин+клавулановая кислота",
                                                    kind=MatchKind.EXACT)
        assert counts == {st.STATUS_ACTIVE: 1, st.STATUS_EXPIRED: 1, st.STATUS_ANNULLED: 1}
        assert revived == 0


async def test_search_by_inn_does_not_fuzzy_match_other_vitamin_suffix():
    vitamin_e = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-200001", trade_name="Витамин Е", inn_name="Витамин Е"))
    vitamin_d3 = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-200002", trade_name="Витамин D3", inn_name="Колекальциферол"))
    async with GrlsStorage() as s:
        await s.replace_all([vitamin_e, vitamin_d3],
                            GrlsImport(archive_name="test", registry_date=date(2026, 8, 17),
                                      status_counts={st.STATUS_ACTIVE: 2}))
        assert (await s.search_by_inn("Витамин D"))[0] == []
        assert (await s.search_by_inn("Витамин D3"))[0] == []
        for kind in MatchKind:
            assert (await s.inn_status_counts("Витамин D", kind=kind))[0] == {}


async def test_grls_norm_parity_with_python():
    samples = ['  "ЭФКУРИЯ®"  ', "«Кей Джи Пи»", "Ёлкин\tчай", "Аспирин™ 500", "~", "Bayer's", "A B"]
    async with GrlsStorage() as s:
        async with s._pool.connection() as conn:
            for text in samples:
                cur = await conn.execute("SELECT grls_norm(%(t)s) AS v", {"t": text})
                row = await cur.fetchone()
                assert (row["v"] or "") == normalize_query(text), text


async def test_short_discriminator_must_match_as_a_whole_word():
    """«Витамин В1» не должен находиться как «Витамин В12».

    Тиамин и цианокобаламин — разные вещества, а строки похожи на 0.750.
    Прежняя охрана искала различитель ПОДСТРОКОЙ, и «в1» спокойно находился
    внутри «в12».
    """
    b1 = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-300001", trade_name="Тиамин", inn_name="Витамин В1"))
    b12 = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-300002", trade_name="Цианокобаламин", inn_name="Витамин В12"))
    async with GrlsStorage() as s:
        await s.replace_all([b1, b12], GrlsImport(
            archive_name="test", registry_date=date(2026, 8, 17),
            status_counts={st.STATUS_ACTIVE: 2}))

        found, kind = await s.search_by_inn("Витамин В1")
        assert kind is MatchKind.EXACT
        assert {r.reg_number for r in found} == {"ЛП-300001"}

        found, kind = await s.search_by_inn("Витамин В12")
        assert kind is MatchKind.EXACT
        assert {r.reg_number for r in found} == {"ЛП-300002"}


async def test_salt_form_is_found_by_the_bare_inn():
    """«Метопролол» обязан находить «Метопролола сукцинат».

    similarity этой пары — 0.455, ниже порога 0.6: раньше поиск по МНН отдавал
    пусто и запрос уходил в торговые наименования, а оттуда в реестр БАД.
    """
    salt = build_record(st.STATUS_ACTIVE, sample_row(
        reg_number="ЛП-400001", trade_name="Беталок ЗОК", inn_name="Метопролола сукцинат"))
    async with GrlsStorage() as s:
        await s.replace_all([salt], GrlsImport(
            archive_name="test", registry_date=date(2026, 8, 17),
            status_counts={st.STATUS_ACTIVE: 1}))

        found, kind = await s.search_by_inn("Метопролол")
        assert kind is MatchKind.CONTAINS
        assert {r.reg_number for r in found} == {"ЛП-400001"}

        counts, _ = await s.inn_status_counts("Метопролол", kind=kind)
        assert counts == {st.STATUS_ACTIVE: 1}, (
            "счётчики обязаны считаться тем же условием, что и список: иначе "
            "«Регистраций: N» разъедется с показанными записями"
        )


async def test_counts_report_registrations_live_at_the_visit_date():
    """РУ, умершее после визита, на дату визита действовало.

    Ветка торговых наименований это учитывала всегда, ветка МНН — нет.
    """
    expired_after = build_record(st.STATUS_EXPIRED, sample_row(
        reg_number="ЛП-500001", trade_name="Тестовин", inn_name="тестамол",
        expires_at="31.12.2025"))
    async with GrlsStorage() as s:
        await s.replace_all([expired_after], GrlsImport(
            archive_name="test", registry_date=date(2026, 8, 17),
            status_counts={st.STATUS_EXPIRED: 1}))

        counts, revived = await s.inn_status_counts(
            "тестамол", kind=MatchKind.EXACT, on=date(2025, 3, 10))
        assert counts == {st.STATUS_EXPIRED: 1}
        assert revived == 1

        _, revived_now = await s.inn_status_counts("тестамол", kind=MatchKind.EXACT)
        assert revived_now == 0, "без даты визита оживлять нечего"
