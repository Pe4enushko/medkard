"""GrlsStorage — async psycopg3 interface for grls_registry / grls_imports.

Поиск идёт по хранимым нормализованным колонкам `inn_norm` / `trade_norm`
(миграция 029), а не по выражению `grls_norm(...)`. Причина в замере
`docs/grls-search-cost-2026-08-23.md`: индекс по выражению спасает только те
запросы, которые планировщик сумел через него провести, а любой оставшийся
пересчитывал нормализацию на всех 39 тыс. строк — 98 % стоимости такого
запроса приходилось на translate+regexp_replace, а не на поиск.
"""
from __future__ import annotations

import logging
from datetime import date

from psycopg import sql
from psycopg.types.json import Jsonb

from grls.match import (FUZZY_THRESHOLD, MIN_CONTAINED_LEN, MatchKind,
                        discriminator_tokens, like_pattern)
from grls.normalize import normalize_query
from grls.status import LIVE_STATUSES, STATUS_RANK
from storage.base import BaseStorage
from storage.models.grls_record import GrlsImport, GrlsRecord

logger = logging.getLogger(__name__)

_COLS = ("status", "reg_number", "registered_at", "expires_at", "annulled_at", "holder",
         "holder_country", "trade_name", "inn_name", "forms", "forms_raw", "dosage_forms",
         "dispensing", "is_substance", "production_stages", "normative_docs", "pharm_group",
         "is_vital", "narcotic_list", "is_orphan", "row_hash")
_SELECT_COLS = "id, imported_at, " + ", ".join(_COLS)
_INSERT_SQL = (
    f"INSERT INTO grls_registry ({', '.join(_COLS)}) VALUES ("
    + ", ".join(f"%({c})s" for c in _COLS)
    + ") ON CONFLICT (row_hash) DO NOTHING"
)
_BATCH = 1000
_TRADE_FUZZY_THRESHOLD = 0.85

_RANK_CASE = sql.SQL("CASE status {} ELSE 9 END").format(
    sql.SQL(" ").join(sql.SQL("WHEN {} THEN {}").format(sql.Literal(s), sql.Literal(r))
                      for s, r in STATUS_RANK.items()))
_ORDER = sql.SQL("ORDER BY {} ASC, sim DESC, expires_at DESC NULLS FIRST").format(_RANK_CASE)

# Различители обязаны совпасть ОТДЕЛЬНЫМ словом: сравнение по подстроке
# пропускало «в1» внутрь «в12», то есть тиамин находился как цианокобаламин.
_GUARD = sql.SQL(
    "NOT EXISTS (SELECT 1 FROM unnest(%(tokens)s::text[]) AS token "
    "            WHERE NOT (token = ANY(string_to_array({col}, ' '))))"
)
# Вхождение в обе стороны: врач пишет и короче реестра («Метопролол» при
# «Метопролола сукцинат»), и длиннее («Левотироксин натрия» при «Левотироксин»).
_CONTAINS = sql.SQL(
    "((length(%(q)s) >= {min_len} AND {col} LIKE %(like)s) "
    " OR (length({col}) >= {min_len} AND position({col} in %(q)s) > 0))"
)
# `%%` дополнительно ограничен GUC pg_trgm.similarity_threshold (по умолчанию
# 0.3) — действующая отсечка равна max(GUC, явного порога ниже).
_FUZZY = sql.SQL("({col} %% %(q)s AND similarity({col}, %(q)s) >= %(fuzzy)s)")


def _tier_predicate(column: sql.SQL, kind: MatchKind) -> sql.Composed:
    """Условие одного уровня. Зеркало grls.match.classify — правьте вместе."""
    if kind is MatchKind.EXACT:
        return sql.SQL("{col} = %(q)s").format(col=column)
    body = (_CONTAINS if kind is MatchKind.CONTAINS else _FUZZY).format(
        col=column, min_len=sql.Literal(MIN_CONTAINED_LEN)
    )
    return sql.SQL("{body} AND {guard}").format(
        body=body, guard=_GUARD.format(col=column)
    )


def _row_to_record(row: dict) -> GrlsRecord:
    return GrlsRecord(**{k: row[k] for k in ("id", "imported_at", *_COLS)})


def _record_params(rec: GrlsRecord) -> dict:
    return {c: getattr(rec, c) for c in _COLS}


def _short_discriminator_tokens(query: str) -> list[str]:
    """Tokens too short for safe trigram fuzzy matching must survive literally."""
    return [
        token
        for token in normalize_query(query).split()
        if 0 < len(token) <= _SHORT_DISCRIMINATOR_MAX_LEN
        and any(ch.isalpha() or ch.isdigit() for ch in token)
    ]


class GrlsStorage(BaseStorage):
    """Usage::
        async with GrlsStorage() as storage:
            hits = await storage.search_by_trade_name("амоксиклав")
    """

    async def replace_all(self, records: list[GrlsRecord], imp: GrlsImport) -> int:
        """Full replacement in one transaction (DELETE, not TRUNCATE — readers are not blocked).

        Returns the resulting row count in grls_registry after the insert —
        i.e. len(records) minus rows dropped by ON CONFLICT(row_hash) DO
        NOTHING (cross-sheet duplicates are expected in the export). This is
        NOT necessarily len(records); a mismatch is logged, not silent.
        """
        inserted = 0
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM grls_registry")
                async with conn.cursor() as cur:
                    for i in range(0, len(records), _BATCH):
                        batch = records[i:i + _BATCH]
                        await cur.executemany(_INSERT_SQL, [_record_params(r) for r in batch])
                cur2 = await conn.execute("SELECT count(*) AS n FROM grls_registry")
                inserted = (await cur2.fetchone())["n"]
                if inserted != len(records):
                    logger.info("GRLS replace_all: %d records in, %d rows after dedup (row_hash conflicts)",
                                len(records), inserted)
                await conn.execute(
                    """
                    INSERT INTO grls_imports (archive_name, registry_date, status_counts, skipped_files)
                    VALUES (%(archive_name)s, %(registry_date)s, %(status_counts)s, %(skipped_files)s)
                    """,
                    {"archive_name": imp.archive_name, "registry_date": imp.registry_date,
                     "status_counts": Jsonb(imp.status_counts), "skipped_files": Jsonb(imp.skipped_files)},
                )
        return inserted

    async def search_by_inn(self, query: str, *, limit: int = 20,
                            include_substances: bool = False,
                            ) -> tuple[list[GrlsRecord], MatchKind | None]:
        """Найти МНН по уровням: точное → вхождение → триграммы.

        Возвращает записи и уровень, на котором они нашлись. Уровень обязан
        доехать до ответа: нечёткое совпадение — это «похоже», а не «это оно».
        """
        return await self._search(sql.SQL("inn_norm"), query,
                                  limit=limit, include_substances=include_substances,
                                  fuzzy=FUZZY_THRESHOLD)

    async def search_by_trade_name(self, query: str, *, limit: int = 6,
                                   include_substances: bool = False,
                                   ) -> tuple[list[GrlsRecord], MatchKind | None]:
        """То же по торговому наименованию; триграммный порог тут строже."""
        return await self._search(sql.SQL("trade_norm"), query,
                                  limit=limit, include_substances=include_substances,
                                  fuzzy=_TRADE_FUZZY_THRESHOLD)

    async def _search(self, column: sql.SQL, query: str, *, limit: int,
                      include_substances: bool, fuzzy: float,
                      ) -> tuple[list[GrlsRecord], MatchKind | None]:
        q = normalize_query(query)
        if not q:
            return [], None
        params = {"q": q, "like": like_pattern(q), "fuzzy": fuzzy,
                  "tokens": discriminator_tokens(q),
                  "inc": include_substances, "limit": limit}
        for kind in MatchKind:
            stmt = sql.SQL(
                "SELECT " + _SELECT_COLS + ", similarity({col}, %(q)s) AS sim "
                "FROM grls_registry WHERE {pred} AND (%(inc)s OR NOT is_substance) "
                "{order} LIMIT %(limit)s"
            ).format(col=column, pred=_tier_predicate(column, kind), order=_ORDER)
            async with self._pool.connection() as conn:
                cur = await conn.execute(stmt, params)
                rows = await cur.fetchall()
            if rows:
                return [_row_to_record(r) for r in rows], kind
        return [], None

    async def inn_status_counts(self, query: str, *, kind: MatchKind, on: date | None = None,
                                include_substances: bool = False) -> tuple[dict[str, int], int]:
        """Регистрации по статусам ТЕМ ЖЕ условием, каким нашлись записи.

        Иначе «Регистраций: N» разъедется с показанным списком: счёт по одному
        правилу, список по другому.

        Второе значение — сколько мёртвых сегодня РУ были действительны на дату
        визита. Ветка торговых наименований это учитывала всегда
        (`format_record(..., lookup.on)`), ветка МНН — нет.
        """
        q = normalize_query(query)
        if not q:
            return {}, 0
        column = sql.SQL("inn_norm")
        stmt = sql.SQL(
            "SELECT status, count(*) AS n, "
            "       count(*) FILTER (WHERE %(on)s::date IS NOT NULL "
            "                          AND COALESCE(annulled_at, expires_at) >= %(on)s::date"
            "                       ) AS valid_at_visit "
            "FROM grls_registry "
            "WHERE {pred} AND (%(inc)s OR NOT is_substance) GROUP BY status"
        ).format(pred=_tier_predicate(column, kind))
        async with self._pool.connection() as conn:
            cur = await conn.execute(stmt, {"q": q, "like": like_pattern(q),
                                            "fuzzy": FUZZY_THRESHOLD,
                                            "tokens": discriminator_tokens(q),
                                            "on": on, "inc": include_substances})
            rows = await cur.fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        revived = sum(r["valid_at_visit"] for r in rows if r["status"] not in LIVE_STATUSES)
        return counts, revived

    async def latest_import(self) -> GrlsImport | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, archive_name, registry_date, status_counts, skipped_files, imported_at "
                "FROM grls_imports ORDER BY id DESC LIMIT 1")
            row = await cur.fetchone()
        if not row:
            return None
        return GrlsImport(id=row["id"], archive_name=row["archive_name"], registry_date=row["registry_date"],
                          status_counts=row["status_counts"], skipped_files=row["skipped_files"],
                          imported_at=row["imported_at"])

    async def count(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM grls_registry")
            return (await cur.fetchone())["n"]
