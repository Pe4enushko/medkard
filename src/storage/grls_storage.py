"""GrlsStorage — async psycopg3 interface for grls_registry / grls_imports (migration 027)."""
from __future__ import annotations

from psycopg import sql
from psycopg.types.json import Jsonb

from grls.normalize import normalize_query
from grls.status import STATUS_RANK
from storage.base import BaseStorage
from storage.models.grls_record import GrlsImport, GrlsRecord

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
_INN_FUZZY_THRESHOLD = 0.6

_RANK_CASE = sql.SQL("CASE status {} ELSE 9 END").format(
    sql.SQL(" ").join(sql.SQL("WHEN {} THEN {}").format(sql.Literal(s), sql.Literal(r))
                      for s, r in STATUS_RANK.items()))
_ORDER = sql.SQL("ORDER BY {} ASC, sim DESC, expires_at DESC NULLS FIRST").format(_RANK_CASE)


def _row_to_record(row: dict) -> GrlsRecord:
    return GrlsRecord(**{k: row[k] for k in ("id", "imported_at", *_COLS)})


def _record_params(rec: GrlsRecord) -> dict:
    return {c: getattr(rec, c) for c in _COLS}


class GrlsStorage(BaseStorage):
    """Usage::
        async with GrlsStorage() as storage:
            hits = await storage.search_by_trade_name("амоксиклав")
    """

    async def replace_all(self, records: list[GrlsRecord], imp: GrlsImport) -> int:
        """Full replacement in one transaction (DELETE, not TRUNCATE — readers are not blocked)."""
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
                await conn.execute(
                    """
                    INSERT INTO grls_imports (archive_name, registry_date, status_counts, skipped_files)
                    VALUES (%(archive_name)s, %(registry_date)s, %(status_counts)s, %(skipped_files)s)
                    """,
                    {"archive_name": imp.archive_name, "registry_date": imp.registry_date,
                     "status_counts": Jsonb(imp.status_counts), "skipped_files": Jsonb(imp.skipped_files)},
                )
        return inserted

    async def search_by_trade_name(self, query: str, *, threshold: float = 0.85, limit: int = 6,
                                   include_substances: bool = False) -> list[GrlsRecord]:
        q = normalize_query(query)
        if not q:
            return []
        stmt = sql.SQL(
            "SELECT " + _SELECT_COLS + ", similarity(grls_norm(trade_name), %(q)s) AS sim "
            "FROM grls_registry "
            "WHERE grls_norm(trade_name) %% %(q)s "
            "  AND similarity(grls_norm(trade_name), %(q)s) >= %(threshold)s "
            "  AND (%(inc)s OR NOT is_substance) {} LIMIT %(limit)s"
        ).format(_ORDER)
        async with self._pool.connection() as conn:
            cur = await conn.execute(stmt, {"q": q, "threshold": threshold, "inc": include_substances, "limit": limit})
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    async def search_by_inn(self, query: str, *, limit: int = 20,
                            include_substances: bool = False) -> list[GrlsRecord]:
        q = normalize_query(query)
        if not q:
            return []
        stmt = sql.SQL(
            "SELECT " + _SELECT_COLS + ", similarity(grls_norm(inn_name), %(q)s) AS sim "
            "FROM grls_registry "
            "WHERE (grls_norm(inn_name) = %(q)s "
            "       OR (grls_norm(inn_name) %% %(q)s AND similarity(grls_norm(inn_name), %(q)s) >= %(fuzzy)s)) "
            "  AND (%(inc)s OR NOT is_substance) {} LIMIT %(limit)s"
        ).format(_ORDER)
        async with self._pool.connection() as conn:
            cur = await conn.execute(stmt, {"q": q, "fuzzy": _INN_FUZZY_THRESHOLD, "inc": include_substances, "limit": limit})
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    async def inn_status_counts(self, query: str, *, include_substances: bool = False) -> dict[str, int]:
        q = normalize_query(query)
        if not q:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT status, count(*) AS n FROM grls_registry
                WHERE (grls_norm(inn_name) = %(q)s
                       OR (grls_norm(inn_name) %% %(q)s AND similarity(grls_norm(inn_name), %(q)s) >= %(fuzzy)s))
                  AND (%(inc)s OR NOT is_substance)
                GROUP BY status
                """,
                {"q": q, "fuzzy": _INN_FUZZY_THRESHOLD, "inc": include_substances},
            )
            rows = await cur.fetchall()
        return {r["status"]: r["n"] for r in rows}

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
