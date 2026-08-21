"""Доступ к Postgres для эвалов.

Разделение обязанностей простое: харнесс СТРОИТ рабочую таблицу, методы её
только ЧИТАЮТ. Гарантия не на честном слове — после сборки сессия переводится в
read-only (`default_transaction_read_only = on`), и дальше Postgres сам
запрещает и DDL, и запись в постоянные таблицы. Методы получают на руки
ReadOnlyDB, в котором никакого способа что-то записать просто нет.

Таблица — ВРЕМЕННАЯ (pg_temp): живёт внутри сессии и исчезает при выходе, в БД
не остаётся ничего. Это позволяет гонять эвал против рабочей базы medkard.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

TABLE = "eval_docs"


def pg_params_from_env() -> dict[str, str]:
    """Параметры соединения по отдельности, а не URI.

    URI собирать нельзя: пароль может содержать '/', ':' или '@', и тогда
    postgresql://… разбирается не туда — вплоть до попытки резолвить логин как
    хост. Ошибка при этом выглядит как сетевая, а не как «плохой пароль».
    """
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "medkard"),
        "user": os.getenv("POSTGRES_USER", ""),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def mask(params: dict[str, str]) -> str:
    return f"{params['user']}@{params['host']}:{params['port']}/{params['dbname']}"


class ReadOnlyDB:
    """Всё, что видит способ поиска: выполнить SELECT и забрать строки.

    Ни INSERT, ни DDL здесь нет намеренно — не потому, что «не принято», а
    потому что к моменту создания объекта сессия уже read-only и сервер их не
    примет.
    """

    def __init__(self, conn: Any, table: str = TABLE) -> None:
        self._conn = conn
        self.table = table

    def fetch_one(self, sql: str, params: dict | None = None) -> tuple | None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.fetchone()

    def fetch_all(self, sql: str, params: dict | None = None) -> list[tuple]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.fetchall()

    def assert_read_only(self) -> None:
        row = self.fetch_one("SHOW transaction_read_only")
        if not row or row[0] != "on":
            raise RuntimeError(
                "сессия не в read-only: методы поиска обязаны работать только на чтение"
            )


class EvalWorkspace:
    """Строит временную таблицу с корпусом и запечатывает сессию.

    Порядок жёсткий: build() → seal(). После seal() записи невозможны, и наружу
    выдаётся ReadOnlyDB для методов.
    """

    def __init__(self, conn: Any, table: str = TABLE) -> None:
        self._conn = conn
        self.table = table
        self._sealed = False

    def build(self, docs_norm: Sequence[str], names_norm: Sequence[str],
              rest: Sequence[str], vectors: Sequence[str], dim: int) -> None:
        if self._sealed:
            raise RuntimeError("сессия уже переведена в read-only")
        with self._conn.cursor() as cur:
            for ext in ("pg_trgm", "vector"):
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                    self._conn.commit()
                except Exception as e:  # noqa: BLE001 — сообщение важнее типа
                    self._conn.rollback()
                    raise SystemExit(
                        f"нет расширения {ext} и его не создать: {e}\n"
                        "эвал считает лексику и вектор средствами Postgres — без них никак"
                    )
            cur.execute(f"""
                CREATE TEMP TABLE {self.table} (
                    id         int PRIMARY KEY,
                    doc_norm   text,
                    names_norm text,
                    rest       text,
                    rest_tsv   tsvector GENERATED ALWAYS AS
                               (to_tsvector('russian', coalesce(rest, ''))) STORED,
                    emb        vector({dim})
                ) ON COMMIT PRESERVE ROWS
            """)
            self._assert_temp(cur)
            cur.executemany(
                f"INSERT INTO {self.table} (id, doc_norm, names_norm, rest, emb) "
                "VALUES (%s, %s, %s, %s, %s)",
                list(zip(range(len(docs_norm)), docs_norm, names_norm, rest, vectors)),
            )
            # Текстовые индексы — как будут в проде (спека engine §4.5).
            cur.execute(f"CREATE INDEX ON {self.table} USING GIN (doc_norm gin_trgm_ops)")
            cur.execute(f"CREATE INDEX ON {self.table} USING GIN (names_norm gin_trgm_ops)")
            cur.execute(f"CREATE INDEX ON {self.table} USING GIN (rest_tsv)")
            cur.execute(f"ANALYZE {self.table}")
            self._conn.commit()

    def _assert_temp(self, cur: Any) -> None:
        """Дешёвая страховка от опечатки в имени: таблица обязана быть временной."""
        cur.execute(
            "SELECT c.relpersistence, n.nspname FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.oid = %s::regclass", (self.table,))
        persistence, schema = cur.fetchone()
        if persistence != "t" or not schema.startswith("pg_temp"):
            raise RuntimeError(
                f"{self.table} оказалась не временной ({persistence}, {schema}) — "
                "эвал не имеет права писать в постоянные таблицы"
            )

    def seal(self) -> ReadOnlyDB:
        """Дальше — только чтение. Проверяется у сервера, а не декларируется."""
        self._conn.commit()
        with self._conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
        self._conn.commit()
        self._sealed = True
        db = ReadOnlyDB(self._conn, self.table)
        db.assert_read_only()
        return db
