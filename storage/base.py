"""
BaseStorage — shared async connection pool lifecycle for all storage classes.
"""

import os
from typing import Any, Callable, Coroutine

import psycopg
import psycopg.rows
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

load_dotenv()


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


class BaseStorage:
    """Async context-manager that owns an AsyncConnectionPool.

    Subclasses may override _configure() to run per-connection setup
    (e.g. registering pgvector codecs).
    """

    async def _configure(self, conn: psycopg.AsyncConnection) -> None:
        """Called once per new connection. Override to add codec registrations."""

    async def __aenter__(self) -> "BaseStorage":
        self._pool: AsyncConnectionPool = AsyncConnectionPool(
            conninfo=_conninfo(),
            open=False,
            configure=self._configure,
            kwargs={"row_factory": psycopg.rows.dict_row},
        )
        await self._pool.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._pool.close()
