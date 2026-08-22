"""
BaseStorage — shared async connection pool lifecycle for all storage classes.
"""

import os
import asyncio
from typing import Any

import psycopg.rows
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

load_dotenv()

_shared_pool: AsyncConnectionPool | None = None
_shared_pool_lock = asyncio.Lock()


def _conninfo() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


async def _get_shared_pool() -> AsyncConnectionPool:
    global _shared_pool
    async with _shared_pool_lock:
        if _shared_pool is None or _shared_pool.closed:
            _shared_pool = AsyncConnectionPool(
                conninfo=_conninfo(),
                min_size=int(os.environ.get("POSTGRES_POOL_MIN_SIZE", "1")),
                max_size=int(os.environ.get("POSTGRES_POOL_MAX_SIZE", "5")),
                open=False,
                kwargs={"row_factory": psycopg.rows.dict_row},
            )
            await _shared_pool.open()
    return _shared_pool


async def reopen_shared_pool() -> AsyncConnectionPool:
    """Drop the shared psycopg pool and create a fresh one.

    This is used after PostgreSQL/VPN drops a connection while the process keeps
    running. The next operation should not be forced to reuse the stale pool.
    """
    global _shared_pool
    async with _shared_pool_lock:
        if _shared_pool is not None and not _shared_pool.closed:
            await _shared_pool.close()
        _shared_pool = None
    return await _get_shared_pool()


class BaseStorage:
    """Async context-manager backed by a shared module-level connection pool.

    All subclasses share one pool so concurrent instantiations don't multiply
    open connections. Subclasses that need per-connection setup (codec
    registration etc.) should override _configure() — called once on __aenter__
    rather than per connection, since the pool is shared.
    """

    async def __aenter__(self) -> "BaseStorage":
        self._pool = await _get_shared_pool()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass  # pool is shared; do not close it here
