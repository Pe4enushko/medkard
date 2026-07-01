"""
Integration tests for storage.api_keys_storage.ApiKeysStorage — hits the
real configured Postgres against the api_keys / api_key_organizations
tables, cleaning up every key it creates.

Fixtures here are function-scoped (not module-scoped): pytest.ini only sets
asyncio_mode = auto with no asyncio_default_fixture_loop_scope, so each test
gets its own event loop — a module-scoped async fixture would be bound to
whichever loop first created it and hang when a later test's (different)
loop tries to reuse it.
"""

from __future__ import annotations

import uuid

import pytest
from dotenv import load_dotenv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from storage.api_keys_storage import ApiKeysStorage, hash_key
from storage.organizations_storage import OrganizationsStorage


@pytest.fixture
async def alenka_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("Alenka")


@pytest.fixture
async def mds_org_id() -> str:
    async with OrganizationsStorage() as organizations:
        return await organizations.get_id_by_name("MDS")


def _unique_key() -> str:
    return f"medkard_test_{uuid.uuid4().hex}"


async def test_create_key_requires_at_least_one_org():
    async with ApiKeysStorage() as api_keys:
        with pytest.raises(ValueError):
            await api_keys.create_key("test", _unique_key(), [])


async def test_created_key_is_authorized_for_its_scoped_org(alenka_org_id: str):
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("test", raw_key, [alenka_org_id])
        try:
            assert await api_keys.is_key_authorized_for_org(raw_key, alenka_org_id)
        finally:
            await api_keys.revoke_key(key_id)


async def test_key_is_not_authorized_for_unscoped_org(alenka_org_id: str, mds_org_id: str):
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("test", raw_key, [alenka_org_id])
        try:
            assert not await api_keys.is_key_authorized_for_org(raw_key, mds_org_id)
        finally:
            await api_keys.revoke_key(key_id)


async def test_key_scoped_to_multiple_orgs_authorizes_both(alenka_org_id: str, mds_org_id: str):
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("test", raw_key, [alenka_org_id, mds_org_id])
        try:
            assert await api_keys.is_key_authorized_for_org(raw_key, alenka_org_id)
            assert await api_keys.is_key_authorized_for_org(raw_key, mds_org_id)
        finally:
            await api_keys.revoke_key(key_id)


async def test_wrong_key_is_invalid(alenka_org_id: str):
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("test", raw_key, [alenka_org_id])
        try:
            assert not await api_keys.is_valid_key(_unique_key())
        finally:
            await api_keys.revoke_key(key_id)


async def test_revoked_key_is_invalid(alenka_org_id: str):
    raw_key = _unique_key()
    async with ApiKeysStorage() as api_keys:
        key_id = await api_keys.create_key("test", raw_key, [alenka_org_id])
        await api_keys.revoke_key(key_id)
        assert not await api_keys.is_valid_key(raw_key)
        assert not await api_keys.is_key_authorized_for_org(raw_key, alenka_org_id)


def test_hash_key_is_deterministic_and_not_reversible():
    h1 = hash_key("medkard_abc")
    h2 = hash_key("medkard_abc")
    assert h1 == h2
    assert h1 != "medkard_abc"
    assert len(h1) == 64  # sha256 hex digest
