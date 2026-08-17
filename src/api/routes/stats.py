"""
api/stats.py — operational statistics routes.

Separate from visits.py, which serves the card-exchange contract (check /
pull / push / export). These routes answer questions *about* the stored data
rather than moving cards, so they live under their own /stats prefix.

Auth is the same require_org_access dependency: the org is named by slug in
?org= and the bearer key must be authorized for it, so one integrator cannot
read another organization's storage figures.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_org_access
from api.models import StorageStatsResponse
from storage.stats_storage import StatsStorage

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/storage", response_model=StorageStatsResponse)
async def storage(
    org_access: tuple[str, str] = Depends(require_org_access),
) -> StorageStatsResponse:
    """Kilobytes of stored data for the requesting organization.

    Covers the two tables holding per-organization card data: done_cards and
    audit_overwrite_journal (old cards kept when a push overwrote an audit).
    """
    org_id, org_name = org_access
    async with StatsStorage() as stats:
        sizes = await stats.storage_kb(organization_id=org_id)

    return StorageStatsResponse(organization=org_name, **sizes)
