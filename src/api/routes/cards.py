"""
api/cards.py — check/pull routes for the pull API.

Route handlers only: parse query params, delegate to
reporting.api_formatter.ApiFormatter, return the result. No DB access or
business logic here.

One unified API key authenticates the integrating app, scoped to specific
organizations (see api/auth.py); each request names the organization it
wants via ?org=<name>, same as scripts/create_report.py's --org flag.

`pull` returns an xlsx file (not JSON): the integrating service stores it
and runs it through its own RAG ingestion pipeline as a file, so the wire
format has to be the actual workbook, not a JSON description of one.
"""

from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from api.auth import require_org_access
from api.models import CheckResponse, PushResponse
from reporting.api_formatter import ApiFormatter
from storage.done_cards_storage import DoneCardsStorage

router = APIRouter(prefix="/cards", tags=["cards"])

_XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/check", response_model=CheckResponse)
async def check(
    date_: date = Query(..., alias="date"),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> CheckResponse:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        count = await formatter.check(date_, org_id)
    return CheckResponse(date=date_.isoformat(), count=count)


@router.get("/pull")
async def pull(
    date_: date = Query(..., alias="date"),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> Response:
    org_id, org_name = org_access
    async with ApiFormatter() as formatter:
        count = await formatter.check(date_, org_id)
        if count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No cards for {org_name} on {date_.isoformat()}",
            )
        xlsx_bytes = await formatter.make_xlsx(date_, org_id)

    filename = f"report_{org_name}_{date_.isoformat()}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export")
async def export(
    since: str | None = Query(default=None),
    limit: int = Query(default=0, ge=0),
    cursor: int = Query(default=0, ge=0),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[dict]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        return await formatter.export(org_id, since, limit, cursor)


def _extract_card_guid(card: dict) -> str | None:
    priem = card.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid).lower() if guid else None


@router.post("/push", response_model=PushResponse)
async def push(
    card: dict,
    org_access: tuple[str, str] = Depends(require_org_access),
) -> PushResponse:
    org_id, _ = org_access
    card_guid = _extract_card_guid(card)
    if not card_guid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Card is missing Прием.GUID",
        )

    async with DoneCardsStorage() as storage:
        await storage.upsert_pending(
            card_guid=card_guid,
            card_data=json.dumps(card, ensure_ascii=False),
            organization_id=org_id,
        )

    return PushResponse(card_guid=card_guid, status="pending")
