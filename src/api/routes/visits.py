"""
api/visits.py — check/pull routes for the pull API.

Route handlers only: parse query params, delegate to
reporting.api_formatter.ApiFormatter, return the result. No DB access or
business logic here.

One unified API key authenticates the integrating app, scoped to specific
organizations (see api/auth.py); each request names the organization it
wants via ?org=<name>, same as scripts/operator/create_report.py's --org flag.

`pull` returns an xlsx file (not JSON): the integrating service stores it
and runs it through its own RAG ingestion pipeline as a file, so the wire
format has to be the actual workbook, not a JSON description of one.
"""

from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from api import demo_doctors
from api.auth import require_org_access
from api.models import CheckResponse, DoctorEntry, PushResponse
from parsers.excel import build_empty_report_bytes
from reporting.api_formatter import ApiFormatter
from storage.done_cards_storage import DoneCardsStorage

router = APIRouter(prefix="/visits", tags=["visits"])

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
    # The pattern keeps doctor_code safe to interpolate into the
    # Content-Disposition filename below: no quotes, no CR/LF, no separators.
    doctor_code: str | None = Query(
        default=None, min_length=1, max_length=64, pattern=r"^[\w-]+$"
    ),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> Response:
    org_id, org_name = org_access
    async with ApiFormatter() as formatter:
        count = await formatter.check(date_, org_id, doctor_code)
        if count == 0 and doctor_code is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No visits for {org_name} on {date_.isoformat()}",
            )
        if count == 0:
            # Personal report: the caller delivers a file per doctor either way,
            # so "no visits" is stated inside the workbook, not as a 404.
            xlsx_bytes = build_empty_report_bytes(
                f"За {date_.strftime('%d.%m.%Y')} приёмов врача с кодом {doctor_code} не обнаружено"
            )
        else:
            xlsx_bytes = await formatter.make_xlsx(date_, org_id, doctor_code)

    suffix = f"_doc{doctor_code}" if doctor_code is not None else ""
    filename = f"report_{org_name}_{date_.isoformat()}{suffix}.xlsx"
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
    include_ignored: bool = Query(
        default=False,
        description="Include cards the audit deliberately skipped (status='ignored')",
    ),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[dict]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        return await formatter.export(org_id, since, limit, cursor, include_ignored)


@router.get("/check_updates")
async def check_updates(
    since: str | None = Query(default=None),
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[dict]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        return await formatter.check_updates(org_id, since)


@router.get("/doctors", response_model=list[DoctorEntry])
async def doctors(
    org_access: tuple[str, str] = Depends(require_org_access),
) -> list[DoctorEntry]:
    org_id, _ = org_access
    async with ApiFormatter() as formatter:
        rows = await formatter.doctors(org_id)
    return [DoctorEntry(code=row["code"], name=row["name"]) for row in rows]


def _extract_card_guid(card: dict) -> str | None:
    priem = card.get("Прием") or {}
    guid = priem.get("GUID")
    return str(guid) if guid else None


_MIN_VIABLE_VISIT_KEYS = ("Пациент", "Услуги", "Диагнозы")


def _looks_like_a_visit(card: dict) -> bool:
    """True if at least one of Пациент/Услуги/Диагнозы is present as a key.

    Presence, not non-emptiness: real cards sometimes legitimately carry an
    empty Пациент/Услуги/Диагнозы, so rejecting on empty values would risk
    dropping genuine visits. This only screens out payloads missing all
    three sections outright — an empty shell rather than a real visit.
    """
    return any(key in card for key in _MIN_VIABLE_VISIT_KEYS)


@router.post("/push", response_model=PushResponse)
async def push(
    card: dict,
    org_access: tuple[str, str] = Depends(require_org_access),
) -> PushResponse:
    org_id, org_name = org_access
    card_guid = _extract_card_guid(card)
    if not card_guid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Card is missing Прием.GUID",
        )
    if not _looks_like_a_visit(card):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Card has none of Пациент/Услуги/Диагнозы — looks like an empty shell, not a real visit",
        )

    async with DoneCardsStorage() as storage:
        # ВРЕМЕННЫЙ КОСТЫЛЬ (api/demo_doctors.py): 1С не шлёт врача, а показать
        # персональные отчёты надо. Убирается тремя строками отсюда.
        if demo_doctors.enabled_for(org_name):
            card = demo_doctors.stamp(card, previous=await storage.get_priem(card_guid))
        await storage.upsert_pending(
            card_guid=card_guid,
            card_data=json.dumps(card, ensure_ascii=False),
            organization_id=org_id,
        )

    return PushResponse(card_guid=card_guid, status="pending")
