"""
api/demo_doctors.py — TEMPORARY: give one clinic's visits a made-up doctor.

WHY THIS EXISTS: the personal-report mailing and the "talk about this
doctor's cards" flow both hang on `Прием.Врач_код` (the filter of the pull
API and the key of the engine's doctor_user_map) and on `Прием.Врач`. 1C
does not send either for Alenka yet, so on live data both are empty and
neither feature has anything to show. Until 1C starts sending doctors, every
card of the named organization gets one of the made-up doctors from
resources/demo_doctors.json as it arrives.

THIS IS A CRUTCH AND IS BUILT TO BE REMOVED: `git rm` this module, drop the
three lines it occupies in api/routes/visits.py and the three in
audit/excel_formatter.py, and the crutch is gone. Nothing else imports it
except the backfill script.

The made-up doctor is for the pull API only. The xlsx report is read by
another product, so `unstamp` takes the doctor back out on the way into it.

The switch is a single variable — DEMO_DOCTOR_STAMP_ORG=Alenka, empty means
off — rather than a flag plus an organization name, which would eventually
disagree with each other and stamp the wrong clinic in silence.

The stamp never overwrites a doctor that is already there:
- the card 1C pushed carries one — real data outranks the crutch, and the
  day 1C starts sending doctors the stamp becomes a no-op by itself;
- the stored card carries one — upsert_pending rewrites card_data whole on
  every push, so without carrying the old value over a re-pushed visit would
  draw a new doctor each time and answer with a different doctor mid-demo.
"""

from __future__ import annotations

import json
import logging
import os
import random
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_FILE = _ROOT / "resources" / "demo_doctors.json"

_ORG_ENV = "DEMO_DOCTOR_STAMP_ORG"
_FILE_ENV = "DEMO_DOCTOR_FILE"


def enabled_for(org_name: str) -> bool:
    """True when the crutch is switched on for this organization.

    Case-insensitive: ?org= is resolved case-insensitively by
    require_org_access, so an .env written in lower case must still match the
    canonical name the DB returns.
    """
    wanted = os.getenv(_ORG_ENV, "").strip()
    return bool(wanted) and wanted.casefold() == (org_name or "").strip().casefold()


def doctors_path() -> str:
    return os.getenv(_FILE_ENV, "").strip() or str(_DEFAULT_FILE)


@lru_cache(maxsize=4)
def load_doctors(path: str | None = None) -> list[dict]:
    """The made-up doctors, or an empty list if the file is missing or broken.

    Never raises: a demo file must not cost the clinic its ingest. An empty
    list turns the stamp into a no-op, which is logged once per push.
    """
    try:
        raw = json.loads(Path(path or doctors_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("демо-врачи: файл %s не прочитан (%s) — штамп выключен",
                       path or doctors_path(), exc)
        return []
    if not isinstance(raw, list):
        logger.warning("демо-врачи: в файле %s ожидался массив — штамп выключен",
                       path or doctors_path())
        return []
    return [{"code": str(d["code"]), "name": str(d["name"])} for d in raw
            if isinstance(d, dict) and d.get("code") and d.get("name")]


def stamp(card: dict, *, previous: dict | None, doctors: list[dict] | None = None) -> dict:
    """The card with a doctor in its Прием block, or the card as it came.

    *previous* is the stored Прием block of the same card (DoneCardsStorage
    .get_priem), or None for a card seen for the first time.
    """
    priem = card.get("Прием")
    if not isinstance(priem, dict):
        return card

    if priem.get("Врач_код"):
        return card

    if previous and previous.get("Врач_код"):
        doctor = {"code": previous["Врач_код"], "name": previous.get("Врач") or ""}
    else:
        doctors = load_doctors() if doctors is None else doctors
        if not doctors:
            logger.warning("демо-врачи: список пуст, карта уходит без врача")
            return card
        picked = random.choice(doctors)
        doctor = {"code": picked["code"], "name": picked["name"]}

    return {**card, "Прием": {**priem, "Врач": doctor["name"], "Врач_код": doctor["code"]}}


def unstamp(card: dict) -> dict:
    """The card with the doctor taken back out of its Прием block.

    The xlsx report is read by another product, and on its demo the clinic
    has to look the way it looked before this crutch: with no doctor at all.
    Both keys go, not only the made-up ones — while the stamp is on there is
    no other kind of doctor on this clinic's cards, and a report that drops
    some doctors and keeps others would be harder to explain than one that
    shows none. The stored card is not touched: this is a view of it.
    """
    priem = card.get("Прием")
    if not isinstance(priem, dict):
        return card
    if "Врач" not in priem and "Врач_код" not in priem:
        return card
    return {**card, "Прием": {k: v for k, v in priem.items() if k not in ("Врач", "Врач_код")}}
