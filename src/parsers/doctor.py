"""
parsers/doctor.py — bring a clinic's doctor fields into our card form.

Our form (docs/clinic-data-requirements.md, what MDS sends): `Прием.Врач` is
the doctor's name, `Прием.Врач_код` a stable doctor code, and the top-level
`Врач` block carries only `SPECIALIZATION`. Everything downstream — the pull
API filter, /visits/doctors, personal reports, the engine's doctor_user_map —
reads exactly those three places.

Alenka ignored that and sends one object at the top: `Врач = {GUID, FIO,
SPECIALIZATION}`. Rather than teach every reader both shapes, the card is
converted once, on the way in, and stored in our form only. A third clinic
with a third shape gets a third branch here, nowhere else.

Called at every point where a card enters done_cards: POST /visits/push
(before the demo-doctor stamp, so the stamp sees a real code and stays out),
the nightly 1C pipeline, and scripts/operator/backfill-priem.py.
"""

from __future__ import annotations

from typing import Any

_MOVED_KEYS = ("GUID", "FIO")


def normalize_doctor(card: dict[str, Any]) -> dict[str, Any]:
    """The card in our doctor form. Cards already in it come back unchanged.

    Never invents a doctor: an empty GUID or FIO leaves the matching Прием
    field alone. A non-empty one overwrites whatever is there — Alenka does
    not send Прием.Врач/Врач_код, so a value already present can only be the
    demo-doctor stamp, and the real doctor outranks it.
    """
    doctor = card.get("Врач")
    if not isinstance(doctor, dict) or not any(key in doctor for key in _MOVED_KEYS):
        return card

    priem = dict(card.get("Прием") or {})
    if doctor.get("FIO"):
        priem["Врач"] = doctor["FIO"]
    if doctor.get("GUID"):
        priem["Врач_код"] = doctor["GUID"]
    top = {key: value for key, value in doctor.items() if key not in _MOVED_KEYS}
    return {**card, "Прием": priem, "Врач": top}
