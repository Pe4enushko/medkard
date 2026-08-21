"""Human/LLM-readable rendering of a GRLS lookup (shared by the tool and the graph node)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from grls.status import (LIVE_STATUSES, STATUS_ANNULLED, STATUS_CONFIRMING, STATUS_EXPIRED,
                         STATUS_FOREIGN_PACK, STATUS_SUSPENDED, StatusAtVisit, status_at)
from storage.models.dietary_supplement import DietarySupplement
from storage.models.grls_record import GrlsRecord

MAX_TRADE_RECORDS = 6
MAX_LIST_ITEMS = 5
TRADE_THRESHOLD = 0.85
NOT_FOUND = "Препарат или БАД не найден в реестрах."

_NOTES = {
    STATUS_SUSPENDED: "приостановлено применение (предупреждение, не запрет назначения)",
    STATUS_CONFIRMING: "на подтверждении регистрации",
    STATUS_FOREIGN_PACK: "в иностранной упаковке",
}


@dataclass
class MedicineLookup:
    query: str
    on: date | None
    registry_date: date | None
    inn_records: list[GrlsRecord] = field(default_factory=list)
    inn_counts: dict[str, int] = field(default_factory=dict)
    trade_records: list[GrlsRecord] = field(default_factory=list)
    supplements: list[DietarySupplement] = field(default_factory=list)


def _iso(d: date | None) -> str:
    return d.isoformat() if d else ""


def status_line(record: GrlsRecord, on: date | None) -> str:
    sv = status_at(record, on)
    ru = f"РУ {record.reg_number}"
    if sv is StatusAtVisit.ACTIVE:
        term = f"действует до {_iso(record.expires_at)}" if record.expires_at else "бессрочно"
        return f"{record.status} ({ru}, {term})"
    if sv is StatusAtVisit.ACTIVE_WITH_NOTE:
        term = f", срок до {_iso(record.expires_at)}" if record.expires_at else ""
        return f"Действующий, {_NOTES[record.status]} ({ru}{term})"
    if sv is StatusAtVisit.UNKNOWN_END:
        return f"{record.status} (дата неизвестна; {ru})"
    if record.status == STATUS_EXPIRED:
        event = f"истекло {_iso(record.expires_at)}"
    else:  # STATUS_ANNULLED
        event = f"аннулировано {_iso(record.annulled_at or record.expires_at)}"
    if sv is StatusAtVisit.VALID_AT_VISIT:
        return f"{record.status} ({event}; на дату визита {_iso(on)} действовало; {ru})"
    return f"{record.status} ({event}; {ru})"


def _join_capped(items: list[str]) -> str:
    head = "; ".join(items[:MAX_LIST_ITEMS])
    rest = len(items) - MAX_LIST_ITEMS
    return f"{head} (+ ещё {rest})" if rest > 0 else head


def format_record(record: GrlsRecord, on: date | None) -> str:
    parts = [f"Торговое наименование: {record.trade_name}"]
    if record.inn_name:
        parts.append(f"МНН: {record.inn_name}")
    parts.append(f"Статус РУ: {status_line(record, on)}")
    if record.dosage_forms:
        parts.append(f"Лекарственные формы: {_join_capped(record.dosage_forms)}")
    if record.dispensing:
        parts.append(f"Отпуск: {_join_capped(record.dispensing)}")
    if record.pharm_group:
        parts.append(f"ФТГ: {record.pharm_group}")
    if record.is_vital is not None:
        parts.append(f"ЖНВЛП: {'да' if record.is_vital else 'нет'}")
    if record.narcotic_list:
        parts.append(f"ПКУ: {record.narcotic_list}")
    return "\n".join(parts)


def _format_supplement(s: DietarySupplement) -> str:
    parts = [f"Наименование: {s.product_name}"]
    if s.registration_number:
        parts.append(f"Свидетельство: {s.registration_number}")
    if s.status:
        parts.append(f"Статус: {s.status}")
    if s.label_info:
        parts.append(f"Информация на этикетке: {s.label_info}")
    return "\n".join(parts)


def _registry_note(lookup: MedicineLookup) -> str:
    return f"реестр от {_iso(lookup.registry_date)}" if lookup.registry_date else "дата реестра неизвестна"


def format_medicine_lookup(lookup: MedicineLookup) -> str:
    if lookup.inn_records:
        total = sum(lookup.inn_counts.values()) or len(lookup.inn_records)
        live = sum(n for s, n in lookup.inn_counts.items() if s in LIVE_STATUSES)
        names: list[str] = []
        for r in lookup.inn_records:
            if r.trade_name not in names:
                names.append(r.trade_name)
        lines = [f"В ГРЛС «{lookup.query}» — это МНН. "
                 f"Регистраций: {total}, из них действующих: {live}. "
                 f"Примеры торговых наименований: {', '.join(names[:MAX_LIST_ITEMS])} "
                 f"({_registry_note(lookup)})."]
        if live == 0:
            lines.append("Внимание: все РУ по этому МНН истекли или аннулированы.")
        return "\n".join(lines)
    if lookup.trade_records:
        recs = lookup.trade_records[:MAX_TRADE_RECORDS]
        lines = [f"Найдено в ГРЛС ({len(recs)}; {_registry_note(lookup)}):\n"]
        lines += [f"--- {i} ---\n{format_record(r, lookup.on)}" for i, r in enumerate(recs, 1)]
        return "\n\n".join(lines)
    if lookup.supplements:
        lines = [f"Найдено как БАД в Едином реестре свидетельств о государственной регистрации ({len(lookup.supplements)}):\n"]
        lines += [f"--- {i} ---\n{_format_supplement(s)}" for i, s in enumerate(lookup.supplements, 1)]
        return "\n\n".join(lines)
    return NOT_FOUND
