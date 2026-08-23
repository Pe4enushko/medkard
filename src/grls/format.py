"""Human/LLM-readable rendering of a GRLS lookup (shared by the tool and the graph node)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from grls.match import MatchKind
from grls.status import (LIVE_STATUSES, STATUS_ANNULLED, STATUS_CONFIRMING, STATUS_EXPIRED,
                         STATUS_FOREIGN_PACK, STATUS_SUSPENDED, StatusAtVisit, status_at)
from storage.models.dietary_supplement import DietarySupplement
from storage.models.grls_record import GrlsRecord

MAX_TRADE_RECORDS = 6
MAX_LIST_ITEMS = 5
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
    inn_match: MatchKind | None = None
    inn_valid_at_visit: int = 0          # мёртвых сегодня РУ, действовавших на дату визита
    trade_records: list[GrlsRecord] = field(default_factory=list)
    trade_match: MatchKind | None = None
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


def _inn_names(records: list[GrlsRecord]) -> list[str]:
    names: list[str] = []
    for r in records:
        if r.inn_name and r.inn_name not in names:
            names.append(r.inn_name)
    return names


def _trade_names(records: list[GrlsRecord]) -> list[str]:
    names: list[str] = []
    for r in records:
        if r.trade_name not in names:
            names.append(r.trade_name)
    return names


def _inn_headline(lookup: MedicineLookup) -> str:
    """Утверждать «это МНН» можно только про точное совпадение.

    Нечёткое совпадение — похожесть строки, а не опознание препарата: на
    пороге 0.6 «Преднизолон» и «Преднизон» неразличимы, а это разные вещества.
    """
    found = ", ".join(_inn_names(lookup.inn_records)[:MAX_LIST_ITEMS]) or lookup.query
    if lookup.inn_match is MatchKind.EXACT:
        return f"В ГРЛС «{lookup.query}» — это МНН."
    if lookup.inn_match is MatchKind.CONTAINS:
        return f"В ГРЛС «{lookup.query}» входит в МНН «{found}»."
    return (f"Точного совпадения с МНН нет. По написанию похоже на «{found}» — "
            f"совпадение неточное, препарат не опознан.")


def _live_line(lookup: MedicineLookup) -> str:
    total = sum(lookup.inn_counts.values()) or len(lookup.inn_records)
    live = sum(n for s, n in lookup.inn_counts.items() if s in LIVE_STATUSES)
    if lookup.on and lookup.inn_valid_at_visit:
        return (f"Регистраций: {total}, действовавших на дату визита {_iso(lookup.on)}: "
                f"{live + lookup.inn_valid_at_visit} (действующих сейчас: {live}).")
    return f"Регистраций: {total}, из них действующих: {live}."


def format_medicine_lookup(lookup: MedicineLookup) -> str:
    if lookup.inn_records:
        live = sum(n for s, n in lookup.inn_counts.items() if s in LIVE_STATUSES)
        lines = [f"{_inn_headline(lookup)} {_live_line(lookup)} "
                 f"Примеры торговых наименований: "
                 f"{', '.join(_trade_names(lookup.inn_records)[:MAX_LIST_ITEMS])} "
                 f"({_registry_note(lookup)})."]
        if live + lookup.inn_valid_at_visit == 0:
            lines.append("Внимание: все РУ по этому МНН истекли или аннулированы.")
        return "\n".join(lines)
    if lookup.trade_records:
        recs = lookup.trade_records[:MAX_TRADE_RECORDS]
        head = (f"Точное совпадение с торговым наименованием "
                f"({len(recs)}; {_registry_note(lookup)}):")
        if lookup.trade_match is MatchKind.CONTAINS:
            head = (f"Точного совпадения с торговым наименованием нет: написанное врачом и "
                    f"наименование из реестра входят одно в другое ({len(recs)}; "
                    f"{_registry_note(lookup)}). Карточки подходят предположительно:")
        elif lookup.trade_match is MatchKind.FUZZY:
            head = (f"Точного совпадения с торговым наименованием нет; по написанию похожи "
                    f"({len(recs)}; {_registry_note(lookup)}). Препарат не опознан:")
        lines = [head + "\n"]
        lines += [f"--- {i} ---\n{format_record(r, lookup.on)}" for i, r in enumerate(recs, 1)]
        return "\n\n".join(lines)
    if lookup.supplements:
        lines = [
            "В ГРЛС лекарственный препарат не найден. "
            f"Найдены похожие записи БАД в Едином реестре свидетельств о государственной регистрации ({len(lookup.supplements)}):\n"
        ]
        lines += [f"--- {i} ---\n{_format_supplement(s)}" for i, s in enumerate(lookup.supplements, 1)]
        return "\n\n".join(lines)
    return NOT_FOUND
