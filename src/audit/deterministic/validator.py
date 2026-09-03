"""DeterministicValidator — нормативные требования, проверяемые сравнением.

Часть требований нормативки — про **наличие** поля, а не про его качество:
«у визита есть диагноз с кодом МКБ» — это сравнение с пустым значением, суждения
там нет. Гонять на такое модель значит платить токенами за сравнение и получать
недетерминированный ответ на детерминированный вопрос.

Правила лежат в ``deterministic_rules.json`` рядом. Находки возвращаются в том
же виде, что у ``FormalValidator.validate`` — ``{"flag", "issue", "source"}``,
поэтому отчёт и хранилище не трогаются вовсе.

Чего здесь намеренно нет: проверок полей ``ДанныеОсмотра``. Они уже живут в
``formal_structure.required_fields`` и привязаны к шаблону записи — 1С не
присылает незаполненное поле вовсе, и без шаблона «поля нет» неотличимо от
«такого поля у клиники не бывает». Правило без этой привязки палило бы по всем
картам чужого шаблона.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from audit.deterministic.indicators import missing_indicators
from audit.formal_structure.validator import (
    NMU_RE,
    FormalValidator,
    _VISIT_TYPE_RULE_KEY,
)
from parsers.json_parser import patient_age as _patient_age

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).parent / "deterministic_rules.json"
_ALL = "all"
# Та же граница, что в formal_structure и в audit.diagnosis.clinic_recs:
# 168н адресован взрослым «в возрасте 18 лет и старше» (п. 1).
_ADULT_AGE = 18

# Классы МКБ-10, требующие кода внешней причины, и сами коды внешних причин.
# S — травмы по областям тела, T — отравления и прочие последствия внешних
# причин; V-Y — класс XX «Внешние причины заболеваемости и смертности».
_INJURY_CLASSES = ("S", "T")
_EXTERNAL_CAUSE_CLASSES = ("V", "W", "X", "Y")


def _icd_codes(visit: dict[str, Any]) -> list[str]:
    return [
        code
        for d in (visit.get("Диагнозы") or [])
        if isinstance(d, dict)
        for code in [str(d.get("КодМКБ") or "").strip().upper()]
        if code
    ]


def _service_codes(visit: dict[str, Any]) -> list[str]:
    """Коды номенклатуры из всех полей строки услуг.

    Клиника кладёт код то в Код, то в КодЕГИСЗ, то в Артикул — читаем всё,
    как это делает классификатор вида приёма.
    """
    codes: list[str] = []
    for service in (visit.get("Услуги") or []):
        if not isinstance(service, dict):
            continue
        for raw in service.values():
            if not raw:
                continue
            for token in str(raw).split():
                match = NMU_RE.fullmatch(token.strip())
                if match:
                    codes.append(
                        match.group(0).upper().replace("В", "B").replace("А", "A")
                    )
    return codes


def _inspection_text(visit: dict[str, Any]) -> str:
    return " ".join(
        str(item.get("Значение") or "")
        for item in (visit.get("ДанныеОсмотра") or [])
        if isinstance(item, dict)
    )


# ── Проверки ──────────────────────────────────────────────────────────────────
# Каждая возвращает None, когда требование выполнено, и строку, когда нарушено.
# Строка — уточнение к тексту правила: пустая, если уточнять нечего.


def _check_icd_present(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    return None if _icd_codes(visit) else ""


def _check_icd_external_cause(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    codes = _icd_codes(visit)
    injuries = [c for c in codes if c.startswith(_INJURY_CLASSES)]
    if not injuries:
        return None
    if any(code.startswith(_EXTERNAL_CAUSE_CLASSES) for code in codes):
        return None
    return ", ".join(sorted(set(injuries)))


def _check_json_nonempty(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    return None if (visit.get(check["block"]) or []) else ""


def _check_regex_in_text(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    return None if re.search(check["pattern"], _inspection_text(visit)) else ""


def _check_regex_absent(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    return "" if re.search(check["pattern"], _inspection_text(visit)) else None


def _check_service_code_present(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    """Услуги, у которых кода номенклатуры нет ни в одном поле строки.

    Ищем во всех полях, а не только в КодЕГИСЗ: на боевых картах код сплошь и
    рядом лежит в Артикул, а КодЕГИСЗ пуст. Такая услуга сопоставима с 804н,
    и замечание на неё было бы шумом — это дефект интеграции, не записи.
    """
    nameless: list[str] = []
    for service in (visit.get("Услуги") or []):
        if not isinstance(service, dict):
            continue
        if any(
            NMU_RE.fullmatch(token.strip())
            for raw in service.values() if raw
            for token in str(raw).split()
        ):
            continue
        nameless.append(str(service.get("Наименование") or "").strip() or "без наименования")
    return "; ".join(nameless) if nameless else None


def _check_controlled_indicators(visit: dict[str, Any], check: dict[str, Any]) -> str | None:
    """Контролируемые показатели 168н по диагнозам карты.

    Одна находка на карту со списком недостающего, а не по находке на
    показатель: медиана — шесть показателей на строку перечня, и врач прочитал
    бы шесть отдельных замечаний как шесть дефектов.
    """
    missing = missing_indicators(_icd_codes(visit), _inspection_text(visit))
    return ", ".join(missing) if missing else None


_CHECKS = {
    "icd_present": _check_icd_present,
    "icd_external_cause": _check_icd_external_cause,
    "json_nonempty": _check_json_nonempty,
    "regex_in_text": _check_regex_in_text,
    "regex_absent": _check_regex_absent,
    "service_code_present": _check_service_code_present,
    "controlled_indicators": _check_controlled_indicators,
}


class DeterministicValidator:
    """Проверяет визит по правилам, не требующим суждения."""

    def __init__(self, path: str | Path = _RULES_PATH) -> None:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self._rules: list[dict[str, Any]] = doc["rules"]
        enabled = [r for r in self._rules if r.get("enabled", True)]
        logger.info(
            "[deterministic] rules revised_at=%s всего=%d включено=%d",
            doc.get("revised_at"),
            len(self._rules),
            len(enabled),
        )

    def _applies(
        self,
        rule: dict[str, Any],
        visit: dict[str, Any],
        type_keys: set[str],
    ) -> bool:
        applies = rule.get("applies_to") or {}

        types = applies.get("visit_types") or []
        if types and not (_ALL in types or type_keys & set(types)):
            return False

        # Возраст неизвестен — правило с возрастным скоупом не применяется:
        # трактуем None в сторону молчания, как и остальной аудит.
        age_group = applies.get("age_group", _ALL)
        if age_group != _ALL:
            age = _patient_age(visit.get("Пациент") or {})
            if age is None:
                return False
            if age_group != ("child" if age < _ADULT_AGE else "adult"):
                return False

        prefixes = applies.get("service_prefixes") or []
        if prefixes:
            codes = _service_codes(visit)
            if not any(c.startswith(tuple(prefixes)) for c in codes):
                return False

        excludes = [s.casefold() for s in (applies.get("service_name_excludes") or [])]
        if excludes:
            names = [
                str(s.get("Наименование") or "").casefold()
                for s in (visit.get("Услуги") or [])
                if isinstance(s, dict)
            ]
            if any(token in name for name in names for token in excludes):
                return False

        return True

    async def validate(
        self,
        visit: dict[str, Any],
        visit_types: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Находки по включённым правилам.

        *visit_types* — ключи типов визита из ``rules.json``. Если не переданы,
        считаются здесь же: классификация не ходит в модель и стоит дёшево.
        """
        if visit_types is None:
            types = await FormalValidator().get_visit_types(visit)
            visit_types = {_VISIT_TYPE_RULE_KEY[t] for t in types}

        findings: list[dict[str, str]] = []
        for rule in self._rules:
            if not rule.get("enabled", True):
                continue
            if not self._applies(rule, visit, visit_types):
                continue
            check = rule["check"]
            handler = _CHECKS.get(check["kind"])
            if handler is None:
                logger.warning(
                    "[deterministic] правило %s: неизвестная проверка %r — пропущено",
                    rule.get("rule_id"),
                    check.get("kind"),
                )
                continue
            detail = handler(visit, check)
            if detail is None:
                continue
            issue = rule["issue"] + (f": {detail}" if detail else "")
            findings.append({
                "flag": rule["flag_code"],
                "issue": issue,
                "source": rule.get("source", ""),
            })
        return findings
