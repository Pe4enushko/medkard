"""
FormalValidator — formal-structure audit for a single ambulatory visit.

Workflow::
    validator = FormalValidator()

    visit_type = await validator.get_visit_type(visit)   # VisitType enum
    rules      = validator.get_rules(visit_type)          # applicable rule dicts
    findings   = await validator.validate(visit)          # [{flag, issue}, ...]

The `validate` method combines the two steps above and checks every selected
rule in its own atomic LLM request via LLM.validations.validate_rule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum, auto
from pathlib import Path
from typing import Any

from LLM.chinese_detector import ChineseDetector
from LLM.validations import validate_rule
from LLM.visit_classifier import VisitClassifier
from parsers.json_parser import patient_age as _patient_age

_chinese_detector = ChineseDetector()

# Совершеннолетие: 404н задаёт объём ПМО для «граждан в возрасте 18 лет и
# старше», 192н и 211н говорят о несовершеннолетних. Та же граница, что в
# audit.diagnosis.clinic_recs — держим их согласованными.
_ADULT_AGE = 18

# Средний сегмент: 2 цифры у A-кодов (A04.16.001), 3 у B-кодов (B01.070.001).
NMU_RE = re.compile(r"^[ABАВ]\d{2}\.\d{2,3}\.\d{3}(?:\.\d{3})?$", re.I)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_RULES_PATH = _HERE / "rules.json"
_NMU_PATH = _HERE / "nmu_visit_types.json"
_PROMPT_PATH = Path(__file__).parent.parent.parent / "LLM" / "prompts" / "formal_structure_validator.txt"
# ─────────────────────────────────────────────────────────────────────────────

_RULES_DOC: dict = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
_RULES: list[dict] = _RULES_DOC["rules"]
_REVISED_AT: str = _RULES_DOC["revised_at"]
_NMU_DOC: dict = json.loads(_NMU_PATH.read_text(encoding="utf-8"))
_PROMPT_TEMPLATE: str = _PROMPT_PATH.read_text(encoding="utf-8")

# ── Flag → regulatory source lookup ───────────────────────────────────────────
_FLAG_SOURCE: dict[str, str] = {r["flag_code"]: r.get("source", "") for r in _RULES}
_ALL_FLAGS: list[str] = list(_FLAG_SOURCE)

_VERIFIED_DATES: list[str] = sorted(r["verified_at"] for r in _RULES if r.get("verified_at"))
logger.info(
    "[formal] formal rules revised_at=%s rules=%d oldest verified_at=%s",
    _REVISED_AT,
    len(_RULES),
    _VERIFIED_DATES[0] if _VERIFIED_DATES else "none",
)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ch_a != ch_b)))
        prev = curr
    return prev[-1]


def _enrich_flags(findings: list[dict]) -> list[dict]:
    """Attach 'source' to each finding and drop any flag not in rules.json (Levenshtein > 3)."""
    result: list[dict] = []
    for f in findings:
        flag = f["flag"]
        if flag in _FLAG_SOURCE:
            result.append({**f, "source": _FLAG_SOURCE[flag]})
            continue
        best: str | None = min(_ALL_FLAGS, key=lambda k: _levenshtein(flag, k), default=None)
        if best is not None and _levenshtein(flag, best) <= 3:
            logger.warning("[formal] flag %r fuzzy-matched to %r", flag, best)
            result.append({**f, "flag": best, "source": _FLAG_SOURCE[best]})
        else:
            logger.warning("[formal] dropping unrecognised flag %r (no match within edit distance 3)", flag)
    return result


class VisitType(Enum):
    """Type of ambulatory visit derived from the service name."""
    PRIMARY = auto()                   # первичный
    REPEAT = auto()                    # повторный
    PROPHYLACTIC = auto()              # профилактический
    PROPHYLACTIC_TUBERCULIN = auto()   # профилактический туберкулинодиагностика (Z11.1)
    LAB_RESEARCH_INTERVENTION = auto() # лабораторные/инструментальные/вмешательства (A-коды)
    OTHER = auto()                     # не удалось определить тип


# Mapping VisitType → value expected in rules.json "applies_to.visit_types"
_VISIT_TYPE_RULE_KEY: dict[VisitType, str] = {
    VisitType.PRIMARY:                   "primary",
    VisitType.REPEAT:                    "repeat",
    VisitType.PROPHYLACTIC:              "prophylactic",
    VisitType.PROPHYLACTIC_TUBERCULIN:   "prophylactic_tuberculin",
    VisitType.LAB_RESEARCH_INTERVENTION: "lab_research_intervention",
    VisitType.OTHER:                     "other",
}

# Код номенклатуры → тип приёма, по приказу 804н (см. _NMU_DOC["scope"]:
# в словаре только амбулаторные приёмы; всё остальное разбирается по
# наименованию услуги). Средний сегмент кода — это специальность врача, а не
# признак приёма, поэтому вывести тип из самого кода арифметически нельзя:
# B01.047.001/.002 — терапевт первичный/повторный, а B01.047.010/.011 — врач по
# водолазной медицине первичный/повторный.
_NMU_VISIT_TYPES: dict[str, VisitType] = {
    code: VisitType[entry["visit_type"]] for code, entry in _NMU_DOC["codes"].items()
}
_NMU_NAMES: dict[str, str] = {
    code: entry["name"] for code, entry in _NMU_DOC["codes"].items()
}

logger.info(
    "[formal] NMU dictionary source=%s verified_at=%s codes=%d",
    _NMU_DOC["source"],
    _NMU_DOC["verified_at"],
    len(_NMU_VISIT_TYPES),
)

_LLM_LABEL_TO_TYPE: dict[str, VisitType] = {
    "primary":                   VisitType.PRIMARY,
    "repeat":                    VisitType.REPEAT,
    "prophylactic":              VisitType.PROPHYLACTIC,
    "prophylactic_tuberculin":   VisitType.PROPHYLACTIC_TUBERCULIN,
    "lab_research_intervention": VisitType.LAB_RESEARCH_INTERVENTION,
    "other":                     VisitType.OTHER,
}


class FormalValidator:
    """Validates the formal structure of a single ambulatory visit record.

    All rule data is loaded once at class instantiation from ``rules.json``
    and the system prompt template is loaded from ``LLM/prompts/``.

    # TODO: refactor to use parsers.json_parser.AppointmentParser.parse() instead of direct key access
    """

    async def get_visit_types(self, visit: dict[str, Any]) -> set[VisitType]:
        """Determine all visit types present in a visit by checking each service.

        Each service entry is classified independently:
        1. Z11.1 among visit["Диагнозы"][].КодМКБ → always adds PROPHYLACTIC_TUBERCULIN.
        2. Per-service NMU code scan:
           - A*                                → LAB_RESEARCH_INTERVENTION
           - code found in ``nmu_visit_types.json`` (804н) → its visit type
           - any other code                    → no verdict, step 3 decides
        3. Keyword fallback on ``Наименование`` — for every service the codes
           left undecided, not only for services without a code at all:
           повторн / первичн / профилактическ
        4. A service neither step decided contributes nothing; OTHER is the
           answer only when no service contributed anything at all.
        """
        result: set[VisitType] = set()

        # ── Z11.1 always adds PROPHYLACTIC_TUBERCULIN ─────────────────────────
        diagnoses: list = visit.get("Диагнозы") or []
        if any(
            str(d.get("КодМКБ") or "").strip().upper() == "Z11.1"
            for d in diagnoses
            if isinstance(d, dict)
        ):
            result.add(VisitType.PROPHYLACTIC_TUBERCULIN)

        services: list = visit.get("Услуги") or []
        if not services:
            logger.warning("[formal] visit has empty or missing Услуги — defaulting to OTHER")
            result.add(VisitType.OTHER)
            return result

        for svc in services:
            if not isinstance(svc, dict):
                continue

            svc_type: VisitType | None = None

            for raw in svc.values():
                if not raw:
                    continue
                for token in str(raw).split():
                    m = NMU_RE.match(token.strip())
                    if not m:
                        continue
                    code = m.group(0).upper().replace("В", "B").replace("А", "A")
                    if code.startswith("A"):
                        # Лабораторные, инструментальные и вмешательства — по
                        # префиксу: их тысячи, а конкретную услугу правила
                        # отбирают через applies_to.service_code_prefixes.
                        svc_type = VisitType.LAB_RESEARCH_INTERVENTION
                        break
                    if svc_type is None:
                        # Кода нет в словаре 804н (не приём либо услуга клиники
                        # вне номенклатуры) — вердикта не выносим, пусть решает
                        # наименование.
                        svc_type = _NMU_VISIT_TYPES.get(code)
                else:
                    continue
                break  # inner for-raw broke via A-code, propagate

            if svc_type is None:
                # ── keyword fallback for this service ─────────────────────────
                # Достижим и тогда, когда код у услуги есть, но словарь 804н его
                # не знает: раньше такой код молча становился OTHER и глушил
                # разбор наименования.
                name: str = (svc.get("Наименование") or "").lower()
                if "повторн" in name:
                    svc_type = VisitType.REPEAT
                elif "первичн" in name:
                    svc_type = VisitType.PRIMARY
                elif "профилактическ" in name:
                    svc_type = VisitType.PROPHYLACTIC

            if svc_type is not None:
                result.add(svc_type)

        if not result:
            logger.warning("[formal] could not determine visit type — defaulting to OTHER")
            result.add(VisitType.OTHER)

        return result

    @staticmethod
    def _service_metadata(visit: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        codes: list[str] = []
        names: list[str] = []
        for service in ((visit or {}).get("Услуги") or []):
            if not isinstance(service, dict):
                continue
            name = str(service.get("Наименование") or "").strip()
            if name:
                names.append(name.casefold())
            for raw in service.values():
                if not raw:
                    continue
                for token in str(raw).split():
                    match = NMU_RE.fullmatch(token.strip())
                    if match:
                        codes.append(
                            match.group(0).upper().replace("В", "B").replace("А", "A")
                        )
        return codes, names

    def get_rules(
        self,
        visit_types: set[VisitType],
        patient_age: int | None,
        icd_codes: list[str] | None = None,
        visit: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Return rules applicable to the given visit types, age and ICD codes.

        Age group matching: a rule passes if its ``age_group`` is ``"all"``,
        or matches the derived group (``"child"`` if age < 18, ``"adult"``
        otherwise — the boundary 404н draws with «граждане в возрасте 18 лет и
        старше»).  ``patient_age=None`` means the card did not state a usable
        age: only ``age_group="all"`` rules are kept, so an unknown age can
        never widen the rule set into the wrong cohort.

        ICD matching: a rule carrying ``applies_to.icd_prefixes`` passes only if
        one of ``icd_codes`` starts with one of those prefixes.  Without codes
        such a rule never applies.

        NMU matching: ``service_code_prefixes`` and ``service_name_keywords``
        are deterministic prefilters evaluated from ``visit["Услуги"]`` before
        a rule is sent to the LLM.
        """
        type_keys = {_VISIT_TYPE_RULE_KEY[vt] for vt in visit_types}
        age_group: str | None = None
        if patient_age is not None:
            age_group = "child" if patient_age < _ADULT_AGE else "adult"

        codes = [c.strip().upper() for c in (icd_codes or []) if c and c.strip()]
        service_codes, service_names = self._service_metadata(visit)

        seen: set[str] = set()
        rules: list[dict] = []
        for rule in _RULES:
            applies = rule.get("applies_to", {})
            visit_type_applies = applies.get("visit_types", [])
            if not ("all" in visit_type_applies or type_keys & set(visit_type_applies)):
                continue
            rule_age = applies.get("age_group", "all")
            if rule_age != "all" and rule_age != age_group:
                # age_group is None when the age is unknown: nothing but "all"
                # can match, which is the deliberate narrow side of the fork.
                continue
            prefixes = applies.get("icd_prefixes") or []
            if prefixes and not any(c.startswith(p) for c in codes for p in prefixes):
                continue
            service_prefixes = [
                str(prefix).strip().upper().replace("В", "B").replace("А", "A")
                for prefix in (applies.get("service_code_prefixes") or [])
            ]
            if service_prefixes and not any(
                code.startswith(prefix)
                for code in service_codes
                for prefix in service_prefixes
            ):
                continue
            service_name_keywords = [
                str(keyword).casefold()
                for keyword in (applies.get("service_name_keywords") or [])
            ]
            if service_name_keywords and not any(
                keyword in name
                for name in service_names
                for keyword in service_name_keywords
            ):
                continue
            fc = rule.get("flag_code", "")
            if fc not in seen:
                seen.add(fc)
                rules.append(rule)
        return rules

    def _format_rules(self, rules: list[dict]) -> str:
        """Format rules for prompt injection.

        Each rule is rendered as one line:
        ``[FLAG_CODE] <condition minus last char>: <expectation>``
        or, when no condition is present:
        ``[FLAG_CODE] <expectation>``
        """
        lines: list[str] = []
        for rule in rules:
            flag = rule.get("flag_code", "")
            expectation: str = rule.get("expectation", "")
            condition: str = rule.get("condition", "")
            if condition:
                prefix = condition.rstrip()[:-1] + ": "
                text = prefix + expectation
            else:
                text = expectation
            lines.append(f"({flag}) {text}")
        return "\n".join(lines)

    def _render_prompt(self) -> str:
        """Return the common prefix used by every atomic rule request."""
        return _PROMPT_TEMPLATE

    def _check_nmu_keyword_contradiction(self, visit: dict[str, Any]) -> dict[str, str] | None:
        """Return a NMU_CODE_CONTRADICTION finding if NMU code and service name disagree.

        Both sides are read the same way as in ``get_visit_types``: the code
        through the 804н dictionary, the name through the same word stems.
        Codes outside the dictionary (not an appointment, or a clinic-internal
        article) carry no primary/repeat claim and cannot contradict anything.
        """
        services: list = visit.get("Услуги") or []
        for svc in services:
            if not isinstance(svc, dict):
                continue
            name: str = (svc.get("Наименование") or "").lower()
            name_type: VisitType | None = None
            if "повторн" in name:
                name_type = VisitType.REPEAT
            elif "первичн" in name:
                name_type = VisitType.PRIMARY
            if name_type is None:
                continue
            for raw in svc.values():
                if not raw:
                    continue
                for token in str(raw).split():
                    m = NMU_RE.match(token.strip())
                    if not m:
                        continue
                    code = m.group(0).upper().replace("В", "B").replace("А", "A")
                    code_type = _NMU_VISIT_TYPES.get(code)
                    if code_type is None or code_type == name_type:
                        continue
                    if code_type not in (VisitType.PRIMARY, VisitType.REPEAT):
                        continue
                    expected = "повторному" if code_type is VisitType.REPEAT else "первичному"
                    claimed = "«повторный»" if name_type is VisitType.REPEAT else "«первичный»"
                    return {
                        "flag": "NMU_CODE_CONTRADICTION",
                        "issue": (
                            f"NMU-код {code} по номенклатуре 804н соответствует "
                            f"{expected} приёму ({_NMU_NAMES[code]}), "
                            f"но наименование услуги содержит {claimed}"
                        ),
                    }
        return None

    async def validate(
        self,
        visit: dict[str, Any],
    ) -> tuple[list[dict[str, str]], int]:
        """Validate a visit record against the applicable formal-structure rules.

        1. Determines all visit types via ``get_visit_types``.
        2. Filters ``rules.json`` to the rules applicable to that visit type.
        3. Starts one atomic LLM request per selected rule in parallel.
        4. Returns the combined structured findings in rule order.

        Args:
            visit: Raw visit dict (as parsed from the source JSON).

        Returns:
            (findings, tokens) — findings is a list of ``{"flag": ..., "issue": ...}``
            dicts; tokens is the total LLM token count for this call.
        """
        visit_types = await self.get_visit_types(visit)
        logger.debug("[formal] visit_types resolved: %s", {vt.name for vt in visit_types})

        patient_age: int | None = _patient_age(visit.get("Пациент") or {})
        if patient_age is None:
            logger.warning(
                "[formal] возраст пациента не прочитан (%r) — применяем только правила age_group=all",
                (visit.get("Пациент") or {}).get("AGE"),
            )
        icd_codes: list[str] = [
            str(d.get("КодМКБ") or "")
            for d in (visit.get("Диагнозы") or [])
            if isinstance(d, dict)
        ]
        rules = self.get_rules(visit_types, patient_age, icd_codes, visit)
        logger.debug("[formal] applicable rules (%d): %s", len(rules), [r.get("flag_code") for r in rules])

        system_prompt = self._render_prompt()
        atomic_results = await asyncio.gather(
            *(
                validate_rule(
                    system_prompt,
                    visit,
                    self._format_rules([rule]),
                    flag_code=rule["flag_code"],
                    rule_id=rule["rule_id"],
                )
                for rule in rules
            )
        )
        findings = []
        tokens = 0
        for rule, (rule_findings, rule_tokens) in zip(rules, atomic_results, strict=True):
            findings.extend(
                {**finding, "source": rule.get("source", "")}
                for finding in rule_findings
            )
            tokens += rule_tokens
        logger.info("[formal] LLM returned %d finding(s), tokens=%d: %s", len(findings), tokens, findings)

        for i, finding in enumerate(findings):
            if _chinese_detector.check_str(finding["issue"]):
                repaired, repair_tokens = await _chinese_detector.repair_issue(finding["issue"])
                findings[i] = {**finding, "issue": repaired}
                tokens += repair_tokens

        contradiction = self._check_nmu_keyword_contradiction(visit)
        if contradiction:
            logger.warning("[formal] NMU/keyword contradiction: %s", contradiction["issue"])
            # NMU contradictions are always kept; source is not from rules.json
            findings.append({**contradiction, "source": ""})

        return findings, tokens
