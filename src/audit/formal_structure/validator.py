"""
FormalValidator — formal-structure audit for a single ambulatory visit.

Workflow::
    validator = FormalValidator()

    visit_type = await validator.get_visit_type(visit)   # VisitType enum
    rules      = validator.get_rules(visit_type)          # applicable rule dicts
    findings   = await validator.validate(visit)          # [{flag, issue}, ...]

The `validate` method combines the two steps above, renders the rules into
the system prompt, and calls the LLM via LLM.validations.validate_visit.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum, auto
from pathlib import Path
from typing import Any

from LLM.chinese_detector import ChineseDetector
from LLM.validations import validate_visit
from LLM.visit_classifier import VisitClassifier

_chinese_detector = ChineseDetector()

# Средний сегмент: 2 цифры у A-кодов (A04.16.001), 3 у B-кодов (B01.070.001).
NMU_RE = re.compile(r"^[ABАВ]\d{2}\.\d{2,3}\.\d{3}(?:\.\d{3})?$", re.I)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_RULES_PATH = _HERE / "rules.json"
_PROMPT_PATH = Path(__file__).parent.parent.parent / "LLM" / "prompts" / "formal_structure_validator.txt"
# ─────────────────────────────────────────────────────────────────────────────

_RULES_DOC: dict = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
_RULES: list[dict] = _RULES_DOC["rules"]
_REVISED_AT: str = _RULES_DOC["revised_at"]
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
           - A*                                  → LAB_RESEARCH_INTERVENTION
           - B04.*                               → PROPHYLACTIC
           - B01.070.{001}                       → PRIMARY
           - B01.070.{011,012}                   → REPEAT
           - B01 with middle≠070 or unknown sfx  → OTHER
        3. Keyword fallback on ``Наименование`` (only for services with no NMU code):
           повторн / первичн / профилактическ
        4. Services that match nothing contribute OTHER.
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
                    parts = code.split(".")
                    prefix, last = parts[0], parts[-1]
                    if prefix.startswith("A"):
                        svc_type = VisitType.LAB_RESEARCH_INTERVENTION
                        break
                    if svc_type is None:
                        if prefix == "B04":
                            svc_type = VisitType.PROPHYLACTIC
                        elif prefix == "B01":
                            middle = parts[1] if len(parts) > 1 else ""
                            if middle != "070" or last not in {"001", "011", "012"}:
                                svc_type = VisitType.OTHER
                            elif last == "001":
                                svc_type = VisitType.PRIMARY
                            else:  # 011 or 012
                                svc_type = VisitType.REPEAT
                else:
                    continue
                break  # inner for-raw broke via A-code, propagate

            if svc_type is None:
                # ── keyword fallback for this service ─────────────────────────
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

    def get_rules(
        self,
        visit_types: set[VisitType],
        patient_age: int | None,
        icd_codes: list[str] | None = None,
    ) -> list[dict]:
        """Return rules applicable to the given visit types, age and ICD codes.

        Age group matching: a rule passes if its ``age_group`` is ``"all"``,
        or matches the derived group (``"child"`` if age < 18, ``"adult"`` otherwise).
        ``patient_age=None`` skips age filtering (matches any age_group).

        ICD matching: a rule carrying ``applies_to.icd_prefixes`` passes only if
        one of ``icd_codes`` starts with one of those prefixes.  Without codes
        such a rule never applies.
        """
        type_keys = {_VISIT_TYPE_RULE_KEY[vt] for vt in visit_types}
        age_group: str | None = None
        if patient_age is not None:
            age_group = "child" if patient_age < 18 else "adult"

        codes = [c.strip().upper() for c in (icd_codes or []) if c and c.strip()]

        seen: set[str] = set()
        rules: list[dict] = []
        for rule in _RULES:
            applies = rule.get("applies_to", {})
            visit_type_applies = applies.get("visit_types", [])
            if not ("all" in visit_type_applies or type_keys & set(visit_type_applies)):
                continue
            if age_group is not None:
                rule_age = applies.get("age_group", "all")
                if rule_age != "all" and rule_age != age_group:
                    continue
            prefixes = applies.get("icd_prefixes") or []
            if prefixes and not any(c.startswith(p) for c in codes for p in prefixes):
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

    def _render_prompt(self, rules: list[dict]) -> str:
        """Render the system prompt with the given rules injected."""
        rules_text = self._format_rules(rules)
        return _PROMPT_TEMPLATE.replace("{rules}", rules_text)

    def _check_nmu_keyword_contradiction(self, visit: dict[str, Any]) -> dict[str, str] | None:
        """Return a NMU_CODE_CONTRADICTION finding if NMU code and service name disagree.

        Checks only the PRIMARY/REPEAT pair: B01.xxx.002 with «первичн» in the
        name, or B01.xxx.001 with «повторн» in the name.  All other combinations
        are either unambiguous (B04, A-codes) or have no name-based signal.
        """
        services: list = visit.get("Услуги") or []
        for svc in services:
            if not isinstance(svc, dict):
                continue
            name: str = (svc.get("Наименование") or "").lower()
            for raw in svc.values():
                if not raw:
                    continue
                for token in str(raw).split():
                    m = NMU_RE.match(token.strip())
                    if not m:
                        continue
                    code = m.group(0).upper().replace("В", "B").replace("А", "A")
                    parts = code.split(".")
                    prefix, last = parts[0], parts[-1]
                    if prefix != "B01":
                        continue
                    if last == "002" and "первичный" in name:
                        return {
                            "flag": "NMU_CODE_CONTRADICTION",
                            "issue": (
                                f"NMU-код {m.group(0)} соответствует повторному приёму "
                                f"(суффикс .002), но наименование услуги содержит «первичный»"
                            ),
                        }
                    if last == "001" and "повторный" in name:
                        return {
                            "flag": "NMU_CODE_CONTRADICTION",
                            "issue": (
                                f"NMU-код {m.group(0)} соответствует первичному приёму "
                                f"(суффикс .001), но наименование услуги содержит «повторный»"
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
        3. Renders the system prompt with those rules.
        4. Calls the LLM and returns structured findings.

        Args:
            visit: Raw visit dict (as parsed from the source JSON).

        Returns:
            (findings, tokens) — findings is a list of ``{"flag": ..., "issue": ...}``
            dicts; tokens is the total LLM token count for this call.
        """
        visit_types = await self.get_visit_types(visit)
        logger.debug("[formal] visit_types resolved: %s", {vt.name for vt in visit_types})

        _raw_age = (visit.get("Пациент") or {}).get("AGE")
        patient_age: int | None = int(_raw_age) if isinstance(_raw_age, str) and _raw_age.strip().isdigit() else (_raw_age if isinstance(_raw_age, int) else None)
        icd_codes: list[str] = [
            str(d.get("КодМКБ") or "")
            for d in (visit.get("Диагнозы") or [])
            if isinstance(d, dict)
        ]
        rules = self.get_rules(visit_types, patient_age, icd_codes)
        logger.debug("[formal] applicable rules (%d): %s", len(rules), [r.get("flag_code") for r in rules])

        system_prompt = self._render_prompt(rules)

        findings, tokens = await validate_visit(system_prompt, visit)
        logger.info("[formal] LLM returned %d finding(s), tokens=%d: %s", len(findings), tokens, findings)

        for i, finding in enumerate(findings):
            if _chinese_detector.check_str(finding["issue"]):
                repaired, repair_tokens = await _chinese_detector.repair_issue(finding["issue"])
                findings[i] = {**finding, "issue": repaired}
                tokens += repair_tokens

        findings = _enrich_flags(findings)

        contradiction = self._check_nmu_keyword_contradiction(visit)
        if contradiction:
            logger.warning("[formal] NMU/keyword contradiction: %s", contradiction["issue"])
            # NMU contradictions are always kept; source is not from rules.json
            findings.append({**contradiction, "source": ""})

        return findings, tokens
