"""
FormalValidator — formal-structure audit for a single ambulatory visit.

Workflow::
    validator = FormalValidator()

    visit_type = validator.get_visit_type(visit)   # VisitType enum
    rules      = validator.get_rules(visit_type)    # applicable rule dicts
    findings   = await validator.validate(visit)    # [{flag, issue}, ...]

The `validate` method combines the two steps above, renders the rules into
the system prompt, and calls the LLM via LLM.validations.validate_visit.
"""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any

from LLM.validations import validate_visit

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_RULES_PATH = _HERE / "rules.json"
_PROMPT_PATH = Path(__file__).parent.parent.parent / "LLM" / "prompts" / "formal_structure_validator.txt"
# ─────────────────────────────────────────────────────────────────────────────

_RULES: list[dict] = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
_PROMPT_TEMPLATE: str = _PROMPT_PATH.read_text(encoding="utf-8")


class VisitType(Enum):
    """Type of ambulatory visit derived from the service name."""
    PRIMARY = auto()        # первичный
    REPEAT = auto()         # повторный
    PROPHYLACTIC = auto()   # профилактический


# Mapping VisitType → value expected in rules.json "applies_to.visit_types"
_VISIT_TYPE_RULE_KEY: dict[VisitType, str] = {
    VisitType.PRIMARY:       "primary",
    VisitType.REPEAT:        "repeat",
    VisitType.PROPHYLACTIC:  "prophylactic",
}


class FormalValidator:
    """Validates the formal structure of a single ambulatory visit record.

    All rule data is loaded once at class instantiation from ``rules.json``
    and the system prompt template is loaded from ``LLM/prompts/``.
    """

    def get_visit_type(self, visit: dict[str, Any]) -> VisitType:
        """Determine the visit type from the first service name.

        Inspects ``visit["Услуги"][0]["Наименование"]`` (case-insensitive) for
        the presence of «первичный», «повторный», or «профилактический».

        Args:
            visit: Raw visit dict (as parsed from the source JSON).

        Returns:
            A :class:`VisitType` enum member.

        Raises:
            ValueError: If the visit type cannot be determined.
        """
        try:
            name: str = visit["Услуги"][0]["Наименование"].lower()
        except (KeyError, IndexError, AttributeError) as exc:
            raise ValueError(
                "Cannot determine visit type: Услуги[0].Наименование is missing or invalid."
            ) from exc

        if "первичн" in name:
            return VisitType.PRIMARY
        if "повторн" in name:
            return VisitType.REPEAT
        if "профилактическ" in name:
            return VisitType.PROPHYLACTIC

        raise ValueError(
            f"Cannot determine visit type from service name: {name!r}. "
            "Expected 'первичный', 'повторный', or 'профилактический'."
        )

    def get_rules(self, visit_type: VisitType) -> list[dict]:
        """Return the subset of rules applicable to the given visit type.

        Args:
            visit_type: A :class:`VisitType` enum member.

        Returns:
            List of rule dicts from ``rules.json`` whose
            ``applies_to.visit_types`` includes the corresponding key.
        """
        key = _VISIT_TYPE_RULE_KEY[visit_type]
        return [
            rule for rule in _RULES
            if key in rule.get("applies_to", {}).get("visit_types", [])
        ]

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

    async def validate(
        self,
        visit: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Validate a visit record against the applicable formal-structure rules.

        1. Determines the visit type from ``Услуги[0].Наименование``.
        2. Filters ``rules.json`` to the rules applicable to that visit type.
        3. Renders the system prompt with those rules.
        4. Calls the LLM and returns structured findings.

        Args:
            visit: Raw visit dict (as parsed from the source JSON).

        Returns:
            List of finding dicts: ``[{"flag": ..., "issue": ...}, ...]``.
            Empty list means no formal-structure defects were detected.
        """
        visit_type = self.get_visit_type(visit)
        logger.debug("[formal] visit_type resolved: %s", visit_type.name)

        rules = self.get_rules(visit_type)
        logger.debug("[formal] applicable rules (%d): %s", len(rules), [r.get("flag_code") for r in rules])

        system_prompt = self._render_prompt(rules)
        logger.debug("[formal] rendered system prompt:\n%s", system_prompt)
        logger.debug(
            "[formal] visit payload sent to LLM:\n%s",
            json.dumps(visit, ensure_ascii=False, indent=2),
        )

        findings = await validate_visit(system_prompt, visit)
        logger.info("[formal] LLM returned %d finding(s): %s", len(findings), findings)
        return findings
