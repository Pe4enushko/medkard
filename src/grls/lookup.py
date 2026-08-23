"""Open the registries and assemble a MedicineLookup (I/O lives here, formatting in format.py)."""
from __future__ import annotations

import logging
from datetime import date

from grls.format import MedicineLookup
from grls.mention import search_candidates
from storage.dietary_supplements_storage import DietarySupplementsStorage
from storage.grls_storage import GrlsStorage

logger = logging.getLogger(__name__)


async def lookup_medicine(mention: str, *, on: date | None = None) -> MedicineLookup:
    """МНН, затем торговое наименование, затем БАДы — по каждому кандидату.

    Внутри каждого поиска — свои уровни: точное совпадение, вхождение,
    триграммы. Уровень возвращается наружу и обязан дойти до ответа: препарат,
    найденный по похожести строки, не опознан, а только похож.

    Кандидатов у одного упоминания бывает несколько: «Моксонидин (Физиотенз)» —
    это два законных имени одного препарата, и выбирать между ними вслепую
    хуже, чем попробовать оба. Перебор идёт до первого попадания, поэтому у
    подавляющего большинства упоминаний — ровно один запрос к базе, как раньше
    (на прогоне 2026-08-23 таких было 67 из 84).
    """
    candidates = search_candidates(mention) or [mention]
    query = candidates[0]
    inn_records: list = []
    trade_records: list = []
    inn_match = trade_match = None
    async with GrlsStorage() as grls:
        imp = await grls.latest_import()
        for candidate in candidates:
            inn_records, inn_match = await grls.search_by_inn(candidate)
            trade_records, trade_match = (
                ([], None) if inn_records else await grls.search_by_trade_name(candidate)
            )
            if inn_records or trade_records:
                query = candidate
                break
        inn_counts, inn_revived = (
            await grls.inn_status_counts(query, kind=inn_match, on=on)
            if inn_records and inn_match
            else ({}, 0)
        )
    supplements = []
    if not inn_records and not trade_records:
        async with DietarySupplementsStorage() as supps:
            for candidate in candidates:
                supplements = await supps.search(candidate)
                if supplements:
                    query = candidate
                    break
    logger.info('💊 GRLS lookup "%s" (из %d кандидат(ов) по «%s»): inn=%d (%s) trade=%d (%s) supplements=%d',
                query, len(candidates), mention,
                len(inn_records), inn_match.value if inn_match else "—",
                len(trade_records), trade_match.value if trade_match else "—",
                len(supplements))
    return MedicineLookup(query=query, on=on, registry_date=imp.registry_date if imp else None,
                          inn_records=inn_records, inn_counts=inn_counts,
                          inn_match=inn_match, inn_valid_at_visit=inn_revived,
                          trade_records=trade_records, trade_match=trade_match,
                          supplements=supplements)
