"""Open the registries and assemble a MedicineLookup (I/O lives here, formatting in format.py)."""
from __future__ import annotations

import logging
from datetime import date

from grls.format import TRADE_THRESHOLD, MedicineLookup
from storage.dietary_supplements_storage import DietarySupplementsStorage
from storage.grls_storage import GrlsStorage

logger = logging.getLogger(__name__)


async def lookup_medicine(query: str, *, on: date | None = None) -> MedicineLookup:
    """INN first, then trade name, then dietary supplements — same order as the old tool."""
    async with GrlsStorage() as grls:
        imp = await grls.latest_import()
        inn_records = await grls.search_by_inn(query)
        inn_counts = await grls.inn_status_counts(query) if inn_records else {}
        trade_records = [] if inn_records else await grls.search_by_trade_name(query, threshold=TRADE_THRESHOLD)
    supplements = []
    if not inn_records and not trade_records:
        async with DietarySupplementsStorage() as supps:
            supplements = await supps.search(query)
    logger.info('💊 GRLS lookup "%s": inn=%d trade=%d supplements=%d',
                query, len(inn_records), len(trade_records), len(supplements))
    return MedicineLookup(query=query, on=on, registry_date=imp.registry_date if imp else None,
                          inn_records=inn_records, inn_counts=inn_counts,
                          trade_records=trade_records, supplements=supplements)
