"""Лексические способы: триграммы и tsvector."""
from __future__ import annotations

from .base import RRF_K, SearchMethod


class Trigram(SearchMethod):
    key = "trigram"
    title = "trigram"
    doc = "pg_trgm по всему индексируемому тексту"

    def order_by(self) -> str:
        return "d_trgm"


class Tsv(SearchMethod):
    key = "tsv"
    title = "tsv"
    doc = "to_tsvector('russian') по «остальному» (ФТГ, формы, отпуск, держатель), без названий"

    def order_by(self) -> str:
        return "d_tsv"


class TrgmTsv(SearchMethod):
    """Продакшн-кандидат: то, что планируется в Искре, — базовая линия.

    Триграммы по названиям (торговое + МНН) плюс tsv по остальному, слияние
    через RRF. Остальные методы оцениваются тем, насколько они его обходят.
    """

    key = "trgm+tsv"
    title = "trgm+tsv"
    doc = "ПРОДАКШН-КАНДИДАТ: триграммы по названиям + tsv по остальному, RRF"

    def order_by(self) -> str:
        return f"-(1.0/({RRF_K} + d_names) + 1.0/({RRF_K} + d_tsv))"
