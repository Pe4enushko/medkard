"""Смешанные способы: лексика плюс вектор."""
from __future__ import annotations

from .base import RRF_K, SearchMethod


class Hybrid(SearchMethod):
    """Форма из миграции 084: сумма рангов с весом, подобранным под клинреки."""

    key = "hybrid"
    title = "hybrid"
    doc = "vector + trigram с весом, как в 084: d_vec + 0.3*d_trgm"
    weight = 0.3

    def order_by(self) -> str:
        return f"d_vec + {self.weight} * d_trgm"


class HybridEven(Hybrid):
    """То же без веса.

    Нужен потому, что 0.3 подобран под реестр клинреков на 747 строк и на
    лекарствах может не подойти: если этот метод заметно лучше, коэффициент
    надо подбирать свой, а не переносить из 084.
    """

    key = "hybrid_even"
    title = "hybrid_even"
    doc = "vector + trigram без весов"
    weight = 1.0


class TrgmTsvVec(SearchMethod):
    """Продакшн-кандидат плюс вектор — сколько вектор ДОБАВЛЯЕТ к тому, что и так будет."""

    key = "trgm+tsv+vec"
    title = "trgm+tsv+vec"
    doc = "продакшн-кандидат плюс вектор, слияние через RRF"

    def order_by(self) -> str:
        return (f"-(1.0/({RRF_K} + d_names) + 1.0/({RRF_K} + d_tsv)"
                f" + 1.0/({RRF_K} + d_vec))")
