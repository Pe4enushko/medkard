"""Реестр способов поиска. Добавить метод = добавить класс и строчку сюда."""
from __future__ import annotations

from .base import MethodSet, SearchMethod
from .hybrid import Hybrid, HybridEven, TrgmTsvVec
from .lexical import Trigram, TrgmTsv, Tsv
from .vector import Vector

# Порядок задаёт колонки отчёта: сначала чистая лексика, потом продакшн-кандидат,
# потом вектор и смеси — так видно, что каждый следующий добавляет.
ALL_METHODS: tuple[type[SearchMethod], ...] = (
    Trigram, Tsv, TrgmTsv, Vector, Hybrid, HybridEven, TrgmTsvVec,
)

BY_KEY = {m.key: m for m in ALL_METHODS}


def build(db, keys=None) -> list[SearchMethod]:
    classes = ALL_METHODS if not keys else tuple(BY_KEY[k] for k in keys)
    return [cls(db) for cls in classes]


__all__ = ["ALL_METHODS", "BY_KEY", "MethodSet", "SearchMethod", "build",
           "Trigram", "Tsv", "TrgmTsv", "Vector", "Hybrid", "HybridEven", "TrgmTsvVec"]
