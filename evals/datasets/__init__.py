"""Реестр наборов запросов. Добавить набор = добавить класс и строчку сюда."""
from __future__ import annotations

from .base import Dataset, NameIndex, Query
from .handcrafted import Handcrafted
from .synthetic import Synthetic

BY_KEY = {"synthetic": Synthetic, "handcrafted": Handcrafted}

__all__ = ["Dataset", "NameIndex", "Query", "Synthetic", "Handcrafted", "BY_KEY"]
