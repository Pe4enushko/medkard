"""Чистая семантика: косинус по эмбеддингу."""
from __future__ import annotations

from .base import SearchMethod


class Vector(SearchMethod):
    key = "vector"
    title = "vector"
    doc = "только косинус по эмбеддингу (pgvector)"

    def order_by(self) -> str:
        return "d_vec"
