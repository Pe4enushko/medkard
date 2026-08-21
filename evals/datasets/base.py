"""Общая база для наборов запросов.

Набор запросов — класс, который из корпуса делает список Query. У каждого
запроса известно МНОЖЕСТВО правильных ответов (gold): у названия это обычно
одна запись, у ФТГ — десятки, у «Креона» — все дозировки. Ранг считается по
ПЕРВОМУ релевантному, поэтому метрика читается одинаково везде.

Из-за разного |gold| классы между собой несравнимы; сравнивать надо МЕТОДЫ
внутри класса.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..corpus import grls_norm


@dataclass(frozen=True)
class Query:
    cls: str                    # класс запроса: строка отчёта
    text: str                   # что подаётся на вход поиску, БЕЗ нормализации
    gold: frozenset[int]        # индексы корпуса, считающиеся правильными
    note: str = ""

    def __post_init__(self) -> None:
        if not self.gold:
            raise ValueError(f"пустой эталон у запроса {self.text!r}")


class NameIndex:
    """Разрешение эталона по названию: точное совпадение или префикс.

    Префикс нужен потому, что голого названия в реестре часто НЕТ: «Креон»
    существует только как «Креон® 10000/25000/40000/Микро», «Мезим» — как
    «Мезим® форте». Врач при этом пишет голое имя, и это не край, а норма.
    """

    def __init__(self, corpus: Sequence[dict]) -> None:
        self.by_norm: dict[str, int] = {}
        self.sorted_norms: list[tuple[str, int]] = []
        for i, rec in enumerate(corpus):
            key = grls_norm(rec.get("trade_name"))
            if key and key not in self.by_norm:
                self.by_norm[key] = i
        self.sorted_norms = sorted(self.by_norm.items())

    def exact(self, value: str) -> set[int]:
        idx = self.by_norm.get(grls_norm(value))
        return {idx} if idx is not None else set()

    def prefix(self, value: str) -> set[int]:
        v = grls_norm(value)
        return {i for k, i in self.sorted_norms if k.startswith(v)}

    def resolve(self, mode: str, value: str) -> set[int]:
        if mode == "exact":
            return self.exact(value)
        if mode == "prefix":
            return self.prefix(value)
        raise ValueError(f"неизвестный режим эталона: {mode}")


@dataclass
class Dataset:
    """Набор запросов. Наследник задаёт key/title и реализует build()."""

    key: str = ""
    title: str = ""
    doc: str = ""

    def build(self, corpus: Sequence[dict], rng) -> list[Query]:
        raise NotImplementedError
