"""Синтетический набор: искажения порождаются правилами из самих записей.

ЧЕСТНАЯ ОГОВОРКА. Искажения здесь СЛУЧАЙНЫЕ, а не человеческие: _typo ставит
произвольную букву алфавита в произвольную ВНУТРЕННЮЮ позицию. В конец не
дописывается ничего (единственная операция, задевающая хвост, — перестановка
двух последних букв), поэтому падежного «мексидола» в наборе нет как класса. А
случайная буква даёт «амокшсклав» — строку, которой человек не напишет. Отсюда
классы L2/L3 показывают лексике более тяжёлую задачу, чем реальные ошибки, и
завышают ценность вектора.

Реалистичен из набора L4: о↔а, е↔и, тс↔ц — это редукция безударных, живая
русская ошибка.

Человеческие искажения лежат в handcrafted.py; синтетика оставлена как быстрый
регресс на большом объёме и как точка отсчёта для сравнения с ним.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..corpus import FTG_MIN_CHARS, INN_LEXICAL_MAX, _trigrams, grls_norm, trgm_similarity
from .base import Dataset, Query

_RU = "абвгдежзийклмнопрстуфхцчшщыэюя"
_PHONETIC = [
    ("о", "а"), ("а", "о"), ("е", "и"), ("и", "е"), ("тс", "ц"), ("ц", "тс"),
    ("дт", "т"), ("сс", "с"), ("лл", "л"), ("нн", "н"), ("ф", "в"), ("в", "ф"),
]


def _typo(word: str, rng) -> str:
    """Одна случайная опечатка: замена, перестановка соседей, удаление, вставка."""
    if len(word) < 3:
        return word
    kind = rng.choice(("sub", "swap", "del", "ins"))
    i = rng.randrange(1, len(word) - 1)
    if kind == "sub":
        return word[:i] + rng.choice(_RU) + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "del":
        return word[:i] + word[i + 1:]
    return word[:i] + rng.choice(_RU) + word[i:]


def _phonetic(word: str, rng) -> str:
    pairs = list(_PHONETIC)
    rng.shuffle(pairs)
    for src, dst in pairs:
        if src in word:
            return word.replace(src, dst, 1)
    return _typo(word, rng)


def _partial(text: str, rng) -> str:
    """Непрерывное окно из 1–3 слов — так врач называет группу не целиком."""
    words = grls_norm(text).split()
    if len(words) <= 2:
        return " ".join(words)
    size = rng.randint(1, min(3, len(words) - 1))
    start = rng.randrange(0, len(words) - size + 1)
    return " ".join(words[start:start + size])


class Synthetic(Dataset):
    key = "synthetic"
    title = "синтетика (правила)"
    doc = "L0–L8: случайные опечатки, фонетика, МНН→торговое, ФТГ"

    def __init__(self, per_level: int = 150) -> None:
        self.per_level = per_level

    def build(self, corpus: Sequence[dict], rng) -> list[Query]:
        named = [(i, r) for i, r in enumerate(corpus)
                 if len(grls_norm(r["trade_name"])) >= 5]
        with_inn = [
            (i, r) for i, r in named
            if r.get("inn_name")
            and trgm_similarity(r["inn_name"], _trigrams(r["trade_name"])) < INN_LEXICAL_MAX
        ]
        ftg_members: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(corpus):
            key = grls_norm(r.get("pharm_group"))
            if len(key) >= FTG_MIN_CHARS:
                ftg_members[key].append(i)
        with_ftg = [(i, r) for i, r in named
                    if grls_norm(r.get("pharm_group")) in ftg_members]

        out: list[Query] = []

        def add(level: str, pool: list, make, gold_of=None) -> None:
            if not pool:
                print(f"  ВНИМАНИЕ: класс {level} пуст — пропущен")
                return
            for idx, rec in rng.sample(pool, min(self.per_level, len(pool))):
                text = make(rec)
                if not text or not grls_norm(text):
                    continue
                gold = frozenset(gold_of(rec)) if gold_of else frozenset({idx})
                if gold:
                    out.append(Query(level, text, gold))

        by_ftg = lambda r: ftg_members[grls_norm(r["pharm_group"])]  # noqa: E731

        add("L0 exact", named, lambda r: r["trade_name"])
        # L1 — по сути проверка нормализации: перед лексикой запрос всё равно
        # проходит grls_norm, который складывает и регистр, и ё. Настоящий
        # смысл класса — чувствительность ЭМБЕДДЕРА к регистру и ё: в модель
        # запрос уходит сырым.
        add("L1 typo", named,
            lambda r: grls_norm(r["trade_name"]).upper().replace("е", "ё", 1))
        add("L2 typo1", named, lambda r: _typo(grls_norm(r["trade_name"]), rng))
        add("L3 typo3", named,
            lambda r: _typo(_typo(_typo(grls_norm(r["trade_name"]), rng), rng), rng))
        add("L4 phonetic", named, lambda r: _phonetic(grls_norm(r["trade_name"]), rng))
        add("L5 inn2trade", with_inn, lambda r: r["inn_name"])
        add("L6 ftg exact", with_ftg, lambda r: r["pharm_group"], by_ftg)
        add("L7 ftg typo", with_ftg,
            lambda r: _typo(_typo(grls_norm(r["pharm_group"]), rng), rng), by_ftg)
        add("L8 ftg partial", with_ftg, lambda r: _partial(r["pharm_group"], rng), by_ftg)
        return out
