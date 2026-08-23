"""Уровни сопоставления запроса с записью реестра.

Порядок: точное равенство → вхождение подстроки → триграммы. Смысл порядка в
том, что похожесть строки плохо отличает «слово добавили» от «слово заменили»:

    Левотироксин натрия  ↔ Левотироксин   0.650  — одно и то же
    Преднизолон          ↔ Преднизон      0.692  — разные препараты

Правильное совпадение лежит НИЖЕ неправильного, поэтому порогом их не развести.
Вхождение подстроки разводит: «метопролола сукцинат» содержит «метопролол»,
а «преднизон» в «преднизолон» не содержится.

Оставшийся класс — короткий различитель, который сам является префиксом
другого: «витамин в1» ⊂ «витамин в12» (тиамин против цианокобаламина). Его
ловит `discriminators_agree`: такой токен обязан совпасть отдельным словом.
"""
from __future__ import annotations

from enum import Enum

from grls.normalize import normalize_query

# Короче — уже не название, а обрывок: «ин» нашлось бы в половине реестра.
MIN_CONTAINED_LEN = 4
# Порог триграммного фоллбэка. Ниже — шум, выше — теряются солевые формы.
FUZZY_THRESHOLD = 0.6
# Токен такой длины различает препараты, а не описывает их: «в6», «d3».
SHORT_TOKEN_MAX_LEN = 2


class MatchKind(str, Enum):
    """Как запрос сошёлся с записью. Уровень обязан доезжать до ответа."""

    EXACT = "exact"
    CONTAINS = "contains"
    FUZZY = "fuzzy"


def discriminator_tokens(query: str) -> list[str]:
    """Токены, которые отличают препарат от соседа и обязаны совпасть точно.

    Короткие («в6», «d3») и любые с цифрой («в12»): именно в них разница между
    тиамином и цианокобаламином, и именно её триграммы не замечают.
    """
    return [
        token
        for token in normalize_query(query).split()
        if (len(token) <= SHORT_TOKEN_MAX_LEN or any(ch.isdigit() for ch in token))
        and any(ch.isalpha() or ch.isdigit() for ch in token)
    ]


def discriminators_agree(query: str, candidate: str) -> bool:
    """Различители запроса присутствуют в кандидате ОТДЕЛЬНЫМИ словами.

    Сравнение по подстроке здесь не годится: «в1» лежит внутри «в12».
    """
    candidate_tokens = normalize_query(candidate).split()
    return all(token in candidate_tokens for token in discriminator_tokens(query))


def contains(query: str, candidate: str) -> bool:
    """Одна нормализованная строка входит в другую, в любую сторону.

    В обе стороны — потому что врач пишет и короче реестра («Метопролол» при
    «Метопролола сукцинат»), и длиннее («Левотироксин натрия» при
    «Левотироксин»).
    """
    q, c = normalize_query(query), normalize_query(candidate)
    if not q or not c:
        return False
    shorter, longer = (q, c) if len(q) <= len(c) else (c, q)
    return len(shorter) >= MIN_CONTAINED_LEN and shorter in longer


def classify(query: str, candidate: str) -> MatchKind | None:
    """Уровень совпадения либо None. Чистая функция — зеркало SQL из GrlsStorage."""
    q, c = normalize_query(query), normalize_query(candidate)
    if not q or not c:
        return None
    if q == c:
        return MatchKind.EXACT
    if not discriminators_agree(query, candidate):
        return None
    if contains(query, candidate):
        return MatchKind.CONTAINS
    return MatchKind.FUZZY if _similarity(q, c) >= FUZZY_THRESHOLD else None


def _similarity(a: str, b: str) -> float:
    """pg_trgm.similarity: триграммы слов с добивкой, |A∩B| / |A∪B|.

    Держится здесь ради тестов и отладки без базы; в проде считает Postgres.
    """
    return _jaccard(_trigrams(a), _trigrams(b))


def _trigrams(text: str) -> set[str]:
    out: set[str] = set()
    for word in text.split():
        padded = f"  {word} "
        out.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0
