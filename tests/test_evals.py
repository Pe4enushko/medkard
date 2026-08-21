"""Эвалы поиска по лекарствам — всё, что проверяется без Postgres и эндпоинта."""
import json
import re

import pytest

from evals.base import ReadOnlyDB
from evals.corpus import grls_norm, search_blob
from evals.datasets import Handcrafted, NameIndex, Query, Synthetic
from evals.datasets.handcrafted import DATA
from evals.methods import ALL_METHODS, BY_KEY, MethodSet, build

KINDS = {"inflect", "suffix", "prefix", "tail", "dose", "form", "translit",
         "mixed", "phonetic", "space", "root", "dictation"}


class _FakeDB(ReadOnlyDB):
    def __init__(self):
        self.table = "eval_docs"


def _corpus(*names):
    return [{"trade_name": n, "inn_name": None, "pharm_group": None,
             "forms_raw": None, "holder": None} for n in names]


# ─────────────────────────── разрешение эталона ───────────────────────────
def test_exact_ignores_case_registered_sign_and_yo():
    idx = NameIndex(_corpus("Мезим® форте", "Нурофен® для детей"))
    assert idx.exact("нурофен для детей") == {1}
    assert idx.exact("МЕЗИМ ФОРТЕ") == {0}


def test_prefix_collects_all_dosages():
    idx = NameIndex(_corpus("Креон® 10000", "Креон® 25000", "Кревит"))
    assert idx.prefix("креон") == {0, 1}


def test_query_without_gold_is_rejected():
    with pytest.raises(ValueError):
        Query("tail", "креон", frozenset())


# ─────────────────────────── ручной набор ───────────────────────────
def test_handcrafted_file_is_well_formed():
    raw = json.loads(DATA.read_text(encoding="utf-8"))["queries"]
    assert len(raw) >= 100, "набор задуман не меньше сотни искажений"
    assert {r["kind"] for r in raw} <= KINDS
    assert len({r["query"] for r in raw}) == len(raw), "дубли запросов"
    for r in raw:
        assert r["gold_mode"] in ("exact", "prefix")
        assert r["query"].strip() and r["gold"].strip()


def test_handcrafted_query_differs_from_gold():
    """Каверканье обязано отличаться от эталона — иначе это класс L0, а не искажение.

    Кроме режима prefix: там запрос как раз совпадает с началом эталона, и в
    этом всё содержание класса tail — голого «Креона» в реестре не существует.
    """
    same = [r for r in Handcrafted().load_raw()
            if r["gold_mode"] == "exact" and grls_norm(r["query"]) == grls_norm(r["gold"])]
    assert not same, same


def test_handcrafted_resolves_against_corpus():
    raw = Handcrafted().load_raw()
    corpus = _corpus(*{r["gold"] for r in raw if r["gold_mode"] == "exact"},
                     *{r["gold"] + " 100" for r in raw if r["gold_mode"] == "prefix"})
    assert len(Handcrafted().build(corpus)) == len(raw)


def test_handcrafted_fails_loudly_when_gold_missing():
    """Молчаливая потеря эталонов читалась бы как «метод стал лучше»."""
    with pytest.raises(SystemExit) as e:
        Handcrafted().build(_corpus("Что-то постороннее"))
    assert "не найдено в корпусе" in str(e.value)


# ─────────────────────────── синтетика ───────────────────────────
def test_synthetic_typo_never_appends_to_the_word():
    """Фиксируем известное ограничение: падежного «мексидола» синтетика не даёт.

    Ради этого и существует ручной набор — если ограничение когда-нибудь снимут,
    тест упадёт и заставит переписать оговорку в handcrafted.py. Последнюю букву
    _typo задеть может (перестановкой двух конечных), а вот дописать в конец —
    нет, и именно дописывание порождает падеж.
    """
    import random

    from evals.datasets.synthetic import _typo

    rng = random.Random(1)
    word = "мексидол"
    out = [_typo(word, rng) for _ in range(500)]
    assert not any(len(o) > len(word) and o.startswith(word) for o in out)


def test_synthetic_builds_all_classes():
    import random

    corpus = [{"trade_name": f"Препарат{i}", "inn_name": f"вещество{i}",
               "pharm_group": "противоопухолевое средство растительного происхождения",
               "forms_raw": "таблетки, 5 мг - отпускают по рецепту", "holder": "Держатель"}
              for i in range(30)]
    qs = Synthetic(per_level=5).build(corpus, random.Random(0))
    assert {q.cls for q in qs} == {
        "L0 exact", "L1 typo", "L2 typo1", "L3 typo3", "L4 phonetic",
        "L5 inn2trade", "L6 ftg exact", "L7 ftg typo", "L8 ftg partial"}
    assert all(q.gold for q in qs)


# ─────────────────────────── методы ───────────────────────────
def test_registry_keys_unique():
    assert len({m.key for m in ALL_METHODS}) == len(ALL_METHODS)
    assert set(BY_KEY) == {m.key for m in ALL_METHODS}


def test_sql_uses_dense_rank_for_branches_and_row_number_for_position():
    """Суть починки: ранги веток — dense_rank, итоговая позиция — row_number."""
    sql = MethodSet(_FakeDB(), build(_FakeDB())).sql()
    branches = re.findall(r"dense_rank\(\) OVER \(ORDER BY (\w+)", sql)
    assert set(branches) == {"s_trgm", "s_names", "s_tsv", "s_vec"}
    assert "row_number() OVER (ORDER BY (d_" in sql or "row_number() OVER (ORDER BY (-" in sql
    assert sql.count("row_number") == len(ALL_METHODS)


def test_sql_parses_as_postgres():
    pglast = pytest.importorskip("pglast")
    sql = MethodSet(_FakeDB(), build(_FakeDB())).sql()
    pglast.parse_sql(re.sub(r"%\((\w+)\)s", r"$1", sql))


def _rank_with(method, docs):
    """Считает порядок по формуле метода на подставных сигналах.

    Выражения order_by() — обычная арифметика, одинаково читаемая Postgres и
    Python, поэтому формулу можно проверить, не поднимая базу.
    """
    scored = [(eval(method.order_by(), {}, d), d["id"]) for d in docs]  # noqa: S307
    return [i for _, i in sorted(scored)]


def test_dead_branch_does_not_sink_the_gold():
    """Регресс на дефект, из-за которого trgm+tsv давал recall@1 = 0.06.

    Запрос — название, ветка tsv мертва: ts_rank = 0 у ВСЕХ, поэтому dense_rank
    выдаёт всем один и тот же ранг. Эталон обязан остаться первым.
    """
    dead = 1  # единственный ранг, который dense_rank даёт всей массе нулей
    docs = [{"id": 0, "d_names": 1, "d_trgm": 1, "d_tsv": dead, "d_vec": 1}]
    docs += [{"id": i, "d_names": 500 + i, "d_trgm": 500 + i, "d_tsv": dead,
              "d_vec": 500 + i} for i in range(1, 20)]
    for cls in ALL_METHODS:
        if cls.key == "tsv":
            continue  # метод целиком состоит из мёртвой ветки — сравнивать нечего
        assert _rank_with(cls(_FakeDB()), docs)[0] == 0, cls.key


def test_hybrid_weight_is_the_only_difference_between_hybrids():
    from evals.methods import Hybrid, HybridEven

    assert Hybrid.weight == 0.3, "вес из миграции 084"
    assert HybridEven.weight == 1.0
    assert Hybrid(_FakeDB()).order_by() != HybridEven(_FakeDB()).order_by()


# ─────────────────────────── корпус ───────────────────────────
def test_search_blob_keeps_names_first():
    rec = {"trade_name": "Конкор®", "inn_name": "Бисопролол",
           "pharm_group": "бета-адреноблокатор",
           "forms_raw": "таблетки, покрытые оболочкой, 5 мг - по рецепту",
           "holder": "Мерк"}
    blob = search_blob(rec)
    assert blob.startswith("Конкор® | Бисопролол")
    assert "бета-адреноблокатор" in blob
