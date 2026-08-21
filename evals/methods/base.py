"""Общая база для способов поиска.

Каждый способ — класс, который умеет ровно одно: сказать, как сортировать
кандидатов. Всё остальное (доступ к БД, подсчёт сигналов, измерение позиции
эталона) общее и живёт здесь.

СИГНАЛЫ. Один проход по корпусу считает четыре оценки и четыре ранга:

    s_trgm / d_trgm    триграммы по всему индексируемому тексту;
    s_names / d_names  триграммы только по названиям (торговое + МНН);
    s_tsv  / d_tsv     to_tsvector('russian') по «остальному»;
    s_vec  / d_vec     косинус по эмбеддингу.

Ранги веток — dense_rank, как в проде (движок, миграция
084_clinical_guideline_name_embedding.sql:76-77). Это принципиально, а не
косметика: с row_number мёртвая ветка (ts_rank = 0 у всех, запрос — название)
всё равно обязана выдать кому-то ранги 1, 2, 3… и раздаёт их по id. В RRF такой
ранг весит 1/61 — ровно столько же, сколько точное попадание в живой ветке, и
записи с малым id систематически всплывают поверх настоящего ответа. Подпись
дефекта в отчёте: recall@1 = 0.06 при recall@10 = 0.95. dense_rank схлопывает
всю массу нулей в один ранг; он достаётся и эталону тоже, становится общим
слагаемым и сокращается.

ИТОГОВАЯ позиция, наоборот, считается row_number с добивкой по id: при массовых
ничьих rank отдал бы всей группе лучшую позицию и завысил метрику, а row_number
даёт ту произвольную, но реалистичную позицию, которую вернул бы обычный запрос
с LIMIT.

ОДИН ЗАПРОС НА ВСЕ МЕТОДЫ. order_by() возвращает выражение, а не готовый SQL,
поэтому MethodSet собирает из всех методов одну выборку. Иначе на каждый метод
был бы свой полный проход по корпусу — семь проходов вместо одного на каждый из
полутора тысяч запросов.
"""
from __future__ import annotations

from typing import Sequence

from ..base import ReadOnlyDB

# RRF: score = Σ 1/(K + rank). Так сливает ветки medkard (BM25 + вектор).
RRF_K = 60

_SIGNALS = """
WITH scored AS (
  SELECT id,
         similarity(doc_norm,   %(q)s)                             AS s_trgm,
         similarity(names_norm, %(q)s)                             AS s_names,
         ts_rank(rest_tsv, websearch_to_tsquery('russian', %(q)s)) AS s_tsv,
         1 - (emb <=> %(qv)s::vector)                              AS s_vec
  FROM {table}
), ranked AS (
  SELECT id, s_trgm, s_names, s_tsv, s_vec,
    dense_rank() OVER (ORDER BY s_trgm  DESC) AS d_trgm,
    dense_rank() OVER (ORDER BY s_names DESC) AS d_names,
    dense_rank() OVER (ORDER BY s_tsv   DESC) AS d_tsv,
    dense_rank() OVER (ORDER BY s_vec   DESC) AS d_vec
  FROM scored
)"""


class SearchMethod:
    """Способ поиска. Наследник обязан задать key/title и вернуть order_by().

    db — read-only (evals.base.ReadOnlyDB): к моменту создания метода сессия уже
    запечатана, записать что-либо нельзя ни случайно, ни намеренно.
    """

    key: str = ""
    title: str = ""
    doc: str = ""

    def __init__(self, db: ReadOnlyDB) -> None:
        self.db = db

    def order_by(self) -> str:
        """SQL-выражение сортировки. МЕНЬШЕ = ЛУЧШЕ (ASC).

        У методов, где больше = лучше (RRF), выражение возвращается со знаком
        минус — так контракт остаётся один на всех.
        """
        raise NotImplementedError

    def rank_of_gold(self, query: str, qvec: str, gold: Sequence[int]) -> int:
        """Позиция лучшего эталона. Отдельным запросом — для отладки одного метода."""
        return MethodSet(self.db, [self]).ranks_of_gold(query, qvec, gold)[0]


class MethodSet:
    """Все методы разом: один проход по корпусу на запрос."""

    def __init__(self, db: ReadOnlyDB, methods: Sequence[SearchMethod]) -> None:
        self.db = db
        self.methods = list(methods)

    def sql(self) -> str:
        cols = ",\n    ".join(
            f"row_number() OVER (ORDER BY ({m.order_by()}) ASC, id) AS m_{i}"
            for i, m in enumerate(self.methods)
        )
        mins = ", ".join(f"min(m_{i})" for i in range(len(self.methods)))
        return (_SIGNALS.format(table=self.db.table)
                + f", fused AS (\n  SELECT id,\n    {cols}\n  FROM ranked\n)\n"
                + f"SELECT {mins} FROM fused WHERE id = ANY(%(gold)s)")

    def ranks_of_gold(self, query: str, qvec: str, gold: Sequence[int]) -> list[int]:
        row = self.db.fetch_one(self.sql(), {"q": query, "qv": qvec, "gold": list(gold)})
        return list(row) if row else []
