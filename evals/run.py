#!/usr/bin/env python
"""Эвал способов поиска по реестру лекарств — на настоящем Postgres.

Вопрос: нужен ли вектор поверх лексики, и какая лексика лучше. От ответа
зависит, есть ли в drug_registry движка колонка embedding, HNSW на 39 тыс.
строк и фаза эмбеддинга в синке (спека engine
docs/superpowers/specs/2026-08-20-grls-integration-design.md, §4.5 — другой
репозиторий, ветка grls-integration).

Считается НЕ приближением, а тем же движком, что будет в проде: создаётся
ВРЕМЕННАЯ таблица в БД из .env, туда кладётся корпус, строятся текстовые
индексы (GIN pg_trgm, GIN tsvector), ранжирование — средствами Postgres.
Временная таблица исчезает вместе с сессией; после сборки сессия переводится в
read-only, и методы поиска физически не могут ничего записать (evals/base.py).

Запуск:

    python -m evals.run --from-zip ~/projects/grls2026-08-17-1.zip
    python -m evals.run --from-zip … --dataset handcrafted --limit 0
    python -m evals.run --from-db "postgresql://user@host/medkard" --index name

Эндпоинт, ключ, модель и параметры Postgres берутся из .env репозитория
(EMBEDDING_BASE_URL или OPENAI_BASE_URL, OPENAI_API_KEY, EMBEDDING_MODEL,
POSTGRES_*) — как в medkard; перебиваются флагами.

ВАЖНО: выгрузка открытых данных (data-*.csv / data-*.json) НЕ годится — в ней
ноль торговых наименований и МНН (та же спека, §3.3). Имена — только xlsx-архив
ГРЛС или уже загруженная таблица grls_registry.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом, а не через -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "evals"

from .base import EvalWorkspace, mask, pg_params_from_env
from .corpus import (FTG_MIN_CHARS, dedupe, grls_norm, load_from_db, load_from_zip,
                     names_text, rest_text, search_blob)
from .datasets import BY_KEY as DATASETS
from .datasets import Handcrafted, NameIndex, Synthetic
from .embeddings import embed, load_env, to_pgvector
from .methods import ALL_METHODS, BY_KEY as METHOD_BY_KEY, MethodSet, build as build_methods

ENV_PATH = load_env()
DEFAULT_BASE_URL = (os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

SEED = 20260820
TOP_K = 10


def keep_handcrafted_gold(full: list[dict], sample: list[dict]) -> tuple[list[dict], int]:
    """Дотягивает в урезанный корпус записи, на которые ссылается ручной набор.

    Иначе --limit выбрасывает эталон случайной выборкой, и запрос становится
    неотвечаемым — метрика мерила бы не поиск, а везение выборки. Дистракторы
    при этом остаются подвыборкой, так что задача легче полного корпуса; для
    решения о схеме гонять с --limit 0.
    """
    index = NameIndex(full)
    need: set[int] = set()
    for rec in Handcrafted().load_raw():
        need |= index.resolve(rec["gold_mode"], rec["gold"])
    have = {grls_norm(r["trade_name"]) for r in sample}
    added = [full[i] for i in sorted(need)
             if grls_norm(full[i]["trade_name"]) not in have]
    return sample + added, len(added)


def report(ranks: dict, sizes: dict, methods, index_mode: str, model: str,
           corpus_size: int) -> None:
    width = 96
    print("\n" + "=" * width)
    print(f"индексируется: {index_mode}    модель: {model}    корпус: {corpus_size}")
    print(f"{'класс':<16}{'метод':<14}{'recall@1':>10}{'recall@5':>10}"
          f"{'recall@10':>11}{'MRR':>8}{'n':>6}{'|gold|':>8}")
    print("=" * width)
    for level in sorted(ranks):
        mean_gold = sum(sizes[level]) / max(len(sizes[level]), 1)
        best = (0.0, "")
        for m in methods:
            rs = ranks[level][m.key]
            if not rs:
                continue
            n = len(rs)
            r1 = sum(1 for r in rs if r <= 1) / n
            r5 = sum(1 for r in rs if r <= 5) / n
            r10 = sum(1 for r in rs if r <= TOP_K) / n
            mrr = sum(1.0 / r for r in rs) / n
            best = max(best, (mrr, m.title))
            print(f"{level:<16}{m.title:<14}{r1:>10.3f}{r5:>10.3f}{r10:>11.3f}"
                  f"{mrr:>8.3f}{n:>6}{mean_gold:>8.1f}")
        if best[1]:
            print(f"{'':<16}лучший по MRR: {best[1]}")
        print("-" * width)
    print("\nrecall@k = доля запросов, где хотя бы один правильный ответ попал в топ-k.")
    print("|gold| — среднее число правильных ответов; классы с разным |gold| между")
    print("собой несравнимы, сравнивать надо МЕТОДЫ внутри класса.\n")
    print("Как читать (спека engine §4.5):")
    print("  • trgm+tsv — то, что планируется в Искре; это базовая линия;")
    print("  • trgm+tsv+vec не лучше trgm+tsv → вектор не добавляет ничего поверх")
    print("    того, что и так будет: колонку embedding, HNSW и фазу эмбеддинга убрать;")
    print("  • выигрыш вектора на классах искажения названия → довод ЗА вектор;")
    print("  • выигрыш только на МНН и точной ФТГ → в режиме blob они и так внутри")
    print("    индексируемого текста, лексика их найдёт; взвесить трезво;")
    print("  • hybrid_even заметно лучше hybrid → вес 0.3 из 084 под лекарства не")
    print("    подходит, подбирать свой.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-zip", metavar="FILE", help="архив ГРЛС (xlsx внутри)")
    src.add_argument("--from-db", metavar="DSN", help="Postgres medkard с grls_registry")
    p.add_argument("--dataset", choices=("synthetic", "handcrafted", "both"),
                   default="both", help="какой набор запросов гонять")
    p.add_argument("--index", choices=("blob", "name"), default="blob",
                   help="что индексировать: search_blob как в спеке (по умолчанию) или только название")
    p.add_argument("--limit", type=int, default=5000,
                   help="размер корпуса (0 = все; больше корпус — честнее и дороже)")
    p.add_argument("--per-level", type=int, default=150, help="запросов на класс синтетики")
    p.add_argument("--methods", default="", help="через запятую; по умолчанию все")
    p.add_argument("--pg-dsn", default=None,
                   help="БД для временной таблицы (по умолчанию POSTGRES_* из .env)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dump-queries", metavar="FILE", help="выгрузить запросы в JSON")
    args = p.parse_args()

    try:
        import psycopg
    except ImportError:
        sys.exit("нужен psycopg (v3): pip install 'psycopg[binary]'")

    keys = [k.strip() for k in args.methods.split(",") if k.strip()]
    for k in keys:
        if k not in METHOD_BY_KEY:
            sys.exit(f"нет метода {k!r}; есть: {', '.join(METHOD_BY_KEY)}")

    rng = random.Random(SEED)
    pg = pg_params_from_env()
    print(f".env: {ENV_PATH or 'не найден — переменные только из окружения'}")
    print(f"эндпоинт: {args.base_url}")
    print(f"модель:   {args.model}  (ключ: {'задан' if args.api_key else 'ПУСТ'})")
    print(f"postgres: {args.pg_dsn and re.sub(r'//[^@]*@', '//***@', args.pg_dsn) or mask(pg)}")
    print(f"индекс:   {args.index}    набор: {args.dataset}")

    print("1/5 читаю корпус…", flush=True)
    rows = load_from_zip(args.from_zip) if args.from_zip else load_from_db(args.from_db)
    full = dedupe(rows)
    print(f"    уникальных препаратов: {len(full)}")
    corpus = full
    if args.limit and len(full) > args.limit:
        corpus = rng.sample(full, args.limit)
        print(f"    корпус сокращён до {len(corpus)} (--limit)")
        if args.dataset in ("handcrafted", "both"):
            corpus, added = keep_handcrafted_gold(full, corpus)
            print(f"    возвращено эталонов ручного набора: {added} → {len(corpus)}")
    with_ftg = sum(1 for r in corpus if len(grls_norm(r.get("pharm_group"))) >= FTG_MIN_CHARS)
    print(f"    с пригодной ФТГ: {with_ftg} ({100 * with_ftg // max(len(corpus), 1)}%)")

    print("2/5 строю запросы…", flush=True)
    sets = []
    if args.dataset in ("synthetic", "both"):
        sets.append(Synthetic(args.per_level))
    if args.dataset in ("handcrafted", "both"):
        sets.append(Handcrafted())
    queries = [q for ds in sets for q in ds.build(corpus, rng)]
    print(f"    запросов: {len(queries)}")
    if args.dump_queries:
        Path(args.dump_queries).write_text(json.dumps(
            [{"cls": q.cls, "query": q.text, "n_gold": len(q.gold), "note": q.note}
             for q in queries], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    выгружены: {args.dump_queries}")

    print(f"3/5 эмбеддинг корпуса и запросов моделью {args.model}…", flush=True)
    docs = [search_blob(r) if args.index == "blob" else r["trade_name"] for r in corpus]
    doc_vecs = embed(docs, args.base_url, args.api_key, args.model)
    q_vecs = [to_pgvector(v) for v in
              embed([q.text for q in queries], args.base_url, args.api_key, args.model)]
    dim = len(doc_vecs[0])
    print(f"    размерность: {dim}")

    print("4/5 временная таблица и текстовые индексы…", flush=True)
    conn = psycopg.connect(args.pg_dsn) if args.pg_dsn else psycopg.connect(**pg)
    with conn:
        ws = EvalWorkspace(conn)
        ws.build(
            docs_norm=[grls_norm(d) for d in docs],
            names_norm=[grls_norm(names_text(r)) for r in corpus],
            rest=[rest_text(r) for r in corpus],
            vectors=[to_pgvector(v) for v in doc_vecs],
            dim=dim,
        )
        db = ws.seal()
        methods = build_methods(db, keys or None)
        print(f"5/5 ранжирую средствами Postgres ({len(methods)} методов)…", flush=True)
        mset = MethodSet(db, methods)
        ranks: dict = defaultdict(lambda: defaultdict(list))
        sizes: dict = defaultdict(list)
        for n, (q, qv) in enumerate(zip(queries, q_vecs), start=1):
            gold = sorted(q.gold)
            sizes[q.cls].append(len(gold))
            for m, r in zip(methods, mset.ranks_of_gold(grls_norm(q.text), qv, gold)):
                ranks[q.cls][m.key].append(r)
            if n % 100 == 0:
                print(f"    {n}/{len(queries)}", flush=True)
    # временная таблица исчезла вместе с сессией — в БД ничего не осталось
    report(ranks, sizes, methods, args.index, args.model, len(corpus))


if __name__ == "__main__":
    main()
