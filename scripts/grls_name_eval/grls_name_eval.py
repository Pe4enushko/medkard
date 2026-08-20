#!/usr/bin/env python
"""Эвал: полезен ли вектор названия лекарства поверх триграммного поиска.

Вопрос, на который отвечает скрипт: обучена ли embedding-модель воспринимать
русские торговые наименования лекарств как что-то осмысленное, или на них
работает только лексическое совпадение. От ответа зависит, нужны ли в
`drug_registry` колонка `name_embedding`, HNSW-индекс и фаза эмбеддинга синка
(спека engine docs/superpowers/specs/2026-08-20-grls-integration-design.md,
§4.5 — другой репозиторий, ветка grls-integration).

Сравниваются три способа поиска по одному и тому же корпусу и одним и тем же
запросам:
  trigram  — только лексика (pg_trgm-совместимая мера, см. _trigrams);
  vector   — только косинус по эмбеддингу названия;
  hybrid   — слияние рангов, как в миграции 084: r_vec + W_TSV * r_trgm.

Классы запросов (см. build_queries):
  L0 exact      — точное название: проверка вменяемости, ожидается ~100%;
  L1 typo       — регистр, ё→е, снятие ®, лишние пробелы;
  L2 typo1      — одна опечатка;
  L3 typo3      — две-три опечатки;
  L4 phonetic   — типичные русские искажения на слух (о↔а, е↔и, тс↔ц…);
  L5 inn2trade  — ПО МНН НАЙТИ ТОРГОВОЕ. Главный класс: триграммы этого не
                  могут в принципе, и только здесь вектор может окупиться.

Запуск (нужен доступ к эндпоинту эмбеддингов):

    python grls_name_eval.py --from-zip ~/projects/grls2026-08-17-1.zip
    python grls_name_eval.py --from-db "postgresql://user@host/medkard"

ВАЖНО: выгрузка открытых данных (data-*.csv / data-*.json) для эвала НЕ
годится — в ней ноль торговых наименований и МНН (та же спека, §3.3). Имена —
только xlsx-архив ГРЛС или уже загруженная таблица grls_registry.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

# .env репозитория — грузим ДО чтения переменных, чтобы дефолты в argparse
# уже видели значения. Ищем вверх от скрипта: scripts/grls_name_eval/ → корень.
def _load_env() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate)
            except ImportError:  # без python-dotenv разбираем сами: KEY=VALUE
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return candidate
    return None


ENV_PATH = _load_env()

# Переменные — как у medkard (src/RAG/retrieval/embeddings.py): база берётся из
# EMBEDDING_BASE_URL, иначе OPENAI_BASE_URL; ключ — OPENAI_API_KEY.
DEFAULT_BASE_URL = (
    os.getenv("EMBEDDING_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
)
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Слияние рангов — как в 084_clinical_guideline_name_embedding.sql
W_TSV = 0.3
EMBED_BATCH = 64
TOP_K = 10
SEED = 20260820
# Порог для L5: пары МНН↔торговое с триграммным сходством выше — лексические
# («Церебролизин» ← «церебролизин»), для проверки семантики бесполезны.
INN_LEXICAL_MAX = 0.30

# ─────────────────────────── нормализация ───────────────────────────
# Зеркало grls_norm() канона (спека medkard §3.1): lower, схлопывание пробелов,
# удаление ®™© и кавычек, ё→е, ~→пусто. Держать идентичной канону.
_JUNK = re.compile(r"[®™©\"«»„“”'`]")
_SPACES = re.compile(r"\s+")


def grls_norm(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _JUNK.sub("", text)
    text = text.replace("ё", "е").replace("Ё", "Е").replace("~", "")
    return _SPACES.sub(" ", text).strip().lower()


# ─────────────────────────── триграммы ───────────────────────────
# pg_trgm: каждое слово дополняется двумя пробелами слева и одним справа,
# similarity = |A ∩ B| / |A ∪ B| (Jaccard). Реализация повторяет эту семантику,
# чтобы цифры эвала были сопоставимы с тем, что даст Postgres.
_WORD_SPLIT = re.compile(r"[^0-9a-zа-я]+")


def _trigrams(text: str) -> set[str]:
    out: set[str] = set()
    for word in _WORD_SPLIT.split(grls_norm(text)):
        if not word:
            continue
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            out.add(padded[i:i + 3])
    return out


def trgm_similarity(a: str, b_trg: set[str]) -> float:
    a_trg = _trigrams(a)
    if not a_trg or not b_trg:
        return 0.0
    inter = len(a_trg & b_trg)
    if not inter:
        return 0.0
    return inter / len(a_trg | b_trg)


# ─────────────────────────── чтение корпуса ───────────────────────────
def load_from_zip(path: str) -> list[dict]:
    """Торговое наименование + МНН из xlsx-архива ГРЛС.

    Раскладка листа — как в парсере medkard: строка 5 заголовки, данные с 7-й,
    колонки C..Q, торговое = индекс 6, МНН = 7 внутри среза.
    """
    try:
        import openpyxl
    except ImportError:
        sys.exit("нужен openpyxl: pip install openpyxl")

    first, n_cols = 2, 15
    rows: list[dict] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not names:
            sys.exit(f"в архиве {path} нет xlsx")
        tmp = Path("/tmp/grls_eval_sheets")
        tmp.mkdir(exist_ok=True)
        for name in names:
            target = tmp / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            wb = openpyxl.load_workbook(target, read_only=True, data_only=True)
            ws = wb[wb.sheetnames[0]]
            for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
                if i < 7:
                    continue
                cells = (tuple(row or ()) + (None,) * (first + n_cols))[first:first + n_cols]
                trade, inn = cells[6], cells[7]
                if not trade or not str(trade).strip():
                    continue
                rows.append({
                    "trade_name": str(trade).strip(),
                    "inn_name": str(inn).strip() if inn and str(inn).strip() not in ("~", "") else None,
                })
            wb.close()
            print(f"  прочитан {Path(name).name}: всего строк {len(rows)}", flush=True)
    return rows


def load_from_db(dsn: str) -> list[dict]:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        sys.exit("нужен psycopg2: pip install psycopg2-binary")
    conn = psycopg2.connect(dsn)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT trade_name, inn_name FROM grls_registry "
            "WHERE NOT is_substance AND trade_name IS NOT NULL"
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Один препарат = одно уникальное нормализованное торговое наименование.

    В реестре у одного названия десятки строк (разные упаковки и редакции РУ);
    для эвала поиска они шум: цель — найти НАЗВАНИЕ, а не строку.
    """
    seen: dict[str, dict] = {}
    for r in rows:
        key = grls_norm(r["trade_name"])
        if not key:
            continue
        if key not in seen:
            seen[key] = r
        elif not seen[key].get("inn_name") and r.get("inn_name"):
            seen[key] = r  # предпочитаем запись, где МНН заполнен
    return list(seen.values())


# ─────────────────────────── зашумление ───────────────────────────
_RU = "абвгдежзийклмнопрстуфхцчшщыэюя"
_PHONETIC = [
    ("о", "а"), ("а", "о"), ("е", "и"), ("и", "е"), ("тс", "ц"), ("ц", "тс"),
    ("дт", "т"), ("сс", "с"), ("лл", "л"), ("нн", "н"), ("ф", "в"), ("в", "ф"),
]


def _typo(word: str, rng: random.Random) -> str:
    """Одна опечатка: замена, перестановка соседей, удаление или вставка."""
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


def _phonetic(word: str, rng: random.Random) -> str:
    rng.shuffle(pairs := list(_PHONETIC))
    for src, dst in pairs:
        if src in word:
            return word.replace(src, dst, 1)
    return _typo(word, rng)


def build_queries(corpus: list[dict], per_level: int, rng: random.Random) -> list[dict]:
    """По каждому классу — per_level запросов с известным правильным ответом."""
    named = [r for r in corpus if len(grls_norm(r["trade_name"])) >= 5]
    # L5 имеет смысл только там, где МНН лексически НЕ похож на торговое:
    # «Церебролизин» ← «церебролизин» триграммы находят тривиально, и такая пара
    # ничего не проверяет. Оставляем пары, где связь только семантическая.
    with_inn = [
        r for r in named
        if r.get("inn_name")
        and trgm_similarity(r["inn_name"], _trigrams(r["trade_name"])) < INN_LEXICAL_MAX
    ]
    queries: list[dict] = []

    def add(level: str, pool: list[dict], make) -> None:
        picked = rng.sample(pool, min(per_level, len(pool)))
        for rec in picked:
            q = make(rec)
            if q and grls_norm(q):
                queries.append({"level": level, "query": q, "gold": grls_norm(rec["trade_name"])})

    add("L0 exact", named, lambda r: r["trade_name"])
    add("L1 typo", named, lambda r: grls_norm(r["trade_name"]).upper().replace("е", "ё", 1))
    add("L2 typo1", named, lambda r: _typo(grls_norm(r["trade_name"]), rng))
    add("L3 typo3", named,
        lambda r: _typo(_typo(_typo(grls_norm(r["trade_name"]), rng), rng), rng))
    add("L4 phonetic", named, lambda r: _phonetic(grls_norm(r["trade_name"]), rng))
    # Главный класс: вход — МНН, ожидаемый ответ — торговое наименование.
    add("L5 inn2trade", with_inn, lambda r: r["inn_name"])
    return queries


# ─────────────────────────── эмбеддинги ───────────────────────────
def embed(texts: list[str], base_url: str, api_key: str, model: str) -> list[list[float]]:
    out: list[list[float]] = []
    url = base_url.rstrip("/") + "/embeddings"
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [t or " " for t in texts[start:start + EMBED_BATCH]]
        body = json.dumps({"model": model, "input": batch}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        print(f"    эмбеддинг {min(start + EMBED_BATCH, len(texts))}/{len(texts)}", flush=True)
    return out


def _norm_vec(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


# ─────────────────────────── поиск и метрики ───────────────────────────
def rank_positions(scores: list[tuple[int, float]]) -> dict[int, int]:
    """Индекс документа → его ранг (1 = лучший), по убыванию score."""
    ordered = sorted(scores, key=lambda p: -p[1])
    return {idx: rank for rank, (idx, _s) in enumerate(ordered, start=1)}


def evaluate(queries, corpus, corpus_vecs, query_vecs, corpus_trg, golds) -> dict:
    stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for qi, q in enumerate(queries):
        gold_idx = golds.get(q["gold"])
        if gold_idx is None:
            continue
        # лексика
        trgm = [(i, trgm_similarity(q["query"], corpus_trg[i])) for i in range(len(corpus))]
        # вектор
        qv = query_vecs[qi]
        vec = [(i, sum(a * b for a, b in zip(qv, corpus_vecs[i]))) for i in range(len(corpus))]
        # гибрид: слияние РАНГОВ (как 084), а не сырых score — шкалы разные
        r_trgm, r_vec = rank_positions(trgm), rank_positions(vec)
        hybrid = [(i, -(r_vec[i] + W_TSV * r_trgm[i])) for i in range(len(corpus))]

        for method, scored in (("trigram", trgm), ("vector", vec), ("hybrid", hybrid)):
            rank = rank_positions(scored)[gold_idx]
            s = stats[q["level"]][method]
            s.append(rank)
    return stats


def report(stats: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'класс запросов':<16}{'метод':<10}{'recall@1':>10}{'recall@5':>10}"
          f"{'recall@10':>11}{'MRR':>8}{'n':>6}")
    print("=" * 78)
    for level in sorted(stats):
        for method in ("trigram", "vector", "hybrid"):
            ranks = stats[level][method]
            if not ranks:
                continue
            n = len(ranks)
            r1 = sum(1 for r in ranks if r <= 1) / n
            r5 = sum(1 for r in ranks if r <= 5) / n
            r10 = sum(1 for r in ranks if r <= TOP_K) / n
            mrr = sum(1.0 / r for r in ranks) / n
            print(f"{level:<16}{method:<10}{r1:>10.3f}{r5:>10.3f}{r10:>11.3f}{mrr:>8.3f}{n:>6}")
        print("-" * 78)
    print("\nКак читать: если vector и hybrid не обходят trigram НИГДЕ, кроме")
    print("L5 inn2trade — вектор названия не нужен, хватит pg_trgm (спека §4.5).")
    print("Если и на L5 вектор слаб — модель не понимает русские названия")
    print("лекарств, name_embedding/HNSW/фазу эмбеддинга из спеки убрать.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-zip", metavar="FILE", help="архив ГРЛС (xlsx внутри)")
    src.add_argument("--from-db", metavar="DSN", help="Postgres medkard с grls_registry")
    p.add_argument("--limit", type=int, default=5000,
                   help="размер корпуса уникальных названий (0 = все; больше корпус — честнее и дороже)")
    p.add_argument("--per-level", type=int, default=150, help="запросов на класс")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="по умолчанию EMBEDDING_BASE_URL / OPENAI_BASE_URL из .env")
    p.add_argument("--api-key", default=DEFAULT_API_KEY,
                   help="по умолчанию OPENAI_API_KEY из .env")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="по умолчанию EMBEDDING_MODEL из .env")
    p.add_argument("--dump-misses", metavar="FILE", help="выгрузить промахи в JSON для разбора")
    args = p.parse_args()

    rng = random.Random(SEED)

    print(f".env: {ENV_PATH or 'не найден — переменные только из окружения'}")
    print(f"эндпоинт: {args.base_url}")
    print(f"модель:   {args.model}  (ключ: {'задан' if args.api_key else 'ПУСТ'})")
    if not args.base_url:
        sys.exit("не задан эндпоинт: EMBEDDING_BASE_URL/OPENAI_BASE_URL в .env либо --base-url")

    print("1/4 читаю корпус…", flush=True)
    rows = load_from_zip(args.from_zip) if args.from_zip else load_from_db(args.from_db)
    corpus = dedupe(rows)
    print(f"    уникальных названий: {len(corpus)}")
    if args.limit and len(corpus) > args.limit:
        corpus = rng.sample(corpus, args.limit)
        print(f"    корпус сокращён до {len(corpus)} (--limit)")

    golds = {grls_norm(r["trade_name"]): i for i, r in enumerate(corpus)}
    corpus_trg = [_trigrams(r["trade_name"]) for r in corpus]

    print("2/4 строю запросы…", flush=True)
    queries = build_queries(corpus, args.per_level, rng)
    print(f"    запросов: {len(queries)}")

    print(f"3/4 эмбеддинг корпуса и запросов моделью {args.model}…", flush=True)
    corpus_vecs = [_norm_vec(v) for v in
                   embed([r["trade_name"] for r in corpus], args.base_url, args.api_key, args.model)]
    query_vecs = [_norm_vec(v) for v in
                  embed([q["query"] for q in queries], args.base_url, args.api_key, args.model)]

    print("4/4 считаю…", flush=True)
    stats = evaluate(queries, corpus, corpus_vecs, query_vecs, corpus_trg, golds)
    report(stats)

    if args.dump_misses:
        misses = []
        for q in queries:
            gold_idx = golds.get(q["gold"])
            if gold_idx is None:
                continue
            misses.append({"level": q["level"], "query": q["query"], "gold": q["gold"]})
        Path(args.dump_misses).write_text(
            json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nзапросы выгружены: {args.dump_misses}")


if __name__ == "__main__":
    main()
