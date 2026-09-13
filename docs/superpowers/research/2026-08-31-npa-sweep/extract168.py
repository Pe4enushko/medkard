"""Выгрузка приложений 1-3 приказа 168н в CSV.

Источник — экспорт КонсультантПлюс редакции от 28.02.2024 с текстовым слоем.
Строки таблиц рвутся между страницами: продолжение приходит без номера и кода,
поэтому такие куски дописываются в последнюю открытую запись.
"""
import csv, re, sys
import pymupdf

SRC = sys.argv[1]
OUT = sys.argv[2]

COLS = ["n", "code", "name", "periodicity", "indicators", "duration", "who"]
APPENDIX_RE = re.compile(r"Приложение\s+N\s*(\d)\s*\n?\s*к [Пп]орядку проведения диспансерного")
# служебные строки колонтитула, которые попадают в ячейки
NOISE = re.compile(
    r"КонсультантПлюс|надежная правовая поддержка|www\.consultant\.ru|"
    r"Страница\s+\d+\s+из\s+\d+|Документ предоставлен|Дата сохранения|"
    r"Приказ Минздрава России от 15\.03\.2022|\(ред\. от 28\.02\.2024\)|"
    r"Об утверждении порядка проведения диспансерного наблю"
)


def clean(cell: str) -> str:
    if not cell:
        return ""
    lines = [ln for ln in cell.split("\n") if ln.strip() and not NOISE.search(ln)]
    text = " ".join(lines)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    # перенос внутри слова: PDF рвёт слово без дефиса. Список хвостов собран
    # перебором всех коротких правых фрагментов в выгрузке; «моче» и «век»
    # намеренно не входят — «суточной моче» и «спайку век» это два слова.
    tails = ("ми", "ния", "ний", "ями", "ины", "вая", "льно", "лога", "га",
             "мах", "ного", "ского", "ской", "сти")
    return join_wraps(text)


def join_wraps(text: str) -> str:
    tails = ("ми", "ния", "ний", "ями", "ины", "вая", "льно", "лога", "га",
             "мах", "ного", "ского", "ской", "сти")
    return re.sub(r"([А-Яа-яЁё]{3,})\s+(" + "|".join(tails) + r")\b",
                  r"\1\2", text)


def is_row_start(cells):
    """Строка начинает новую запись, если в ней есть и номер, и код."""
    n, code = clean(cells[0]), clean(cells[1])
    return bool(re.fullmatch(r"\d+\.?", n)) and bool(code)


def column_bounds(table):
    """Границы семи колонок по первой полной строке и верх таблицы.

    Верх нужен, чтобы отрезать колонтитул: на странице без таблицы он
    переносится на две строки, и вторая («проведения диспансерного») мимо
    NOISE проходит.
    """
    for row in table.rows:
        if len(row.cells) >= 7 and all(c is not None for c in row.cells[:7]):
            return [(c[0], c[2]) for c in row.cells[:7]], table.bbox[1]
    return None


def rows_by_columns(page, layout):
    """Страница без распознанной таблицы: слова раскладываются по колонкам.

    find_tables() не видит таблицу на страницах, целиком занятых продолжением
    одной высокой строки, — у такой страницы нет горизонтальных линий. Первая
    выгрузка эти страницы молча пропускала, и из середины строк выпадал текст
    (прил. 2 № 1 и № 3 — стр. 28, 29, 32; прил. 1 — стр. 25; прил. 2 — стр. 41).
    Границы колонок берутся с ближайшей предыдущей страницы с таблицей: вёрстка
    приложения одна. Всё ниже линии сносок и выше верха таблицы отбрасывается.
    """
    bounds, top = layout
    words = page.get_text("words")
    cut = min(
        (w[1] for w in words if w[4].startswith("-----")),
        default=float("inf"),
    )
    lines = {}
    for x0, y0, x1, y1, text, block, line, _ in words:
        if y0 >= cut or y1 <= top:
            continue
        lines.setdefault((block, line), []).append((x0, y0, x1, text))
    cols = [[] for _ in bounds]
    for key in sorted(lines, key=lambda k: (lines[k][0][1], lines[k][0][0])):
        line_text = " ".join(w[3] for w in lines[key])
        if NOISE.search(line_text):
            continue
        for x0, y0, x1, text in lines[key]:
            centre = (x0 + x1) / 2
            for i, (left, right) in enumerate(bounds):
                if left <= centre <= right:
                    cols[i].append((y0, x0, text))
                    break
    return [" ".join(t for _, _, t in sorted(c)) for c in cols]


doc = pymupdf.open(SRC)
records = []
appendix = None
skipped_header_rows = 0
layout = None
recovered_pages = []

for pno in range(doc.page_count):
    page = doc[pno]
    raw = page.get_text()
    m = APPENDIX_RE.search(raw.replace("\n", " ")) or APPENDIX_RE.search(raw)
    if m:
        appendix = int(m.group(1))
    if appendix is None:
        continue
    tables = page.find_tables().tables
    for table in tables:
        layout = column_bounds(table) or layout
    if not tables and layout and records and records[-1]["appendix"] == appendix:
        recovered = rows_by_columns(page, layout)
        if any(v.strip() for v in recovered[2:]):
            recovered_pages.append(pno + 1)
            tables = [None]
    for table in tables:
        extracted = [recovered] if table is None else table.extract()
        for cells in extracted:
            cells = list(cells) + [""] * (7 - len(cells))
            cells = cells[:7]
            vals = [clean(c) for c in cells]
            if not any(vals):
                continue
            if is_row_start(cells):
                rec = dict(zip(COLS, vals))
                rec["n"] = rec["n"].rstrip(".")
                rec["appendix"] = appendix
                rec["pages"] = str(pno + 1)
                records.append(rec)
            elif records and records[-1]["appendix"] == appendix:
                # продолжение предыдущей записи
                rec = records[-1]
                for col, v in zip(COLS, vals):
                    if not v:
                        continue
                    if col in ("n", "code"):
                        continue
                    rec[col] = (rec[col] + " " + v).strip() if rec[col] else v
                if str(pno + 1) not in rec["pages"].split(","):
                    rec["pages"] += "," + str(pno + 1)
            else:
                skipped_header_rows += 1

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "приложение", "n", "код_мкб", "наименование",
        "периодичность", "контролируемые_показатели", "длительность",
        "кто_ведет_условие", "страницы_pdf",
    ])
    for r in records:
        # повторно — склейки ячеек через страницу могли создать новый разрыв
        w.writerow([
            r["appendix"], r["n"], r["code"], join_wraps(r["name"]),
            join_wraps(r["periodicity"]), join_wraps(r["indicators"]),
            join_wraps(r["duration"]), join_wraps(r["who"]), r["pages"],
        ])

by_app = {}
for r in records:
    by_app[r["appendix"]] = by_app.get(r["appendix"], 0) + 1
print("записей:", len(records), "по приложениям:", dict(sorted(by_app.items())))
print("пропущено строк без записи:", skipped_header_rows)
print("страницы, восстановленные раскладкой по колонкам:", recovered_pages)
