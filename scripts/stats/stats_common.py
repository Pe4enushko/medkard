"""
stats_common.py — общее для операторских сводок по `done_cards`.

Интервал, запись выгрузки и отправка файла по scp. Лежит рядом со скриптами,
которые это используют: `stats-findings.py` и `stats-broken.py`. Отдельным
модулем, а не копией в каждом, потому что разбор интервала и отправка обязаны
вести себя одинаково — сводки сравнивают между собой.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"

# `formal_result` не всегда массив: у части старых карт там объект или null, и
# `list_formal_results_to_backfill` в storage не зря фильтрует по jsonb_typeof.
# jsonb_array_length на объекте падает, поэтому приводим к массиву в самом SQL.
FINDINGS_JSON = (
    "CASE WHEN jsonb_typeof(done_cards.formal_result) = 'array' "
    "THEN done_cards.formal_result ELSE '[]'::jsonb END"
)

# Дата приёма приходит и в формате 1С, и в ISO; to_date на чужом формате падает.
# Функция из миграции 026 разбирает оба и возвращает NULL на неразобранном.
VISIT_DATE = "medkard_visit_date(done_cards.card_data -> 'Прием' ->> 'DATE')"

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d")


def target_database() -> str:
    """Куда скрипт подключился — строкой для шапки отчёта.

    Печатается всегда: выгрузка со стенда и выгрузка с прода выглядят одинаково,
    и спутать их — значит чинить не ту базу. Пароль сюда не попадает.
    """
    import os

    host = os.environ.get("POSTGRES_HOST", "?")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "?")
    user = os.environ.get("POSTGRES_USER", "?")
    local = host in ("127.0.0.1", "localhost", "::1")
    return f"{user}@{host}:{port}/{name}" + ("   [локальная база — стенд]" if local else "")


def add_interval_arguments(parser: Any) -> None:
    """--days и --from/--to. Даты в обоих привычных форматах: 1С и ISO."""
    parser.add_argument(
        "--days", type=int, default=7,
        help="интервал в днях, считая назад от сегодня (по умолчанию 7)")
    parser.add_argument(
        "--from", dest="date_from", metavar="ДАТА",
        help="начало интервала, ДД.ММ.ГГГГ или ГГГГ-ММ-ДД; отменяет --days")
    parser.add_argument(
        "--to", dest="date_to", metavar="ДАТА",
        help="конец интервала включительно; по умолчанию сегодня")


def parse_date(raw: str) -> date:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"не разобрал дату {raw!r}: нужен формат ДД.ММ.ГГГГ или ГГГГ-ММ-ДД")


def interval(args: Any) -> tuple[date, date]:
    """(начало, конец) включительно. Конец — сегодня, если не задан."""
    end = parse_date(args.date_to) if args.date_to else date.today()
    if args.date_from:
        start = parse_date(args.date_from)
    else:
        if args.days < 1:
            raise SystemExit("--days должен быть не меньше 1")
        start = end - timedelta(days=args.days - 1)
    if start > end:
        raise SystemExit(f"начало интервала ({start}) позже конца ({end})")
    return start, end


def bucket_expression(column: str, bucket: str) -> str:
    """SQL-выражение колонки периода. Неделя — по понедельникам (date_trunc)."""
    if bucket == "week":
        return f"date_trunc('week', {column})::date"
    if bucket == "month":
        return f"date_trunc('month', {column})::date"
    return f"({column})::date"


def print_table(rows: Sequence[dict], columns: Sequence[tuple[str, str, int]]) -> None:
    """columns — (ключ, заголовок, ширина). Значение шире колонки обрезается."""
    def cell(value: object, width: int) -> str:
        text = "—" if value is None else str(value)
        return text.ljust(width) if len(text) <= width else text[: width - 1] + "…"

    header = "  ".join(cell(title, width) for _, title, width in columns)
    rule = "  ".join("─" * width for _, _, width in columns)
    print(header)
    print(rule)
    for row in rows:
        print("  ".join(cell(row.get(key), width) for key, _, width in columns))
    print(rule)
    print(f"  строк: {len(rows)}")


def write_csv(rows: Iterable[dict], fieldnames: Sequence[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            written += 1
    print(f"\nвыгрузка: {path}  ({written} строк)")
    return path


def export_path(kind: str, start: date, end: date, org: str | None) -> Path:
    suffix = f"-{org}" if org else ""
    return LOGS_DIR / f"stats-{kind}-{start:%Y-%m-%d}_{end:%Y-%m-%d}{suffix}.csv"


def offer_scp(path: Path, alias: str | None, dest: str = "projects/logs") -> None:
    """Отправить выгрузку на ssh-алиас.

    Алиас передан — отправляем без вопросов (годится для cron). Не передан и
    терминал интерактивный — спрашиваем, и пустой ответ означает «не надо».
    Иначе просто печатаем готовую команду: решение за человеком, а не за
    скриптом, потому что файл уезжает на чужую машину.
    """
    if alias is None and sys.stdin.isatty():
        answer = input(f"Отправить {path.name} по scp? ssh-алиас (Enter — не надо): ").strip()
        alias = answer or None
    if not alias:
        print(f"не отправлено; когда понадобится:  scp {path} <алиас>:{dest}/")
        return
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, f"mkdir -p '{dest}'"],
            check=True, capture_output=True,
        )
        subprocess.run(["scp", "-p", "-q", str(path), f"{alias}:{dest}/"], check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"нет клиента ssh/scp: {exc}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace").strip()
        raise SystemExit(f"не отправилось на {alias!r}: {stderr or exc}")
    print(f"отправлено: {alias}:{dest}/{path.name}")
