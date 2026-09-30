"""Операторские сводки: разбор интервала, периоды, выгрузка, отправка.

SQL этих скриптов тестами не покрыт — для него нужна база с боевой формой
`done_cards`, и проверяется он первым запуском на стенде. Здесь всё, что можно
проверить без базы: границы интервала, выражения периода, запись CSV, поведение
отправки и разбор причины падения из стектрейса.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OPERATOR = ROOT / "scripts" / "operator"
sys.path.insert(0, str(OPERATOR))

import stats_common  # noqa: E402


def _load(name: str, filename: str):
    """Скрипты названы через дефис, обычным import их не взять."""
    spec = importlib.util.spec_from_file_location(name, OPERATOR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**kwargs) -> argparse.Namespace:
    base = {"days": 7, "date_from": None, "date_to": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


# ── интервал ──────────────────────────────────────────────────────────────────

def test_days_interval_includes_both_ends():
    """--days 1 — это сегодня, а не пустой интервал."""
    start, end = stats_common.interval(_args(days=1, date_to="29.09.2026"))
    assert (start, end) == (date(2026, 9, 29), date(2026, 9, 29))

    start, end = stats_common.interval(_args(days=7, date_to="29.09.2026"))
    assert (start, end) == (date(2026, 9, 23), date(2026, 9, 29))


def test_from_wins_over_days_and_both_date_formats_are_accepted():
    start, end = stats_common.interval(
        _args(days=7, date_from="24.09.2026", date_to="2026-09-29"))
    assert (start, end) == (date(2026, 9, 24), date(2026, 9, 29))


def test_reversed_and_malformed_intervals_are_refused():
    with pytest.raises(SystemExit):
        stats_common.interval(_args(date_from="30.09.2026", date_to="24.09.2026"))
    with pytest.raises(SystemExit):
        stats_common.interval(_args(days=0, date_to="29.09.2026"))
    with pytest.raises(SystemExit):
        stats_common.interval(_args(date_from="29 сентября"))


# ── SQL-выражения ─────────────────────────────────────────────────────────────

def test_bucket_expression_per_period():
    assert stats_common.bucket_expression("ts", "day") == "(ts)::date"
    assert stats_common.bucket_expression("ts", "week") == "date_trunc('week', ts)::date"
    assert stats_common.bucket_expression("ts", "month") == "date_trunc('month', ts)::date"


def test_findings_json_guards_against_a_non_array_column():
    """У части старых карт в formal_result объект, и jsonb_array_length упал бы."""
    assert "jsonb_typeof(done_cards.formal_result) = 'array'" in stats_common.FINDINGS_JSON
    assert "'[]'::jsonb" in stats_common.FINDINGS_JSON


def test_visit_date_uses_the_tolerant_function_not_to_date():
    """Дата приёма приходит и в формате 1С, и в ISO: to_date падает на чужом."""
    assert stats_common.VISIT_DATE.startswith("medkard_visit_date(")
    assert "to_date" not in stats_common.VISIT_DATE


# ── выгрузка и отправка ───────────────────────────────────────────────────────

def test_export_path_names_the_interval_and_the_org(tmp_path, monkeypatch):
    monkeypatch.setattr(stats_common, "LOGS_DIR", tmp_path)
    path = stats_common.export_path("findings", date(2026, 9, 24), date(2026, 9, 29), "MDS")
    assert path.name == "stats-findings-2026-09-24_2026-09-29-MDS.csv"
    path = stats_common.export_path("broken", date(2026, 9, 24), date(2026, 9, 29), None)
    assert path.name == "stats-broken-2026-09-24_2026-09-29.csv"


def test_write_csv_keeps_the_declared_columns_and_drops_the_rest(tmp_path):
    path = stats_common.write_csv(
        [{"period": "2026-09-29", "flag": "F", "findings": 3, "лишнее": "не пиши"}],
        ("period", "flag", "findings"),
        tmp_path / "out.csv",
    )
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows == [{"period": "2026-09-29", "flag": "F", "findings": "3"}]


def test_offer_scp_without_an_alias_only_prints_the_command(tmp_path, monkeypatch, capsys):
    """Файл уезжает на чужую машину, поэтому без алиаса не отправляем молча."""
    calls = []
    monkeypatch.setattr(stats_common.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(stats_common.sys.stdin, "isatty", lambda: False)

    stats_common.offer_scp(tmp_path / "out.csv", None)

    assert calls == []
    assert "scp" in capsys.readouterr().out


def test_offer_scp_with_an_alias_sends_without_asking(tmp_path, monkeypatch):
    sent = []

    def fake_run(command, **kwargs):
        sent.append(command)

        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(stats_common.subprocess, "run", fake_run)
    stats_common.offer_scp(tmp_path / "out.csv", "remoteclaude")

    assert any(part[0] == "ssh" for part in sent)
    assert sent[-1][0] == "scp" and sent[-1][-1] == "remoteclaude:projects/logs/"


# ── причина падения ───────────────────────────────────────────────────────────

def test_broken_reason_is_the_last_line_of_the_stacktrace():
    broken = _load("stats_broken", "stats-broken.py")
    trace = (
        'Traceback (most recent call last):\n'
        '  File "src/audit/pipeline.py", line 1, in _audit_visit\n'
        '    raise FileNotFoundError("resources/274n_record_minimum.json")\n'
        'FileNotFoundError: resources/274n_record_minimum.json\n'
    )
    assert broken._reason(trace) == "FileNotFoundError: resources/274n_record_minimum.json"
    assert broken._reason("") == "— (стектрейс пуст)"
    assert broken._reason(None) == "— (стектрейс пуст)"


def test_target_database_is_named_and_a_local_one_is_marked(monkeypatch):
    """Выгрузка со стенда и с прода выглядят одинаково, если не сказать, откуда.

    Прогон 30.09: 127 сломанных карт из стенда были прочитаны как боевые, и
    разбор ушёл в сторону — в шапке не было базы.
    """
    monkeypatch.setenv("POSTGRES_HOST", "10.22.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "medkard_db")
    monkeypatch.setenv("POSTGRES_USER", "medkard")
    monkeypatch.setenv("POSTGRES_PASSWORD", "секрет")
    line = stats_common.target_database()
    assert line == "medkard@10.22.0.1:5432/medkard_db"
    assert "секрет" not in line

    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    assert "[локальная база — стенд]" in stats_common.target_database()
