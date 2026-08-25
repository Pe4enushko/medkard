#!/usr/bin/env python3
"""
Full mock audit run using synthetic visit data for today's date.

No 1C connection, no database writes — pure pipeline smoke-test.

Run from project root:
    python scripts/smoke/mock-run-today.py [--org Alenka|MDS] [--visits N] [--excel PATH]

Options:
    --org      Organization name embedded in mock data (default: Alenka)
    --visits   Number of mock visits to generate (default: 3)
    --excel    Export results to this xlsx file (default: mock_report_<date>.xlsx)
    --no-excel Skip Excel export
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit.pipeline import AuditPipeline
from RAG.retrieval.vector_store import close_pool

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--org", default="Alenka", choices=("Alenka", "MDS"))
_parser.add_argument("--visits", type=int, default=3, metavar="N")
_parser.add_argument("--excel", default=None, metavar="PATH")
_parser.add_argument("--no-excel", action="store_true")
_args = _parser.parse_args()

TODAY = datetime.now().strftime("%d.%m.%Y")
EXCEL_PATH = (
    None if _args.no_excel
    else Path(_args.excel) if _args.excel
    else ROOT / f"mock_report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Mock data ─────────────────────────────────────────────────────────────────
_MOCK_DOCTORS = [
    {"ФИО": "Иванов Иван Иванович",   "Специальность": "Терапевт"},
    {"ФИО": "Петрова Мария Сергеевна", "Специальность": "Педиатр"},
    {"ФИО": "Сидоров Алексей Юрьевич", "Специальность": "Хирург"},
]
_MOCK_PATIENTS = [
    {"ФИО": "Козлов Дмитрий Александрович", "ДатаРождения": "01.01.1980", "Пол": "М"},
    {"ФИО": "Смирнова Ольга Павловна",       "ДатаРождения": "15.06.1975", "Пол": "Ж"},
    {"ФИО": "Новиков Сергей Владимирович",   "ДатаРождения": "22.09.1990", "Пол": "М"},
]
_MOCK_DIAGNOSES = [
    [{"КодМКБ": "J06.9", "НаименованиеМКБ": "Острая инфекция верхних дыхательных путей неуточнённая", "ТипДиагноза": "Основной"}],
    [{"КодМКБ": "I10",   "НаименованиеМКБ": "Эссенциальная (первичная) гипертензия",                  "ТипДиагноза": "Основной"}],
    [{"КодМКБ": "K21.0", "НаименованиеМКБ": "Гастроэзофагеальный рефлюкс с эзофагитом",               "ТипДиагноза": "Основной"}],
]
_MOCK_ANAMNESIS = (
    "Пациент предъявляет жалобы на повышение температуры тела до 37.8°C, общую слабость, "
    "головную боль. Болеет в течение 3 дней. Ранее подобных состояний не отмечалось. "
    "Аллергологический анамнез: без особенностей. Сопутствующие заболевания отрицает."
)
_MOCK_INSPECTION = (
    "Состояние удовлетворительное. Кожные покровы обычной окраски, чистые. "
    "Лимфатические узлы не увеличены. Дыхание везикулярное, хрипов нет. ЧДД 16/мин. "
    "Тоны сердца ритмичные, ясные. ЧСС 78/мин. АД 120/80 мм рт. ст. "
    "Живот мягкий, безболезненный. Печень не увеличена."
)
_MOCK_TREATMENT = (
    "1. Постельный режим.\n"
    "2. Обильное питьё.\n"
    "3. Парацетамол 500 мг при повышении температуры свыше 38°C.\n"
    "4. Явка через 3 дня или при ухудшении состояния."
)


def _make_mock_visit(idx: int) -> dict:
    doctor  = _MOCK_DOCTORS[idx % len(_MOCK_DOCTORS)]
    patient = _MOCK_PATIENTS[idx % len(_MOCK_PATIENTS)]
    diags   = _MOCK_DIAGNOSES[idx % len(_MOCK_DIAGNOSES)]
    return {
        "Прием": {
            "GUID":        str(uuid.uuid4()),
            "DATE":        TODAY,
            "Организация": _args.org,
            "Врач":        doctor["ФИО"],
            "Специальность": doctor["Специальность"],
        },
        "Пациент": patient,
        "Диагнозы": diags,
        "Секции": {
            "Жалобы и анамнез": _MOCK_ANAMNESIS,
            "Осмотр":           _MOCK_INSPECTION,
            "Назначения":       _MOCK_TREATMENT,
        },
    }


def _make_mock_payload(n: int) -> dict:
    return {"appointments": [_make_mock_visit(i) for i in range(n)]}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("🧪 Mock run — org=%s date=%s visits=%d", _args.org, TODAY, _args.visits)

    payload = _make_mock_payload(_args.visits)
    log.info("Generated %d mock visit(s)", len(payload["appointments"]))

    try:
        # Use pipeline directly (no context manager) so no DB writes happen.
        pipeline = AuditPipeline(org_id=None)
        # Pass an empty done_guids set so the pipeline never hits the DB.
        pairs = await pipeline.run_batched(payload, num_batches=_args.visits, done_guids=set())

        log.info("Pipeline done: %d result(s)", len(pairs))
        for result, elapsed_ms in pairs:
            log.info("--- Visit result (elapsed %d ms) ---\n%s", elapsed_ms, result.pretty_format())

        if EXCEL_PATH:
            from parsers.excel import AuditExcelWriter
            writer = AuditExcelWriter(EXCEL_PATH)
            for result, _ in pairs:
                writer.append(
                    visit=result.input,
                    formal=result.formal,
                    diagnosis=result.diagnosis,
                )
            log.info("📊 Exported %d row(s) to %s", len(pairs), EXCEL_PATH)

    finally:
        await close_pool()

    log.info("✅ Mock run complete")


if __name__ == "__main__":
    asyncio.run(main())
