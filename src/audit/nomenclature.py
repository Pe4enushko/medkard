"""Номенклатура медицинских услуг 804н — общий доступ к выгрузке.

Выгрузка лежит в ``resources/nomenclature-804n.csv``: 10 441 код, экспорт
текстового слоя приказа. Пользуются ею два места — классификатор вида приёма
в ``formal_structure`` и детерминистичные правила, — поэтому разбор здесь,
а не в каждом по копии.

Выгрузка заморожена на редакции с изм. 24.09.2020: код из более новой
редакции здесь не найдётся, и вызывающий обязан трактовать это как «не знаю»,
а не как «такого кода нет».
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PATH = Path(__file__).resolve().parents[2] / "resources" / "nomenclature-804n.csv"


@lru_cache(maxsize=1)
def names() -> dict[str, str]:
    """Код услуги → наименование по приказу. Пустой словарь, если файла нет."""
    try:
        with open(PATH, encoding="utf-8", newline="") as f:
            return {
                row["code"].strip().upper(): row["name"].strip()
                for row in csv.DictReader(f)
                if row.get("code") and row.get("name")
            }
    except OSError:
        logger.warning("[nomenclature] %s не прочитан — сверка с 804н недоступна", PATH)
        return {}


def name_of(code: str) -> str | None:
    """Наименование услуги по коду, или None если код в выгрузке не найден."""
    return names().get(code.strip().upper())
