"""Отказ replace_all на пустом наборе — до всякой БД.

Полная замена начинается с DELETE. Пустой список почти всегда означает сбой
разбора выгрузки, а не «реестр отменили», и молча вычистить справочник — худший
из исходов: аудит останется без БАД, а заметят это по жалобе, а не по ошибке.
"""
import pytest

from storage.dietary_supplements_storage import DietarySupplementsStorage


async def test_replace_all_refuses_an_empty_registry():
    with pytest.raises(ValueError) as e:
        await DietarySupplementsStorage().replace_all([])
    assert "не трогаем" in str(e.value)
