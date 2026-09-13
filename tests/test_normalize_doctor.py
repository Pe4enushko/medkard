"""normalize_doctor: a clinic's doctor block in our card form (docs/clinic-data-requirements.md)."""

import copy

from parsers.doctor import normalize_doctor

ALENKA_DOCTOR = {
    "GUID": "0a99d563-9ac7-11e8-ba9c-00155d8da706",
    "FIO": "Правкина Ирина Григорьевна",
    "SPECIALIZATION": "Педиатр",
}


def _alenka_card(**overrides):
    card = {
        "Прием": {"GUID": "9491a742-59b0-429a-a05d-2d5501c39b91", "NUM": "ДКА-00223576", "DATE": "09.09.2026"},
        "Врач": dict(ALENKA_DOCTOR),
        "Пациент": {"CODE": "к0162184", "GENDER": "Мужской", "AGE": "1"},
    }
    card.update(overrides)
    return card


def _mds_card():
    return {
        "Прием": {"GUID": "ac2d8c67", "DATE": "26.07.2026", "Врач": "Губарева Елена Александровна", "Врач_код": "00012"},
        "Врач": {"SPECIALIZATION": "Невролог"},
        "Пациент": {"CODE": "Д-002315", "AGE": 67},
    }


def test_alenka_doctor_moves_into_priem():
    card = normalize_doctor(_alenka_card())
    assert card["Прием"]["Врач"] == "Правкина Ирина Григорьевна"
    assert card["Прием"]["Врач_код"] == "0a99d563-9ac7-11e8-ba9c-00155d8da706"


def test_alenka_top_block_keeps_only_specialization():
    card = normalize_doctor(_alenka_card())
    assert card["Врач"] == {"SPECIALIZATION": "Педиатр"}


def test_other_blocks_untouched():
    src = _alenka_card()
    card = normalize_doctor(copy.deepcopy(src))
    assert card["Пациент"] == src["Пациент"]
    assert card["Прием"]["NUM"] == "ДКА-00223576"


def test_mds_card_returned_as_is():
    src = _mds_card()
    assert normalize_doctor(copy.deepcopy(src)) == src


def test_card_without_doctor_block_returned_as_is():
    src = {"Прием": {"GUID": "x"}, "Пациент": {}}
    assert normalize_doctor(copy.deepcopy(src)) == src


def test_missing_guid_sets_no_code():
    card = normalize_doctor(_alenka_card(**{"Врач": {"FIO": "Иванов", "SPECIALIZATION": "Хирург"}}))
    assert card["Прием"]["Врач"] == "Иванов"
    assert "Врач_код" not in card["Прием"]
    assert card["Врач"] == {"SPECIALIZATION": "Хирург"}


def test_empty_fields_do_not_overwrite():
    card = _alenka_card(**{"Врач": {"GUID": "", "FIO": "", "SPECIALIZATION": "Хирург"}})
    card["Прием"].update({"Врач": "Штамп", "Врач_код": "D01"})
    out = normalize_doctor(card)
    assert out["Прием"]["Врач"] == "Штамп"
    assert out["Прием"]["Врач_код"] == "D01"


def test_real_doctor_overwrites_stamped_one():
    card = _alenka_card()
    card["Прием"].update({"Врач": "Выдуманный Врач", "Врач_код": "D01"})
    out = normalize_doctor(card)
    assert out["Прием"]["Врач"] == "Правкина Ирина Григорьевна"
    assert out["Прием"]["Врач_код"] == "0a99d563-9ac7-11e8-ba9c-00155d8da706"


def test_input_card_not_mutated():
    src = _alenka_card()
    snapshot = copy.deepcopy(src)
    normalize_doctor(src)
    assert src == snapshot
