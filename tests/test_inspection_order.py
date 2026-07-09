from parsers.inspection_order import (
    reorder_inspection_data,
    _normalize,
    _levenshtein,
)


def _item(param, value=""):
    return {"Параметр": param, "Значение": value}


def test_normalize_lowercases_dedots_and_strips():
    assert _normalize("  Рекомендации и назначения:  ") == "рекомендации и назначения"
    assert _normalize("Ф20") == "ф20"
    assert _normalize("Жёлчь") == "желчь"          # ё -> е
    assert _normalize("а   б") == "а б"            # whitespace collapse


def test_levenshtein_basic_and_earlyout():
    assert _levenshtein("огол", "огол") == 0
    assert _levenshtein("листка", "листке") == 1
    assert _levenshtein("ого", "огол") == 1
    # length gap exceeds bound -> returns something > max_distance, not the exact distance
    assert _levenshtein("а", "аллергический анамнез", max_distance=2) > 2


def test_reorder_exact_manifest_order():
    data = [_item("Диагноз"), _item("Жалобы на момент осмотра"), _item("Анамнез заболевания")]
    tokens = ["жалобы на момент осмотра", "анамнез заболевания", "диагноз"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == [
        "Жалобы на момент осмотра",
        "Анамнез заболевания",
        "Диагноз",
    ]


def test_reorder_fuzzy_one_char_drift():
    data = [_item("В выдаче листке нетрудоспособности"), _item("Огол")]
    tokens = ["огол", "в выдаче листка нетрудоспособности"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == [
        "Огол",
        "В выдаче листке нетрудоспособности",
    ]


def test_unmatched_data_goes_to_tail_preserving_order():
    data = [_item("Группа здоровья"), _item("Диагноз"), _item("Заметки")]
    tokens = ["диагноз"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == ["Диагноз", "Группа здоровья", "Заметки"]


def test_unmatched_tokens_are_skipped():
    data = [_item("Диагноз")]
    tokens = ["на приеме пациент с", "диагноз", "пациент нуждается в уходе"]
    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == ["Диагноз"]


def test_length_invariant_and_no_duplication():
    data = [_item("A"), _item("B"), _item("C")]
    out = reorder_inspection_data(data, ["b"])
    assert len(out) == 3
    assert sorted(d["Параметр"] for d in out) == ["A", "B", "C"]


def test_duplicate_labels_one_claimed_rest_to_tail():
    data = [_item("Рекомендации", "first"), _item("Диагноз"), _item("Рекомендации", "second")]
    tokens = ["рекомендации", "диагноз"]
    out = reorder_inspection_data(data, tokens)
    # first "Рекомендации" claimed by token, then Диагноз, then leftover duplicate in tail
    assert [(d["Параметр"], d["Значение"]) for d in out] == [
        ("Рекомендации", "first"),
        ("Диагноз", ""),
        ("Рекомендации", "second"),
    ]


def test_no_false_match_diagnoz_vs_obosnovanie():
    # "диагноз" must NOT claim "Обоснование диагноза" (distance >> 2)
    data = [_item("Обоснование диагноза")]
    tokens = ["диагноз"]
    out = reorder_inspection_data(data, tokens)
    # single item, unmatched -> tail; still present exactly once
    assert [d["Параметр"] for d in out] == ["Обоснование диагноза"]


def test_empty_inputs():
    assert reorder_inspection_data([], ["диагноз"]) == []
    data = [_item("Диагноз")]
    assert [d["Параметр"] for d in reorder_inspection_data(data, [])] == ["Диагноз"]
