from parsers.inspection_labels import normalize_visit_labels, rename_internal_labels


def _fields(items):
    return [i["Параметр"] for i in items]


def test_internal_1c_name_becomes_the_doctor_facing_one():
    out = rename_internal_labels([{"Параметр": "родственник лвн", "Значение": "мамой"}])
    assert out == [{"Параметр": "На приеме пациент с", "Значение": "мамой"}]


def test_case_and_colon_drift_is_tolerated():
    out = rename_internal_labels(
        [
            {"Параметр": "Родственник ЛВН:", "Значение": "папой"},
            {"Параметр": " родственник  лвн ", "Значение": "родителями"},
        ]
    )
    assert _fields(out) == ["На приеме пациент с", "На приеме пациент с"]


def test_other_lvn_fields_keep_their_names():
    """Остальные поля ЛВН — про листок нетрудоспособности, а не про сопровождающего."""
    items = [
        {"Параметр": "тип ЛВН", "Значение": "элвн"},
        {"Параметр": "с даты ЛВН", "Значение": "24.08.26"},
        {"Параметр": "работающий лвн", "Значение": "мамой"},
    ]
    assert rename_internal_labels(items) == items


def test_values_and_order_are_untouched():
    items = [
        {"Параметр": "Жалобы", "Значение": "кашель"},
        {"Параметр": "родственник лвн", "Значение": "мамой"},
        {"Параметр": "Диагноз", "Значение": "J06.9"},
    ]
    out = rename_internal_labels(items)
    assert _fields(out) == ["Жалобы", "На приеме пациент с", "Диагноз"]
    assert [i["Значение"] for i in out] == ["кашель", "мамой", "J06.9"]


def test_visit_is_copied_not_mutated():
    visit = {
        "Прием": {"GUID": "x"},
        "ДанныеОсмотра": [{"Параметр": "родственник лвн", "Значение": "мамой"}],
    }
    out = normalize_visit_labels(visit)
    assert out["ДанныеОсмотра"][0]["Параметр"] == "На приеме пациент с"
    assert visit["ДанныеОсмотра"][0]["Параметр"] == "родственник лвн"
    assert out["Прием"] is visit["Прием"]


def test_visit_without_inspection_data_passes_through():
    for visit in ({"Прием": {}}, {"ДанныеОсмотра": []}, {"ДанныеОсмотра": None}):
        assert normalize_visit_labels(visit) is visit
