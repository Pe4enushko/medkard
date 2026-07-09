from parsers.inspection_order import load_inspection_format, reorder_inspection_data


def test_alenka_manifest_orders_a_realistic_fragment():
    tokens = load_inspection_format("Alenka", "standard")
    # a realistic, out-of-order Alenka ДанныеОсмотра fragment (labels as they
    # appear in real exports, incl. the trailing-colon case)
    data = [
        {"Параметр": "Направление к другому специалисту (списком)", "Значение": "Лаборатория"},
        {"Параметр": "Рекомендации и назначения:", "Значение": "..."},
        {"Параметр": "Диагноз", "Значение": "J06.9"},
        {"Параметр": "Жалобы на момент осмотра", "Значение": "кашель"},
        {"Параметр": "Анамнез заболевания", "Значение": "3 дня"},
        {"Параметр": "Температура", "Значение": "37"},
        {"Параметр": "ЧСС", "Значение": "80"},
    ]
    out = reorder_inspection_data(data, tokens)
    labels = [d["Параметр"] for d in out]

    # matched fields follow manifest order
    assert labels.index("Жалобы на момент осмотра") < labels.index("Анамнез заболевания")
    assert labels.index("Анамнез заболевания") < labels.index("Температура")
    assert labels.index("Температура") < labels.index("ЧСС")
    assert labels.index("ЧСС") < labels.index("Диагноз")
    assert labels.index("Диагноз") < labels.index("Рекомендации и назначения:")
    # unmatched field goes to the tail
    assert labels[-1] == "Направление к другому специалисту (списком)"
    # nothing lost
    assert len(out) == len(data)
