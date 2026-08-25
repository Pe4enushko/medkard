import random

from parsers.inspection_order import load_inspection_format, reorder_inspection_data


def test_alenka_manifest_orders_a_realistic_fragment():
    tokens = load_inspection_format("Alenka", "standard")
    # a realistic, out-of-order Alenka ДанныеОсмотра fragment (labels as they
    # appear in real exports, incl. the trailing-colon case)
    data = [
        {"Параметр": "Заметки", "Значение": "..."},
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
    # a label the manifest knows nothing about goes to the tail
    assert labels[-1] == "Заметки"
    # nothing lost
    assert len(out) == len(data)


def test_alenka_manifest_matches_chief_doctor_template():
    """The full order the Alenka chief doctor asked for, on a real pediatric card.

    Labels are the ones 1C actually emits (see the 20-08-2026 export): the ЛВН
    block comes as «родственник лвн» / «В выдаче листке нетрудоспособности»,
    and a card may carry both «Рекомендации и назначения:» and a bare
    «Рекомендации».
    """
    tokens = load_inspection_format("Alenka", "standard")
    expected = [
        "родственник лвн",
        "Пациент нуждается в уходе",
        "В выдаче листке нетрудоспособности",
        "Жалобы на момент осмотра",
        "Анамнез заболевания",
        "Эпидемиологический анамнез",
        "Прививочный анамнез",
        "Аллергологический анамнез",
        "Температура",
        "ЧСС",
        "ЧД",
        "Вес",
        "Рост",
        "Состояние",
        "Сознание",
        "Ф20",
        "Кожные покровы",
        "Видимые слизистые",
        "Слизистые ротоглотки",
        "Миндалины",
        "Периферические лимфоузлы",
        "Неврологический статус",
        "Опорно-двигательная система",
        "Сердечно-сосудистая система",
        "Дыхательная система",
        "Органы брюшной полости",
        "Стул",
        "Мочеиспускание",
        "План обследования",
        "План лечения",
        "Обоснование диагноза",
        "Рекомендации и назначения:",
        "Рекомендации",
        "Направление к другому специалисту (списком)",
        "Рекомендованна следующая плановая консультация",
    ]
    # shuffle deterministically so the assertion proves the manifest ordered it,
    # not the input order
    data = [{"Параметр": p, "Значение": "x"} for p in expected]
    rnd = random.Random(7)
    rnd.shuffle(data)

    out = reorder_inspection_data(data, tokens)
    assert [d["Параметр"] for d in out] == expected
