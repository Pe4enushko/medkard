from storage.models.guideline import name_embed_input


def test_name_with_ages():
    assert name_embed_input("Бронхит", ["Взрослые", "Дети"]) == \
        "Название: Бронхит\nВозрастная группа: [Взрослые, Дети]"


def test_name_without_ages():
    assert name_embed_input("Бронхит", []) == "Название: Бронхит"


def test_none_ages_treated_as_empty():
    assert name_embed_input("Бронхит", None) == "Название: Бронхит"


def test_strips_name_and_ages():
    assert name_embed_input("  Бронхит  ", ["  Дети  "]) == \
        "Название: Бронхит\nВозрастная группа: [Дети]"


def test_drops_blank_age_entries():
    assert name_embed_input("Бронхит", ["Дети", "", "  "]) == \
        "Название: Бронхит\nВозрастная группа: [Дети]"


def test_none_name_becomes_empty_base():
    # name is nullable in the DB; must not crash.
    assert name_embed_input(None, []) == "Название: "
