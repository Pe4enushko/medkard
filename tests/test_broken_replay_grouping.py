"""Unit tests for grouping broken rows by organization."""

from audit.broken_replay import group_by_org


def _row(guid: str, org: str | None) -> dict:
    return {
        "card_guid": guid,
        "card_data": {"Прием": {"GUID": guid}},
        "organization_id": org,
    }


def test_group_by_org_splits_rows_and_carries_visits() -> None:
    rows = [_row("a", "org-1"), _row("b", "org-2"), _row("c", "org-1")]
    groups = group_by_org(rows, {"org-1": "Alenka", "org-2": "MDS"})

    by_id = {group.org_id: group for group in groups}
    assert set(by_id) == {"org-1", "org-2"}
    assert by_id["org-1"].guids == {"a", "c"}
    assert by_id["org-1"].visits == [
        {"Прием": {"GUID": "a"}},
        {"Прием": {"GUID": "c"}},
    ]
    assert by_id["org-2"].org_name == "MDS"


def test_group_by_org_keeps_null_and_unknown_orgs_distinct() -> None:
    groups = group_by_org([_row("a", None), _row("b", "org-ghost")], {})

    by_id = {group.org_id: group for group in groups}
    assert by_id[None].guids == {"a"}
    assert by_id["org-ghost"].guids == {"b"}
    assert by_id["org-ghost"].org_name is None


def test_group_by_org_is_deterministic_with_null_last() -> None:
    rows = [_row("a", "org-2"), _row("b", None), _row("c", "org-1")]
    groups = group_by_org(rows, {"org-1": "Alenka", "org-2": "MDS"})

    assert [group.org_name for group in groups] == ["Alenka", "MDS", None]
    assert group_by_org([], {}) == []
