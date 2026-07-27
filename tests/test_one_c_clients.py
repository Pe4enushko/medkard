"""Request-building tests for the 1C clients — no network, urlopen is captured."""

import base64
import io
import json

from integrations.one_c import AlenkaOneCClient, MdsOneCClient


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _capture_urlopen(monkeypatch, payloads):
    """Patch urlopen to return canned JSON payloads and record each Request."""
    captured = []
    responses = [json.dumps(p).encode("utf-8") for p in payloads]

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return _FakeResponse(responses[len(captured) - 1])

    monkeypatch.setattr("integrations.one_c.urllib.request.urlopen", fake_urlopen)
    return captured


def _basic_token(login, password):
    return "Basic " + base64.b64encode(f"{login}:{password}".encode()).decode("ascii")


def test_alenka_sends_get_with_query_params(monkeypatch):
    captured = _capture_urlopen(monkeypatch, [[{"Прием": {"GUID": "a"}}]])
    client = AlenkaOneCClient("http://one-c/api", "user", "pass")

    result = client.fetch_json_for_period("01.06.2026", "03.06.2026")

    assert len(captured) == 1
    request = captured[0]
    assert request.get_method() == "GET"
    assert "datebegin=01.06.2026" in request.full_url
    assert "dateend=03.06.2026" in request.full_url
    assert request.get_header("Authorization") == _basic_token("user", "pass")
    assert result == [{"Прием": {"GUID": "a"}}]


def test_mds_sends_post_with_iso_date_in_body(monkeypatch):
    captured = _capture_urlopen(monkeypatch, [[{"Прием": {"GUID": "a"}}]])
    client = MdsOneCClient("http://one-c/api", "user", "pass")

    result = client.fetch_json_for_period("03.06.2026", "03.06.2026")

    assert len(captured) == 1
    request = captured[0]
    assert request.get_method() == "POST"
    assert request.full_url == "http://one-c/api"
    assert json.loads(request.data.decode("utf-8")) == {"date": "2026-06-03"}
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == _basic_token("user", "pass")
    assert result == [{"Прием": {"GUID": "a"}}]


def test_mds_multi_day_period_posts_once_per_day_and_merges(monkeypatch):
    captured = _capture_urlopen(
        monkeypatch,
        [
            [{"Прием": {"GUID": "a"}}],
            [],
            [{"Прием": {"GUID": "b"}}, {"Прием": {"GUID": "c"}}],
        ],
    )
    client = MdsOneCClient("http://one-c/api", "user", "pass")

    result = client.fetch_json_for_period("01.06.2026", "03.06.2026")

    dates = [json.loads(r.data.decode("utf-8"))["date"] for r in captured]
    assert dates == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert result == [
        {"Прием": {"GUID": "a"}},
        {"Прием": {"GUID": "b"}},
        {"Прием": {"GUID": "c"}},
    ]


def test_mds_single_visit_dict_payload_is_wrapped_in_list(monkeypatch):
    _capture_urlopen(monkeypatch, [{"Прием": {"GUID": "solo"}}, []])
    client = MdsOneCClient("http://one-c/api", "user", "pass")

    result = client.fetch_json_for_period("01.06.2026", "02.06.2026")

    assert result == [{"Прием": {"GUID": "solo"}}]


def test_mds_allows_empty_password(monkeypatch):
    captured = _capture_urlopen(monkeypatch, [[]])
    client = MdsOneCClient("http://one-c/api", "user", "")

    result = client.fetch_json_for_period("03.06.2026", "03.06.2026")

    assert captured[0].get_header("Authorization") == _basic_token("user", "")
    assert result == []
