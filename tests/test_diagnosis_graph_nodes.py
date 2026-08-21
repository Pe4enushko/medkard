from __future__ import annotations

import json

import LLM.graphs.diagnosis_nodes as nodes
from LLM.graphs.diagnosis_nodes import (
    _default_medicine_lookup,
    _render_criteria_pool,
    _render_pool,
    build_aspect_pool,
    collect_guideline_sources,
    extract_drugs,
    generate_questions,
    judge_aspect,
    lookup_drugs,
    resolve_judge_output,
    retrieve,
    retrieve_criteria,
)
from LLM.graphs.diagnosis_state import JudgeOutput


def _row(
    chunk_id: str,
    *,
    section: str,
    index: int,
    rrf: float,
    rerank: float | None = None,
    text: str = "text",
    content_type: str = "text",
) -> dict:
    return {
        "id": chunk_id,
        "file_id": "file-1",
        "chunk": text,
        "metadata": {
            "section": section,
            "chunk_index": index,
            "content_type": content_type,
        },
        "rrf_score": rrf,
        "rerank_score": rerank,
    }


def test_build_aspect_pool_deduplicates_caps_and_numbers_in_document_order() -> None:
    pool = build_aspect_pool(
        [
            (
                "q1",
                [
                    _row("a", section="3 Лечение", index=20, rrf=0.2),
                    _row("b", section="2 Диагностика", index=10, rrf=0.4),
                ],
            ),
            (
                "q2",
                [
                    _row("a", section="3 Лечение", index=20, rrf=0.3),
                    _row("c", section="4", index=30, rrf=0.1),
                ],
            ),
        ],
        file_id="file-1",
        doc_title="КР",
        limit=2,
    )

    assert [chunk["id"] for chunk in pool] == ["b", "a"]
    assert [chunk["ref"] for chunk in pool] == [1, 2]
    assert pool[1]["questions"] == ["q1", "q2"]
    assert pool[1]["rrf_score"] == 0.3


def test_render_pool_does_not_expose_chunk_index_as_a_competing_ref() -> None:
    pool = build_aspect_pool(
        [("q", [_row("a", section="2.1 Осмотр", index=25, rrf=0.4)])],
        file_id="file-1",
        doc_title="КР",
    )

    rendered = _render_pool(pool)

    assert "Допустимые значения chunk_refs: 1" in rendered
    assert "chunk_ref=1" in rendered
    assert "фрагмент 25" not in rendered


def test_render_criteria_pool_reconstructs_batches_as_one_table() -> None:
    pool = build_aspect_pool(
        [
            (
                "",
                [
                    _row(
                        "a",
                        section="Критерии оценки качества",
                        index=0,
                        rrf=0.0,
                        text=json.dumps(
                            [{"№": "1", "Критерий": "Собран анамнез"}],
                            ensure_ascii=False,
                        ),
                        content_type="table",
                    ),
                    _row(
                        "b",
                        section="Критерии оценки качества",
                        index=1,
                        rrf=0.0,
                        text=json.dumps(
                            [{"№": "2", "Критерий": "Проведён осмотр"}],
                            ensure_ascii=False,
                        ),
                        content_type="table",
                    ),
                ],
            )
        ],
        file_id="file-1",
        doc_title="КР",
        limit=None,
    )

    rendered = _render_criteria_pool(pool)

    assert rendered.count("| chunk_ref (источник) | № | Критерий |") == 1
    assert "| 1 | 1 | Собран анамнез |" in rendered
    assert "| 2 | 2 | Проведён осмотр |" in rendered
    assert "Номер критерия из таблицы не является chunk_ref" in rendered


def test_resolve_judge_output_discards_unknown_and_duplicate_refs(caplog) -> None:
    pool = build_aspect_pool(
        [("q", [_row("a", section="3", index=20, rrf=0.4, text="abcdef")])],
        file_id="file-1",
        doc_title="КР",
    )
    output = JudgeOutput.model_validate(
        {"issues": [{"issue": "  Замечание  ", "chunk_refs": [1, 1, 999]}]}
    )

    issues = resolve_judge_output(
        output, aspect="treatment", pool=pool, cite_max_chars=4
    )

    assert issues == [
        {
            "aspect": "treatment",
            "issue": "Замечание",
            "sources": [
                {
                    "doc_title": "КР",
                    "section": "3",
                    "cite": "abcd",
                    "chunk_id": "a",
                    "chunk_index": 20,
                }
            ],
        }
    ]
    assert "unknown chunk ref=999" in caplog.text


def test_resolve_judge_output_keeps_issue_without_valid_source() -> None:
    output = JudgeOutput.model_validate(
        {"issues": [{"issue": "Замечание", "chunk_refs": [4]}]}
    )

    issues = resolve_judge_output(output, aspect="anamnesis", pool=[])

    assert issues[0]["issue"] == "Замечание"
    assert issues[0]["sources"] == []


def test_collect_guideline_sources_includes_uncited_sections() -> None:
    pool = build_aspect_pool(
        [
            (
                "q",
                [
                    _row("a", section="2 Диагностика", index=10, rrf=0.4),
                    _row("b", section="3 Лечение", index=20, rrf=0.3),
                ],
            )
        ],
        file_id="file-1",
        doc_title="КР",
    )
    issues = resolve_judge_output(
        JudgeOutput.model_validate(
            {"issues": [{"issue": "Замечание", "chunk_refs": [1]}]}
        ),
        aspect="inspection",
        pool=pool,
    )

    sources = collect_guideline_sources({"inspection": pool}, {"inspection": issues})

    assert sources == [
        {
            "file_id": "file-1",
            "doc_title": "КР",
            "sections": [
                {"section": "2 Диагностика", "chunk_indices": [10], "cited": True},
                {"section": "3 Лечение", "chunk_indices": [20], "cited": False},
            ],
        }
    ]


class _Client:
    def __init__(self, response, tokens: int = 7) -> None:
        self.response = response
        self.tokens = tokens
        self.calls = []

    async def call(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, self.tokens


class _Detector:
    def check_str(self, text: str) -> bool:
        return False


async def test_generate_questions_caps_and_labels_model_output(monkeypatch) -> None:
    monkeypatch.setattr("LLM.graphs.diagnosis_nodes.QUESTIONS_PER_ASPECT_MAX", 2)
    client = _Client(
        '{"anamnesis":["a1","a2","a3"],"inspection":["i1"],"treatment":["t1"]}',
        tokens=11,
    )

    update = await generate_questions(
        {
            "patient_block": "patient",
            "diagnosis_block": "J01",
            "visit_context": "visit",
            "toc": ["3 Лечение"],
            "correlation_id": "correlation-1",
        },
        client=client,
    )

    assert update["questions"] == [
        {"aspect": "anamnesis", "text": "a1"},
        {"aspect": "anamnesis", "text": "a2"},
        {"aspect": "inspection", "text": "i1"},
        {"aspect": "treatment", "text": "t1"},
    ]
    assert update["tokens"] == 11
    assert client.calls[0][1]["metadata"]["correlation_id"] == "correlation-1"


async def test_generate_questions_uses_templates_on_invalid_json() -> None:
    update = await generate_questions(
        {"diagnosis_block": "J01"},
        client=_Client("truncated", tokens=13),
    )

    assert len(update["questions"]) == 6
    assert {question["aspect"] for question in update["questions"]} == {
        "anamnesis",
        "inspection",
        "treatment",
    }
    assert update["tokens"] == 13
    assert update["errors"][0].startswith("generate_questions: fallback templates")


async def test_extract_drugs_degrades_on_schema_error() -> None:
    update = await extract_drugs(
        {"visit_context": "Назначен амоксициллин"},
        client=_Client('{"items":[{"as_written":"Амоксициллин"}]}', tokens=5),
    )

    assert update["drug_mentions"] == []
    assert update["tokens"] == 5
    assert update["errors"][0].startswith("extract_drugs:")


async def test_lookup_drugs_passes_visit_date_and_formats_one_context(
    monkeypatch,
) -> None:
    from datetime import date

    calls = []
    trace_events = []
    monkeypatch.setattr(
        nodes,
        "trace_emit",
        lambda event, **fields: trace_events.append((event, fields)),
    )

    async def lookup(query, *, on):
        calls.append((query, on))
        return f"found {query}"

    update = await lookup_drugs(
        {
            "visit_date": date(2025, 3, 10),
            "drug_mentions": [
                {"as_written": "Амоксиклав", "normalized": "амоксиклав"},
                {"as_written": "Ксизал", "normalized": "ксизал"},
            ],
        },
        lookup=lookup,
    )

    assert calls == [
        ("амоксиклав", date(2025, 3, 10)),
        ("ксизал", date(2025, 3, 10)),
    ]
    assert "Дата визита: 2025-03-10" in update["drug_context"]
    assert "- Амоксиклав → found амоксиклав" in update["drug_context"]
    retrieved = [
        fields for event, fields in trace_events if event == "medicine.retrieved"
    ]
    assert [item["mention"]["normalized"] for item in retrieved] == [
        "амоксиклав",
        "ксизал",
    ]
    assert [item["result"] for item in retrieved] == [
        "found амоксиклав",
        "found ксизал",
    ]


async def test_default_medicine_lookup_uses_grls_with_visit_date(
    monkeypatch,
) -> None:
    from datetime import date

    import grls.format as grls_format
    import grls.lookup as grls_lookup

    raw_result = object()
    calls: list[tuple[str, date | None]] = []
    trace_events = []

    async def lookup(query, *, on=None):
        calls.append((query, on))
        return raw_result

    def format_lookup(result):
        assert result is raw_result
        return "GRLS result"

    monkeypatch.setattr(grls_lookup, "lookup_medicine", lookup)
    monkeypatch.setattr(grls_format, "format_medicine_lookup", format_lookup)
    monkeypatch.setattr(
        nodes,
        "trace_emit",
        lambda event, **fields: trace_events.append((event, fields)),
    )
    visit_date = date(2025, 3, 10)

    assert await _default_medicine_lookup("Амоксиклав", visit_date) == "GRLS result"
    assert calls == [("Амоксиклав", visit_date)]
    registry_event = next(
        fields for event, fields in trace_events if event == "medicine.registry.completed"
    )
    assert registry_event["registry"] == "grls"
    assert registry_event["results"] is raw_result


async def test_lookup_drugs_degrades_when_grls_is_unavailable() -> None:
    async def lookup(query, *, on=None):
        del query, on
        raise RuntimeError("GRLS unavailable")

    update = await lookup_drugs(
        {
            "drug_mentions": [
                {"as_written": "Амоксиклав", "normalized": "амоксиклав"}
            ]
        },
        lookup=lookup,
    )

    assert update["drug_context"] == "справка недоступна"
    assert update["errors"] == ["lookup_drugs: GRLS unavailable"]


async def test_retrieve_skips_one_failed_question_and_builds_all_pools(
    monkeypatch,
) -> None:
    monkeypatch.setattr("LLM.graphs.diagnosis_nodes.CANDIDATES_PER_QUESTION", 12)
    monkeypatch.setattr("LLM.graphs.diagnosis_nodes.TOP_K_PER_QUESTION", 3)

    async def search(question, file_id, *, candidates, top_k):
        assert (file_id, candidates, top_k) == ("file-1", 12, 3)
        if question == "fail":
            raise RuntimeError("embedding unavailable")
        return [_row(question, section="2", index=1, rrf=0.4)]

    update = await retrieve(
        {
            "file_id": "file-1",
            "doc_title": "КР",
            "questions": [
                {"aspect": "anamnesis", "text": "ok"},
                {"aspect": "anamnesis", "text": "fail"},
                {"aspect": "inspection", "text": "inspect"},
                {"aspect": "treatment", "text": "treat"},
            ],
        },
        search=search,
    )

    assert [chunk["id"] for chunk in update["pools"]["anamnesis"]] == ["ok"]
    assert "retrieve_anamnesis: embedding unavailable" in update["errors"]
    assert not any(
        error.startswith("retrieve_inspection") for error in update["errors"]
    )


async def test_retrieve_criteria_marks_missing_section_as_degradation() -> None:
    async def get_chunks(file_id, pattern):
        return []

    update = await retrieve_criteria(
        {"file_id": "file-1", "doc_title": "КР"},
        get_chunks=get_chunks,
    )

    assert update == {
        "pools": {"criteria": []},
        "errors": ["retrieve_criteria: section not found"],
    }


async def test_retrieve_criteria_keeps_the_complete_section() -> None:
    rows = [
        _row(
            f"criteria-{index}",
            section="Критерии оценки качества",
            index=index,
            rrf=0.0,
            text=f"row {index}",
            content_type="table",
        )
        for index in range(12)
    ]

    async def get_chunks(file_id, pattern):
        assert (file_id, pattern) == ("file-1", "%критерии оценки качества%")
        return rows

    update = await retrieve_criteria(
        {"file_id": "file-1", "doc_title": "КР"},
        get_chunks=get_chunks,
    )

    assert len(update["pools"]["criteria"]) == 12
    assert [chunk["ref"] for chunk in update["pools"]["criteria"]] == list(range(1, 13))


async def test_judge_criteria_receives_the_reconstructed_table() -> None:
    pool = build_aspect_pool(
        [
            (
                "",
                [
                    _row(
                        "criteria-1",
                        section="Критерии оценки качества",
                        index=7,
                        rrf=0.0,
                        text=json.dumps(
                            [{"№": "20", "Критерий": "Проведён осмотр"}],
                            ensure_ascii=False,
                        ),
                        content_type="table",
                    )
                ],
            )
        ],
        file_id="file-1",
        doc_title="КР",
        limit=None,
    )
    client = _Client('{"issues":[]}')

    update = await judge_aspect(
        {"doc_title": "КР", "pools": {"criteria": pool}},
        "criteria",
        client=client,
        detector=_Detector(),
    )

    user_context = client.calls[0][0][1]["content"]
    assert update["issues"] == {"criteria": []}
    assert "| chunk_ref (источник) | № | Критерий |" in user_context
    assert "| 1 | 20 | Проведён осмотр |" in user_context
    assert "фрагмент 7" not in user_context


async def test_judge_treatment_receives_grls_context() -> None:
    pool = build_aspect_pool(
        [("therapy", [_row("treatment-1", section="Лечение", index=8, rrf=0.5)])],
        file_id="file-1",
        doc_title="КР",
    )
    client = _Client('{"issues":[]}')

    update = await judge_aspect(
        {
            "doc_title": "КР",
            "drug_context": (
                "Дата визита: 2025-03-10\n"
                "- Амоксиклав → Найдено в ГРЛС; МНН: амоксициллин"
            ),
            "pools": {"treatment": pool},
        },
        "treatment",
        client=client,
        detector=_Detector(),
    )

    user_context = client.calls[0][0][1]["content"]
    assert update["issues"] == {"treatment": []}
    assert "## Справка по препаратам" in user_context
    assert "Найдено в ГРЛС" in user_context
    assert "## Фрагменты клинических рекомендаций «КР»" in user_context


async def test_judge_aspect_degrades_on_invalid_json_without_losing_tokens() -> None:
    pool = build_aspect_pool(
        [("q", [_row("a", section="2", index=1, rrf=0.4)])],
        file_id="file-1",
        doc_title="КР",
    )

    update = await judge_aspect(
        {
            "doc_title": "КР",
            "pools": {"inspection": pool},
        },
        "inspection",
        client=_Client("{", tokens=17),
        detector=_Detector(),
    )

    assert update["issues"] == {"inspection": []}
    assert update["tokens"] == 17
    assert update["errors"][0].startswith("judge_inspection:")
