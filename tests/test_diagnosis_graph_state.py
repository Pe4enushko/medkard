from __future__ import annotations

from typing import get_type_hints

from LLM.graphs.diagnosis_state import (
    DiagnosisState,
    DrugList,
    JudgeOutput,
    QuestionSet,
    merge_dicts,
)


def test_structured_output_models_default_to_empty_collections() -> None:
    assert DrugList().model_dump() == {"items": []}
    assert JudgeOutput().model_dump() == {"issues": []}


def test_question_set_requires_every_aspect() -> None:
    schema = QuestionSet.model_json_schema()

    assert set(schema["required"]) == {"anamnesis", "inspection", "treatment"}


def test_judge_output_validates_issue_references() -> None:
    output = JudgeOutput.model_validate(
        {
            "issues": [
                {"issue": "Не выполнен обязательный осмотр.", "chunk_refs": [2, 5]}
            ]
        }
    )

    assert output.issues[0].chunk_refs == [2, 5]


def test_merge_dicts_returns_new_mapping_with_right_hand_updates() -> None:
    left = {"anamnesis": ["a"]}
    right = {"treatment": ["b"]}

    merged = merge_dicts(left, right)

    assert merged == {"anamnesis": ["a"], "treatment": ["b"]}
    assert merged is not left
    assert merged is not right


def test_parallel_state_fields_keep_annotated_reducers() -> None:
    hints = get_type_hints(DiagnosisState, include_extras=True)

    assert hints["pools"].__metadata__ == (merge_dicts,)
    assert hints["issues"].__metadata__ == (merge_dicts,)
    assert callable(hints["errors"].__metadata__[0])
    assert callable(hints["tokens"].__metadata__[0])
