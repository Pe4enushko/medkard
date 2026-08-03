"""
client.py — unified LLM client with retry logic.

Provides LLMClient with two methods:
- call(): raw chat completion, optional Pydantic response_model for json_schema
- call_agent(): LangChain ReAct agent invocation

Both methods retry up to max_retries times, bumping temperature and injecting
a Russian-language failure notice into the message history on each retry.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import os

from langgraph.errors import GraphRecursionError
from openai import APIError, BadRequestError
from pydantic import BaseModel

from LLM.base import MODEL, get_openai_client
from LLM.observability import emit
from LLM.vllm_config import build_vllm_extra_body

logger = logging.getLogger(__name__)

_TEMP_CAP = 1.0


def _is_context_overflow(error_text: str) -> bool:
    lowered = error_text.lower()
    return any(
        marker in lowered
        for marker in ("context length", "input_tokens", "maximum context", "too many tokens")
    )


class LLMClient:
    def __init__(
        self,
        model: str = MODEL,
        max_retries: int = 2,
        temp_bump: float = 0.15,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._temp_bump = temp_bump

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        response_model: type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Chat completion with optional json_schema structured output and retry.

        Args:
            messages:       OpenAI messages list (mutated in-place on retry).
            temperature:    Initial sampling temperature.
            response_model: Pydantic model whose JSON schema constrains decoding
                            via response_format (OpenAI-standard json_schema —
                            vLLM enforces it). Pass None for free-form text.

        Returns:
            (content_str, total_tokens)
        """
        # Use the OpenAI-standard response_format=json_schema, NOT the legacy
        # extra_body={"guided_json": ...}: this vLLM build silently ignores the
        # latter ("fields ignored: {'guided_json'}"), so the schema was never
        # enforced and the model free-formed prose / hit the length cap.
        response_format: dict[str, Any] | None = None
        if response_model is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }

        # gpt-oss reasons at 'medium' by default; for mechanical tasks the
        # reasoning tokens eat the context budget and truncate the answer.
        # 'low' is the floor (reasoning can't be fully disabled on gpt-oss).
        extra_body: dict[str, Any] = build_vllm_extra_body()
        if reasoning_effort is not None:
            extra_body["reasoning_effort"] = reasoning_effort

        total_tokens = 0
        trace_id = uuid.uuid4().hex
        metadata = metadata or {}
        emit("llm_start", trace_id=trace_id, model=self._model, prompt_chars=sum(len(str(m.get("content") or "")) for m in messages), **metadata)
        temp = temperature
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                emit("llm_attempt_start", trace_id=trace_id, attempt=attempt + 1, temperature=temp, **metadata)
                kwargs: dict[str, Any] = dict(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if response_model is not None:
                    kwargs["max_tokens"] = int(os.environ.get("LLM_RAW_MAX_OUTPUT_TOKENS", "4096"))

                resp = await get_openai_client().chat.completions.create(**kwargs)
                total_tokens += resp.usage.total_tokens if resp.usage else 0
                content = resp.choices[0].message.content or ""
                finish_reason = resp.choices[0].finish_reason
                emit(
                    "llm_attempt_end",
                    trace_id=trace_id,
                    attempt=attempt + 1,
                    finish_reason=finish_reason,
                    total_tokens=resp.usage.total_tokens if resp.usage else 0,
                    completion_chars=len(content),
                    **metadata,
                )

                if finish_reason == "stop":
                    emit("llm_success", trace_id=trace_id, finish_reason=finish_reason, total_tokens=resp.usage.total_tokens if resp.usage else 0, completion_chars=len(content), **metadata)
                    return content, total_tokens

                notice = f"Предыдущая попытка завершилась с причиной '{finish_reason}'. Повтори ответ."
                logger.warning(
                    "[llm_client] attempt %d/%d finish_reason=%r — retrying",
                    attempt + 1, self._max_retries + 1, finish_reason,
                )

            except APIError as exc:
                total_tokens += 0
                notice = f"Предыдущая попытка завершилась ошибкой: {str(exc)[:120]}. Повтори ответ."
                last_exc = exc
                emit("llm_error", trace_id=trace_id, exception_type=type(exc).__name__, exception=str(exc)[:300], **metadata)
                if isinstance(exc, BadRequestError) and _is_context_overflow(str(exc)):
                    break
                logger.warning(
                    "[llm_client] attempt %d/%d APIError=%s — retrying",
                    attempt + 1, self._max_retries + 1, str(exc)[:80],
                )

            if attempt < self._max_retries:
                messages = list(messages) + [{"role": "user", "content": notice}]
                temp = min(temp + self._temp_bump, _TEMP_CAP)
                emit("llm_retry", trace_id=trace_id, attempt=attempt + 1, retry_mode="same_contract", **metadata)

        if last_exc is not None:
            raise last_exc
        # Exhausted retries with a non-'stop' finish (usually 'length' → truncated/empty).
        # Surface it: the caller's JSON parse is about to fail and this is the real cause.
        logger.error(
            "[llm_client] exhausted %d attempt(s), finish_reason=%r — returning %d-char content: %r",
            self._max_retries + 1, finish_reason, len(content), content[:200],
        )
        return content, total_tokens  # last attempt's content even on bad finish_reason

    async def call_agent(
        self,
        system_prompt: str,
        tools: list,
        human_message: str,
        response_format: type[BaseModel] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str | BaseModel, int]:
        """Invoke a LangGraph ReAct agent with retry.

        Creates a fresh agent on each attempt and injects a failure notice as
        an extra user message when retrying.

        Args:
            system_prompt:   Fully rendered system prompt for the agent.
            tools:           File-id-bound tool instances.
            human_message:   The user-facing clinical input.
            response_format: Optional Pydantic model — passed to create_react_agent
                             so the final answer is structured via with_structured_output.
                             When provided the return value is a Pydantic instance, not str.

        Returns:
            (last_message_content_or_pydantic_instance, total_tokens)
        """
        total_tokens = 0
        last_exc: Exception | None = None
        current_human = human_message
        trace_id = uuid.uuid4().hex
        metadata = metadata or {}
        compact_retry = False
        base_steps = int(os.environ.get("AGENT_MAX_STEPS", "12"))
        compact_steps = int(os.environ.get("AGENT_COMPACT_MAX_STEPS", "8"))
        normal_tool_calls = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "12"))
        compact_tool_calls = int(os.environ.get("AGENT_COMPACT_MAX_TOOL_CALLS", "4"))
        result_chars = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "12000"))

        from LLM.rag_agent import ToolCallGuard, _sum_agent_tokens, create_checker_agent  # lazy import

        emit("agent_start", trace_id=trace_id, model=self._model, prompt_chars=len(human_message), **metadata)

        for attempt in range(self._max_retries + 1):
            guard = ToolCallGuard(
                max_calls=compact_tool_calls if compact_retry else normal_tool_calls,
                max_result_chars=result_chars,
            )
            try:
                agent = create_checker_agent(
                    system_prompt,
                    tools,
                    response_format=response_format,
                    tool_guard=guard,
                )
                max_steps = compact_steps if compact_retry else base_steps
                emit(
                    "agent_attempt_start",
                    trace_id=trace_id,
                    attempt=attempt + 1,
                    mode="compact" if compact_retry else "normal",
                    recursion_limit=max_steps,
                    **metadata,
                )
                result = await agent.ainvoke(
                    {"messages": [("user", current_human)]},
                    config={"recursion_limit": max_steps},
                )
                total_tokens += _sum_agent_tokens(result)
                for event in guard.events:
                    emit("agent_tool", trace_id=trace_id, attempt=attempt + 1, **event, **metadata)

                last_msg = result["messages"][-1]
                if response_format is not None:
                    finish_reason = (getattr(last_msg, "response_metadata", {}) or {}).get("finish_reason")
                    if finish_reason and finish_reason != "stop":
                        raise ValueError(f"structured output finished with {finish_reason!r}")
                    structured = result.get("structured_response")
                    if structured is not None:
                        emit("agent_success", trace_id=trace_id, attempts=attempt + 1, total_tokens=total_tokens, **metadata)
                        return structured, total_tokens

                    parsed = getattr(last_msg, "parsed", None)
                    if parsed is not None:
                        emit("agent_success", trace_id=trace_id, attempts=attempt + 1, total_tokens=total_tokens, **metadata)
                        return parsed, total_tokens

                    content = getattr(last_msg, "content", "")
                    if isinstance(content, response_format):
                        emit("agent_success", trace_id=trace_id, attempts=attempt + 1, total_tokens=total_tokens, **metadata)
                        return content, total_tokens
                    if not content:
                        raise ValueError("structured response has neither parsed object nor content")
                    return content, total_tokens

                content: str = last_msg.content or ""
                finish_reason = (getattr(last_msg, "response_metadata", {}) or {}).get("finish_reason")
                if not finish_reason or finish_reason == "stop":
                    emit("agent_success", trace_id=trace_id, attempts=attempt + 1, total_tokens=total_tokens, **metadata)
                    return content, total_tokens

                notice = f"Предыдущая попытка завершилась с причиной '{finish_reason}'. Верни итоговый ответ."
                logger.warning(
                    "[llm_client:agent] attempt %d/%d finish_reason=%r — retrying",
                    attempt + 1, self._max_retries + 1, finish_reason,
                )

            except GraphRecursionError as exc:
                last_exc = exc
                notice = "Предыдущая попытка достигла лимита шагов. Не повторяй одинаковые вызовы инструментов и верни итоговый JSON."
                emit("agent_error", trace_id=trace_id, attempt=attempt + 1, exception_type=type(exc).__name__, exception=str(exc)[:300], **metadata)
                for event in guard.events:
                    emit("agent_tool", trace_id=trace_id, attempt=attempt + 1, **event, **metadata)
                if not compact_retry and attempt < self._max_retries:
                    compact_retry = True
                    current_human = f"{human_message}\n\n{notice}"
                    emit("agent_retry", trace_id=trace_id, attempt=attempt + 1, retry_mode="compact", **metadata)
                    continue
                break
            except BadRequestError as exc:
                last_exc = exc
                error_text = str(exc)
                emit("agent_error", trace_id=trace_id, attempt=attempt + 1, exception_type=type(exc).__name__, exception=error_text[:300], **metadata)
                for event in guard.events:
                    emit("agent_tool", trace_id=trace_id, attempt=attempt + 1, **event, **metadata)
                if _is_context_overflow(error_text):
                    break
                notice = f"Предыдущая попытка завершилась ошибкой: {error_text[:120]}. Верни корректный JSON."
            except Exception as exc:
                last_exc = exc
                notice = f"Предыдущая попытка завершилась ошибкой: {str(exc)[:120]}. Верни корректный JSON."
                emit("agent_error", trace_id=trace_id, attempt=attempt + 1, exception_type=type(exc).__name__, exception=str(exc)[:300], **metadata)
                for event in guard.events:
                    emit("agent_tool", trace_id=trace_id, attempt=attempt + 1, **event, **metadata)
                logger.warning(
                    "[llm_client:agent] attempt %d/%d error=%s — retrying",
                    attempt + 1, self._max_retries + 1, str(exc)[:80],
                )

            if attempt < self._max_retries:
                current_human = f"{human_message}\n\n{notice}"
                emit("agent_retry", trace_id=trace_id, attempt=attempt + 1, retry_mode="same_contract", **metadata)

        if last_exc is not None:
            raise last_exc
        return content, total_tokens
