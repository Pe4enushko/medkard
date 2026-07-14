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
from typing import Any

import os

from openai import APIError
from pydantic import BaseModel

from LLM.base import MODEL, get_openai_client

logger = logging.getLogger(__name__)

_TEMP_CAP = 1.0


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
        extra_body: dict[str, Any] | None = None
        if reasoning_effort is not None:
            extra_body = {"reasoning_effort": reasoning_effort}

        total_tokens = 0
        temp = temperature
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body

                resp = await get_openai_client().chat.completions.create(**kwargs)
                total_tokens += resp.usage.total_tokens if resp.usage else 0
                content = resp.choices[0].message.content or ""
                finish_reason = resp.choices[0].finish_reason

                if finish_reason == "stop":
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
                logger.warning(
                    "[llm_client] attempt %d/%d APIError=%s — retrying",
                    attempt + 1, self._max_retries + 1, str(exc)[:80],
                )

            if attempt < self._max_retries:
                messages = list(messages) + [{"role": "user", "content": notice}]
                temp = min(temp + self._temp_bump, _TEMP_CAP)

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

        from LLM.rag_agent import _sum_agent_tokens, create_checker_agent  # lazy — avoids langchain at import time

        for attempt in range(self._max_retries + 1):
            try:
                agent = create_checker_agent(system_prompt, tools, response_format=response_format)
                max_steps = int(os.environ.get("AGENT_MAX_STEPS", "25"))
                result = await agent.ainvoke(
                    {"messages": [("user", current_human)]},
                    config={"recursion_limit": max_steps},
                )
                total_tokens += _sum_agent_tokens(result)

                last_msg = result["messages"][-1]
                if response_format is not None:
                    structured = result.get("structured_response")
                    if structured is not None:
                        return structured, total_tokens

                    parsed = getattr(last_msg, "parsed", None)
                    if parsed is not None:
                        return parsed, total_tokens

                    content = getattr(last_msg, "content", "")
                    if isinstance(content, response_format):
                        return content, total_tokens
                    return content, total_tokens

                content: str = last_msg.content or ""
                finish_reason = (getattr(last_msg, "response_metadata", {}) or {}).get("finish_reason")

                if not finish_reason or finish_reason == "stop":
                    return content, total_tokens

                notice = f"Предыдущая попытка завершилась с причиной '{finish_reason}'. Повтори ответ."
                logger.warning(
                    "[llm_client:agent] attempt %d/%d finish_reason=%r — retrying",
                    attempt + 1, self._max_retries + 1, finish_reason,
                )

            except Exception as exc:
                notice = f"Предыдущая попытка завершилась ошибкой: {str(exc)[:120]}. Повтори ответ."
                last_exc = exc
                logger.warning(
                    "[llm_client:agent] attempt %d/%d error=%s — retrying",
                    attempt + 1, self._max_retries + 1, str(exc)[:80],
                )

            if attempt < self._max_retries:
                current_human = f"{human_message}\n\n{notice}"

        if last_exc is not None:
            raise last_exc
        return content, total_tokens
