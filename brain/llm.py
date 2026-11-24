"""Multi-Provider LLM Client with Token Streaming & Tool Calling.

Supports:
1. `ollama`        : Fully local on AMD RX 9070 16GB (`/api/chat`, `think: False` for fast voice).
2. `openai_compat` : Local LM Studio / llama.cpp / vLLM (`/v1/chat/completions`).
3. `gemini`        : Optional / fallback cloud (`google-genai`).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from config import config

log = logging.getLogger("llm")

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure at most one system message (at index 0) and no adjacent duplicate roles."""
    out: list[dict[str, Any]] = []
    pending_sys: list[str] = []

    def _emit(role: str, content: str, extra: dict[str, Any] | None = None) -> None:
        if out and out[-1]["role"] == role:
            prev = out[-1]["content"]
            out[-1]["content"] = f"{prev}\n\n{content}" if (prev and content) else (prev or content)
            if extra:
                for k, v in extra.items():
                    if k not in ("role", "content"):
                        out[-1][k] = v
        else:
            msg: dict[str, Any] = dict(extra) if extra else {}
            msg["role"] = role
            msg["content"] = content
            out.append(msg)

    for i, raw in enumerate(messages):
        role = raw["role"]
        content = raw.get("content") or ""
        if role == "system":
            if i == 0:
                out.append({**raw, "role": "system", "content": content})
            elif content:
                pending_sys.append(content)
            continue
        if role == "user" and pending_sys:
            prefix = "\n\n".join(pending_sys)
            pending_sys.clear()
            content = f"{prefix}\n\n{content}" if content else prefix
        _emit(role, content, raw)

    if pending_sys:
        _emit("user", "\n\n".join(pending_sys))

    return out


@dataclass
class LLMToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class LLMClient:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=8.0))

    async def close(self) -> None:
        await self._http.aclose()

    def active_provider(self) -> str:
        return config.llm_provider

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        fast_voice: bool = False,
    ) -> LLMResponse:
        provider = config.llm_provider
        try:
            if provider == "ollama":
                return await self._ollama_once(messages, tools=tools, fast_voice=fast_voice)
            if provider == "openai_compat":
                return await self._openai_once(messages, tools=tools, fast_voice=fast_voice)
            if provider == "gemini":
                return await self._gemini_once(messages)
        except Exception as e:
            log.warning("Primary provider '%s' failed (%s), checking fallback...", provider, e)
            if provider != "gemini" and config.gemini_api_key:
                log.info("Falling back to Gemini API...")
                return await self._gemini_once(messages)
            raise
        return await self._ollama_once(messages, tools=tools, fast_voice=fast_voice)

    async def stream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        fast_voice: bool = True,
    ) -> AsyncIterator[str]:
        """Stream tokens one by one for real-time voice sentence responses."""
        provider = config.llm_provider
        try:
            if provider == "ollama":
                async for tok in self._ollama_stream(messages, fast_voice=fast_voice):
                    yield tok
                return
            if provider == "openai_compat":
                async for tok in self._openai_stream(messages, fast_voice=fast_voice):
                    yield tok
                return
        except Exception as e:
            log.warning("Streaming '%s' failed (%s), trying fallback...", provider, e)
            if config.gemini_api_key:
                resp = await self._gemini_once(messages)
                if resp.text:
                    yield resp.text
                return
            raise

        resp = await self._gemini_once(messages)
        if resp.text:
            yield resp.text

    # ---------------- Ollama Native (/api/chat) ----------------
    async def _ollama_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        fast_voice: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": config.ollama_model,
            "messages": normalize_messages(messages),
            "stream": False,
            "keep_alive": config.ollama_keep_alive,
            "think": False,  # Disable long chain-of-thought for instant replies
            "options": {
                "temperature": 0.82,
                "num_predict": 320 if fast_voice else 650,
            },
        }
        if tools:
            payload["tools"] = tools

        r = await self._http.post(f"{config.ollama_base_url.rstrip('/')}/api/chat", json=payload)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think", None)
            r = await self._http.post(f"{config.ollama_base_url.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        content = _THINK_TAG_RE.sub("", msg.get("content", "") or "").strip()

        tcalls: list[LLMToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            if fn.get("name"):
                tcalls.append(LLMToolCall(name=fn["name"], arguments=args))
        return LLMResponse(text=content, tool_calls=tcalls)

    async def _ollama_stream(self, messages: list[dict[str, Any]], fast_voice: bool) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": config.ollama_model,
            "messages": normalize_messages(messages),
            "stream": True,
            "keep_alive": config.ollama_keep_alive,
            "think": False,
            "options": {
                "temperature": 0.82,
                "num_predict": 280 if fast_voice else 600,
            },
        }
        in_think = False
        url = f"{config.ollama_base_url.rstrip('/')}/api/chat"
        async with self._http.stream("POST", url, json=payload) as resp:
            if resp.status_code == 400:
                payload.pop("think", None)
                async with self._http.stream("POST", url, json=payload) as resp2:
                    resp2.raise_for_status()
                    async for line in resp2.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line).get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                return
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                tok = obj.get("message", {}).get("content", "")
                if not tok:
                    continue
                if "<think>" in tok:
                    in_think = True
                    tok = tok.split("<think>", 1)[0]
                if "</think>" in tok:
                    in_think = False
                    tok = tok.split("</think>", 1)[-1]
                if not in_think and tok:
                    yield tok

    # ---------------- OpenAI-Compatible (LM Studio / llama.cpp) ----------------
    async def _openai_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        fast_voice: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": config.openai_model,
            "messages": normalize_messages(messages),
            "temperature": 0.82,
            "max_tokens": 320 if fast_voice else 650,
        }
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {config.openai_api_key}"}
        r = await self._http.post(f"{config.openai_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        choice = r.json()["choices"][0]["message"]
        content = _THINK_TAG_RE.sub("", choice.get("content") or "").strip()
        tcalls: list[LLMToolCall] = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tcalls.append(LLMToolCall(name=fn["name"], arguments=args))
        return LLMResponse(text=content, tool_calls=tcalls)

    async def _openai_stream(self, messages: list[dict[str, Any]], fast_voice: bool) -> AsyncIterator[str]:
        payload = {
            "model": config.openai_model,
            "messages": normalize_messages(messages),
            "temperature": 0.82,
            "max_tokens": 280 if fast_voice else 600,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {config.openai_api_key}"}
        url = f"{config.openai_base_url.rstrip('/')}/chat/completions"
        async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                delta = json.loads(raw)["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield delta

    # ---------------- Gemini Cloud Fallback ----------------
    async def _gemini_once(self, messages: list[dict[str, Any]]) -> LLMResponse:
        from google import genai
        from google.genai import types

        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty and local Ollama is not responding.")

        client = genai.Client(api_key=config.gemini_api_key)
        sys_chunks: list[str] = []
        contents: list[types.Content] = []
        for m in normalize_messages(messages):
            if m["role"] == "system":
                sys_chunks.append(m["content"])
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        for model in (config.gemini_model, *config.gemini_fallback_models):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction="\n\n".join(sys_chunks),
                        temperature=0.85,
                        max_output_tokens=600,
                    ),
                )
                return LLMResponse(text=(resp.text or "").strip())
            except Exception as e:
                log.warning("Gemini model '%s' failed: %s", model, e)
        raise RuntimeError("All Gemini models failed.")
