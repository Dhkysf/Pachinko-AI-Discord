"""Multi-Provider LLM Client with Token Streaming & Tool Calling.

Supports:
1. `ollama`        : Fully local on AMD RX 9070 16GB (`/api/chat`, `think: False` for fast voice).
2. `ninerouter`    : Local 9router OpenAI-compatible gateway (`/v1/chat/completions`).
3. `openai_compat` : Local LM Studio / llama.cpp / vLLM (`/v1/chat/completions`).
4. `gemini`        : Optional / fallback cloud (`google-genai`).
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from brain.tools import TOOL_SCHEMAS
from config import config
from core.safety import redact_secrets

log = logging.getLogger("llm")


def _temperature_kw() -> dict[str, float]:
    return {"temperature": config.llm_temperature} if config.llm_temperature is not None else {}


_TOOLS_REJECTED_RE = re.compile(r"(?:tools?|tool_choice|function)", re.IGNORECASE)


@dataclass(frozen=True)
class _OpenAIEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str


def _to_gemini_tools(types_mod: Any, tools: list[dict[str, Any]] | None) -> list[Any]:
    if not tools:
        return []
    fd_cls = getattr(types_mod, "FunctionDeclaration", None)
    tool_cls = getattr(types_mod, "Tool", None)
    if fd_cls is None or tool_cls is None:
        return []
    decls = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        decls.append(
            fd_cls(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters_json_schema=fn.get("parameters"),
            )
        )
    return [tool_cls(function_declarations=decls)] if decls else []

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    return _UNCLOSED_THINK_RE.sub("", _THINK_TAG_RE.sub("", raw)).strip()
_TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)
_TOOL_TAG_BLOCK_RE = re.compile(
    r"(?:```[a-zA-Z]*\s*)?<tool_call>\s*(.*?)\s*</tool_call>(?:\s*```)?",
    re.DOTALL | re.IGNORECASE,
)
_STRAY_TOOL_TAG_RE = re.compile(r"</?tool_call>", re.IGNORECASE)
_FENCE_WRAP_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_OPEN_FENCE_TAIL_RE = re.compile(r"```[a-zA-Z]*\s*$")
_CLOSE_FENCE_HEAD_RE = re.compile(r"^\s*```")
_CLOSE_TAG_HEAD_RE = re.compile(r"^\s*</tool_call>", re.IGNORECASE)
_JSON_DECODER = json.JSONDecoder()


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
    finish_reason: str | None = None
    has_reasoning: bool = False
    provider: str = ""
    model: str = ""


def extract_tool_calls_from_text(
    text: str, allowed_names: set[str] | frozenset[str] | None = None
) -> tuple[str, list[LLMToolCall]]:
    """Parse and strip leaked tool-call JSON/tags from model content."""
    if not text:
        return "", []
    allowed = _TOOL_NAMES if allowed_names is None else allowed_names
    calls: list[LLMToolCall] = []

    def _parse_tool_obj(obj: Any) -> LLMToolCall | None:
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        if not isinstance(name, str) or name not in allowed:
            return None
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if not isinstance(args, dict):
            return None
        return LLMToolCall(name=name, arguments=args)

    def _replace_tagged(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        fm = _FENCE_WRAP_RE.match(inner)
        if fm:
            inner = fm.group(1).strip()
        pos = 0
        while pos < len(inner):
            idx = inner.find("{", pos)
            if idx == -1:
                break
            try:
                obj, end = _JSON_DECODER.raw_decode(inner, idx)
            except json.JSONDecodeError:
                pos = idx + 1
                continue
            tc = _parse_tool_obj(obj)
            if tc is not None:
                calls.append(tc)
            pos = end
        return ""

    s = _TOOL_TAG_BLOCK_RE.sub(_replace_tagged, text)

    out: list[str] = []
    last = 0
    i = 0
    n = len(s)
    while i < n:
        pos = s.find("{", i)
        if pos == -1:
            break
        try:
            obj, end = _JSON_DECODER.raw_decode(s, pos)
        except json.JSONDecodeError:
            i = pos + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and (
            "arguments" in obj or _CLOSE_TAG_HEAD_RE.match(s[end:])
        ):
            tc = _parse_tool_obj(obj)
            if tc is not None:
                calls.append(tc)
            span_start, span_end = pos, end
            cm = _CLOSE_TAG_HEAD_RE.match(s[span_end:])
            if cm:
                span_end += cm.end()
            fm_open = _OPEN_FENCE_TAIL_RE.search(s[last:span_start])
            fm_close = _CLOSE_FENCE_HEAD_RE.match(s[span_end:])
            if fm_open and fm_close:
                span_start = last + fm_open.start()
                span_end += fm_close.end()
            out.append(s[last:span_start])
            last = span_end
            i = span_end
        else:
            i = end
    out.append(s[last:])
    cleaned = _STRAY_TOOL_TAG_RE.sub("", "".join(out)).strip()
    return cleaned, calls


_STREAM_TRIGGERS_RE = re.compile(
    r"(?P<think><think>)"
    r"|(?P<tool_tag>(?:```[a-zA-Z]*\s*)?<tool_call>)"
    r"|(?P<close_tag></tool_call>)"
    r'|(?P<tool_json>(?:```[a-zA-Z]*\s*)?\{\s*"name")',
    re.IGNORECASE,
)
_STREAM_JSON_NAME_RE = re.compile(r'^\{\s*"name"')


def _partial_trigger_index(buf: str) -> int:
    """Return earliest index in buf that could be the start of a split trigger."""
    n = len(buf)
    hold = n
    for marker in ("<think>", "<tool_call>", "</tool_call>"):
        low = buf.lower()
        for k in range(min(len(marker) - 1, n), 0, -1):
            if low.endswith(marker[:k]):
                hold = min(hold, n - k)
                break
    # Check trailing `{...` that could still become `{"name"`
    brace_idx = buf.rfind("{")
    if brace_idx != -1 and brace_idx < hold:
        tail = buf[brace_idx:]
        if len(tail) < 16 and (
            _STREAM_JSON_NAME_RE.match(tail)
            or '{"name"'.startswith(re.sub(r"^\{\s*", "{", tail))
        ):
            hold = min(hold, brace_idx)
    # Check trailing backtick fence that could precede `<tool_call>` or `{"name"`
    tick_idx = buf.rfind("`")
    if tick_idx != -1 and tick_idx < hold:
        tail = buf[tick_idx:]
        if len(tail) < 24 and re.match(r"^`{1,3}[a-zA-Z]*\s*(?:<|\{\s*(?:\"(?:n(?:a(?:me?)?)?)?)?)?$", tail):
            hold = min(hold, tick_idx)
    return hold


async def _filter_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Strip <think>, <tool_call>...</tool_call>, and {"name"...} JSON across split chunks."""
    buf = ""
    mode = "normal"  # normal | think | tool_tag | tool_json
    drop_fence = False
    depth = 0
    in_str = False
    escape = False

    async for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        while buf:
            if mode == "normal":
                if drop_fence:
                    m_f = _CLOSE_FENCE_HEAD_RE.match(buf)
                    if m_f:
                        buf = buf[m_f.end():]
                        drop_fence = False
                        continue
                    if buf.lstrip().startswith("`") and len(buf.lstrip()) < 3:
                        break
                    drop_fence = False
                m = _STREAM_TRIGGERS_RE.search(buf)
                if m:
                    if m.start() > 0:
                        yield buf[: m.start()]
                    kind = m.lastgroup
                    if kind == "think":
                        buf = buf[m.end():]
                        mode = "think"
                    elif kind == "tool_tag":
                        drop_fence = m.group("tool_tag").lstrip().startswith("```")
                        buf = buf[m.end():]
                        mode = "tool_tag"
                    elif kind == "close_tag":
                        buf = buf[m.end():]
                    else:  # tool_json
                        grp = m.group("tool_json")
                        drop_fence = grp.lstrip().startswith("```")
                        brace_pos = m.start() + grp.index("{")
                        buf = buf[brace_pos:]
                        mode = "tool_json"
                        depth = 0
                        in_str = False
                        escape = False
                    continue
                hold = _partial_trigger_index(buf)
                if hold > 0:
                    yield buf[:hold]
                    buf = buf[hold:]
                break

            if mode == "think":
                idx = buf.lower().find("</think>")
                if idx != -1:
                    buf = buf[idx + 8 :]
                    mode = "normal"
                    continue
                if len(buf) > 8:
                    buf = buf[-8:]
                break

            if mode == "tool_tag":
                idx = buf.lower().find("</tool_call>")
                if idx != -1:
                    buf = buf[idx + 12 :]
                    mode = "normal"
                    continue
                if len(buf) > 12:
                    buf = buf[-12:]
                break

            # mode == "tool_json"
            closed_at = -1
            for j, ch in enumerate(buf):
                if escape:
                    escape = False
                    continue
                if in_str:
                    if ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed_at = j + 1
                        break
            if closed_at != -1:
                buf = buf[closed_at:]
                cm = _CLOSE_TAG_HEAD_RE.match(buf)
                if cm:
                    buf = buf[cm.end():]
                mode = "normal"
                continue
            buf = ""
            break

    if mode == "normal" and buf:
        if drop_fence:
            buf = _CLOSE_FENCE_HEAD_RE.sub("", buf)
        rem = _STRAY_TOOL_TAG_RE.sub("", buf)
        if rem:
            yield rem


class LLMClient:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=8.0))
        self._no_tools_models: set[str] = set()
        self._gemini_client: Any = None
        self._gemini_api_key: str = ""
        self.last_ok: bool = False
        self.last_ok_ts: float = 0.0
        self._last_probe_ts: float = 0.0

    def _mark_ok(self) -> None:
        self.last_ok = True
        self.last_ok_ts = time.time()

    def _tools_key(self, provider: str | None = None, model: str | None = None) -> str:
        p = provider or config.llm_provider
        m = config.active_model() if model is None else model
        return f"{p}:{m}"

    def _openai_endpoint(self, provider: str) -> _OpenAIEndpoint:
        if provider == "ninerouter":
            return _OpenAIEndpoint(
                name="ninerouter",
                base_url=config.ninerouter_base_url,
                api_key=config.ninerouter_api_key,
                model=config.ninerouter_model,
            )
        return _OpenAIEndpoint(
            name="openai_compat",
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            model=config.openai_model,
        )

    def _log_openai_status(self, ep: _OpenAIEndpoint, status_code: int) -> None:
        label = "9router" if ep.name == "ninerouter" else ep.name
        if status_code in (401, 403):
            log.error("%s API key missing or wrong", label)
        elif status_code == 429:
            log.warning("%s rate limited (HTTP 429)", label)

    def _log_openai_connect_error(self, ep: _OpenAIEndpoint) -> None:
        label = "9router" if ep.name == "ninerouter" else ep.name
        log.error("%s is not running at %s", label, ep.base_url.rstrip("/"))

    async def check_health(self, max_age: float = 60.0) -> bool:
        """Return recent LLM health, probing local/gateway endpoints at most once per max_age."""
        now = time.time()
        if self.last_ok and (now - self.last_ok_ts) < max_age:
            return True
        provider = config.llm_provider
        if provider == "gemini":
            return self.last_ok
        if (now - self._last_probe_ts) >= max_age:
            self._last_probe_ts = now
            try:
                if provider in ("ninerouter", "openai_compat"):
                    ep = self._openai_endpoint(provider)
                    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
                    r = await self._http.get(
                        f"{ep.base_url.rstrip('/')}/models",
                        headers=headers,
                        timeout=2.0,
                    )
                else:
                    r = await self._http.get(f"{config.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
                self.last_ok = r.status_code == 200
                if self.last_ok:
                    self.last_ok_ts = now
            except Exception:
                self.last_ok = False
        return self.last_ok

    def supports_tools(self) -> bool:
        return self._tools_key() not in self._no_tools_models

    def _get_gemini_client(self) -> Any:
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty and local Ollama is not responding.")
        if self._gemini_client is None or self._gemini_api_key != config.gemini_api_key:
            from google import genai

            self._gemini_client = genai.Client(api_key=config.gemini_api_key)
            self._gemini_api_key = config.gemini_api_key
        return self._gemini_client

    async def close(self) -> None:
        await self._http.aclose()
        if self._gemini_client is not None:
            client = self._gemini_client
            self._gemini_client = None
            self._gemini_api_key = ""
            aio = getattr(client, "aio", None)
            aclose = getattr(aio, "aclose", None)
            if callable(aclose):
                res = aclose()
                if inspect.isawaitable(res):
                    await res
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

    def active_provider(self) -> str:
        return config.llm_provider

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        fast_voice: bool = False,
        max_tokens_scale: int = 1,
    ) -> LLMResponse:
        provider = config.llm_provider
        scale = max(1, int(max_tokens_scale))
        try:
            if provider == "ollama":
                res = await self._ollama_once(messages, tools=tools, fast_voice=fast_voice, max_tokens_scale=scale)
            elif provider in ("ninerouter", "openai_compat"):
                res = await self._openai_once(
                    messages,
                    tools=tools,
                    fast_voice=fast_voice,
                    endpoint=self._openai_endpoint(provider),
                    max_tokens_scale=scale,
                )
            elif provider == "gemini":
                res = await self._gemini_once(messages, tools=tools, max_tokens_scale=scale)
            else:
                res = await self._ollama_once(messages, tools=tools, fast_voice=fast_voice, max_tokens_scale=scale)
            self._mark_ok()
            return res
        except Exception as e:
            log.warning("Primary provider '%s' failed (%s), checking fallback...", provider, redact_secrets(str(e)))
            if provider != "gemini" and config.gemini_api_key:
                log.info("Falling back to Gemini API...")
                try:
                    res = await self._gemini_once(messages, tools=tools, max_tokens_scale=scale)
                    self._mark_ok()
                    return res
                except Exception:
                    self.last_ok = False
                    raise
            self.last_ok = False
            raise

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
                self._mark_ok()
                return
            if provider in ("ninerouter", "openai_compat"):
                async for tok in self._openai_stream(
                    messages,
                    fast_voice=fast_voice,
                    endpoint=self._openai_endpoint(provider),
                ):
                    yield tok
                self._mark_ok()
                return
        except Exception as e:
            log.warning("Streaming '%s' failed (%s), trying fallback...", provider, redact_secrets(str(e)))
            if provider != "gemini" and config.gemini_api_key:
                try:
                    resp = await self._gemini_once(messages)
                    self._mark_ok()
                    if resp.text:
                        yield resp.text
                    return
                except Exception:
                    self.last_ok = False
                    raise
            self.last_ok = False
            raise

        try:
            resp = await self._gemini_once(messages)
            self._mark_ok()
            if resp.text:
                yield resp.text
        except Exception:
            self.last_ok = False
            raise

    # ---------------- Ollama Native (/api/chat) ----------------
    async def _ollama_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        fast_voice: bool,
        max_tokens_scale: int = 1,
    ) -> LLMResponse:
        model = config.ollama_model
        scale = max(1, int(max_tokens_scale))
        payload: dict[str, Any] = {
            "model": model,
            "messages": normalize_messages(messages),
            "stream": False,
            "keep_alive": config.ollama_keep_alive,
            "think": False,  # Disable long chain-of-thought for instant replies
            "options": {
                **_temperature_kw(),
                "num_predict": (320 if fast_voice else (300 if config.llm_compact_prompt else 650)) * scale,
            },
        }
        tools_key = self._tools_key("ollama", model)
        if tools and tools_key not in self._no_tools_models:
            payload["tools"] = tools

        url = f"{config.ollama_base_url.rstrip('/')}/api/chat"
        r = await self._http.post(url, json=payload)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think", None)
            r = await self._http.post(url, json=payload)
        if r.status_code == 400 and "tools" in payload and "does not support tools" in r.text.lower():
            if tools_key not in self._no_tools_models:
                self._no_tools_models.add(tools_key)
                log.warning("Ollama model '%s' does not support tools; disabling tools until restart.", model)
            payload.pop("tools", None)
            r = await self._http.post(url, json=payload)
            if r.status_code == 400 and "think" in r.text.lower():
                payload.pop("think", None)
                r = await self._http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        raw_content = msg.get("content", "") or ""
        has_reasoning = bool(
            msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content") or "<think>" in raw_content.lower()
        )
        content = _strip_think(raw_content)
        content, text_calls = extract_tool_calls_from_text(content)

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
        tcalls.extend(text_calls)
        return LLMResponse(
            text=content,
            tool_calls=tcalls,
            finish_reason=data.get("done_reason"),
            has_reasoning=has_reasoning,
            provider="ollama",
            model=model,
        )

    async def _ollama_stream(self, messages: list[dict[str, Any]], fast_voice: bool) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": config.ollama_model,
            "messages": normalize_messages(messages),
            "stream": True,
            "keep_alive": config.ollama_keep_alive,
            "think": False,
            "options": {
                **_temperature_kw(),
                "num_predict": 280 if fast_voice else (300 if config.llm_compact_prompt else 600),
            },
        }
        url = f"{config.ollama_base_url.rstrip('/')}/api/chat"

        async def _raw_tokens() -> AsyncIterator[str]:
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
                    tok = json.loads(line).get("message", {}).get("content", "")
                    if tok:
                        yield tok

        async for tok in _filter_stream(_raw_tokens()):
            yield tok

    # ---------------- OpenAI-Compatible (9router / LM Studio / llama.cpp) ----------------
    async def _openai_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        fast_voice: bool,
        endpoint: _OpenAIEndpoint | None = None,
        max_tokens_scale: int = 1,
    ) -> LLMResponse:
        ep = endpoint or self._openai_endpoint("openai_compat")
        tools_key = self._tools_key(ep.name, ep.model)
        scale = max(1, int(max_tokens_scale))
        payload: dict[str, Any] = {
            "model": ep.model,
            "messages": normalize_messages(messages),
            **_temperature_kw(),
            "max_tokens": (320 if fast_voice else (300 if config.llm_compact_prompt else 650)) * scale,
        }
        if tools and tools_key not in self._no_tools_models:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        url = f"{ep.base_url.rstrip('/')}/chat/completions"
        try:
            r = await self._http.post(url, json=payload, headers=headers)
        except httpx.ConnectError:
            self._log_openai_connect_error(ep)
            raise
        if r.status_code in (400, 422) and "tools" in payload and _TOOLS_REJECTED_RE.search(r.text):
            payload.pop("tools", None)
            try:
                r = await self._http.post(url, json=payload, headers=headers)
            except httpx.ConnectError:
                self._log_openai_connect_error(ep)
                raise
            if r.status_code < 400 and tools_key not in self._no_tools_models:
                self._no_tools_models.add(tools_key)
                log.warning("%s model '%s' rejected tools; disabling tools until restart.", ep.name, ep.model)
        if r.status_code >= 400:
            self._log_openai_status(ep, r.status_code)
            r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
        choice = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
        raw_content = choice.get("content") or ""
        has_reasoning = bool(
            choice.get("reasoning")
            or choice.get("reasoning_content")
            or choice.get("thinking")
            or "<think>" in raw_content.lower()
        )
        content = _strip_think(raw_content)
        content, text_calls = extract_tool_calls_from_text(content)
        tcalls: list[LLMToolCall] = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    args = {"query": raw_args}
            else:
                args = raw_args if isinstance(raw_args, dict) else {}
            if fn.get("name"):
                tcalls.append(LLMToolCall(name=fn["name"], arguments=args))
        tcalls.extend(text_calls)
        return LLMResponse(
            text=content,
            tool_calls=tcalls,
            finish_reason=choice0.get("finish_reason"),
            has_reasoning=has_reasoning,
            provider=ep.name,
            model=ep.model,
        )

    async def _openai_stream(
        self,
        messages: list[dict[str, Any]],
        fast_voice: bool,
        endpoint: _OpenAIEndpoint | None = None,
    ) -> AsyncIterator[str]:
        ep = endpoint or self._openai_endpoint("openai_compat")
        payload = {
            "model": ep.model,
            "messages": normalize_messages(messages),
            **_temperature_kw(),
            "max_tokens": 280 if fast_voice else (300 if config.llm_compact_prompt else 600),
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        url = f"{ep.base_url.rstrip('/')}/chat/completions"

        async def _raw_tokens() -> AsyncIterator[str]:
            logged_bad_json = False
            try:
                async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        self._log_openai_status(ep, resp.status_code)
                        resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            break
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            if not logged_bad_json:
                                log.debug("Skipping invalid SSE JSON line from %s: %r", ep.name, raw[:120])
                                logged_bad_json = True
                            continue
                        if not isinstance(obj, dict):
                            continue
                        choices = obj.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice0 = choices[0]
                        delta = choice0.get("delta") if isinstance(choice0, dict) else None
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            yield content
            except httpx.ConnectError:
                self._log_openai_connect_error(ep)
                raise

        async for tok in _filter_stream(_raw_tokens()):
            yield tok

    # ---------------- Gemini Cloud / Fallback ----------------
    async def _gemini_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens_scale: int = 1,
    ) -> LLMResponse:
        client = self._get_gemini_client()
        from google.genai import types
        scale = max(1, int(max_tokens_scale))
        sys_chunks: list[str] = []
        contents: list[types.Content] = []
        for m in normalize_messages(messages):
            if m["role"] == "system":
                sys_chunks.append(m["content"])
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        gemini_tools = _to_gemini_tools(types, tools)
        for model in (config.gemini_model, *config.gemini_fallback_models):
            if not model:
                continue
            try:
                cfg_kw: dict[str, Any] = {
                    "system_instruction": "\n\n".join(sys_chunks),
                    **_temperature_kw(),
                    "max_output_tokens": 600 * scale,
                }
                if gemini_tools and self._tools_key("gemini", model) not in self._no_tools_models:
                    cfg_kw["tools"] = gemini_tools
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg_kw),
                )
                text_parts: list[str] = []
                tcalls: list[LLMToolCall] = []
                has_reasoning = False
                finish_reason: str | None = None
                candidates = getattr(resp, "candidates", None) or []
                if candidates:
                    fr = getattr(candidates[0], "finish_reason", None)
                    finish_reason = str(fr) if fr is not None else None
                if candidates and getattr(candidates[0], "content", None):
                    for part in getattr(candidates[0].content, "parts", None) or []:
                        fc = getattr(part, "function_call", None)
                        if fc is not None and getattr(fc, "name", None):
                            args = getattr(fc, "args", None)
                            tcalls.append(LLMToolCall(name=fc.name, arguments=dict(args) if isinstance(args, dict) else {}))
                        elif getattr(part, "thought", False):
                            if getattr(part, "text", None):
                                has_reasoning = True
                        elif getattr(part, "text", None):
                            text_parts.append(part.text)
                else:
                    for fc in getattr(resp, "function_calls", None) or []:
                        if getattr(fc, "name", None):
                            args = getattr(fc, "args", None)
                            tcalls.append(LLMToolCall(name=fc.name, arguments=dict(args) if isinstance(args, dict) else {}))
                    if getattr(resp, "text", None):
                        text_parts.append(resp.text)

                raw_joined = "".join(text_parts)
                if "<think>" in raw_joined.lower():
                    has_reasoning = True
                content = _strip_think(raw_joined)
                content, text_calls = extract_tool_calls_from_text(content)
                tcalls.extend(text_calls)
                return LLMResponse(
                    text=content,
                    tool_calls=tcalls,
                    finish_reason=finish_reason,
                    has_reasoning=has_reasoning,
                    provider="gemini",
                    model=model,
                )
            except Exception as e:
                log.warning("Gemini model '%s' failed: %s", model, redact_secrets(str(e)))
        raise RuntimeError("All Gemini models failed.")
