"""Indonesian reply evaluation script for local Ollama models (no LLM scoring)."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.llm import LLMClient
from brain.profile import ProfileManager
from brain.prompt import assemble_messages, detect_register, normalize_register
from brain.runner import _PREFIX_STRIP_RE, _assistant_tool_turn, parse_reminder_request
from brain.tools import TOOL_SCHEMAS
from config import DATA_DIR, config
from core.safety import filter_outgoing
from heart.engine import extract_tags

TEST_MESSAGES: list[tuple[str, bool]] = [
    ("halo, lagi ngapain?", False),
    ("gacor juga kamu, ngobrol yuk", False),
    ("kamu ngomong apa sih?", False),
    ("kamu VTuber bukan?", False),
    ("kamu suka kopi gak?", False),
    ("wkwk iya juga ya", False),
    ("capek banget abis kuliah", False),
    ("ceritain dikit dong soal diri kamu", False),
    ("ingatkan aku minum 1 menit lagi", True),
    ("jelasin singkat apa itu RAM", False),
    ("aku mau cerita dong, kamu mau dengerin gak?", False),
    ("gue mau cerita dong, lo mau dengerin gak?", False),
]

_TAG_LEAK_RE = re.compile(
    r"</?(?:feel|emo|data|now|turn|style_note|think|tool_call|silence|system|examples)\b|<emo:[^>]*$|<feel\b[^>]*$",
    re.IGNORECASE,
)
_JSON_LEAK_RE = re.compile(
    r'\{\s*"(?:name|arguments|minutes|note|query|fact|self|user)"\s*:|```json\b',
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_WORD_RE = re.compile(r"[a-z']+")
_GUE_PRONOUN_RE = re.compile(r"\b(?:gue|gw|lo|lu|elo|elu)\b", re.IGNORECASE)
_AKU_PRONOUN_RE = re.compile(r"\b(?:aku|kamu)\b", re.IGNORECASE)
_SAYA_PRONOUN_RE = re.compile(r"\bsaya\b", re.IGNORECASE)


def has_wrong_pronoun(reply: str, expected_register: str) -> bool:
    if _SAYA_PRONOUN_RE.search(reply):
        return True
    if expected_register == "gue":
        return bool(_AKU_PRONOUN_RE.search(reply))
    return bool(_GUE_PRONOUN_RE.search(reply))

_EN_WORDS = frozenset({
    "the", "is", "are", "was", "were", "you", "your", "i'm", "don't", "can't",
    "won't", "it's", "that's", "let's", "what", "how", "with", "from", "have",
    "has", "not", "but", "and", "this", "that", "about", "just", "really",
    "something", "anything", "please", "would", "could", "should", "because",
    "when", "where", "they", "them", "their", "there", "here", "know", "think",
    "want", "need", "help", "sorry", "hello", "thanks", "sure", "actually",
    "does", "did", "will", "can", "been", "being", "my", "me", "we", "our",
    "she", "he", "his", "her",
})
_ID_WORDS = frozenset({
    "aku", "kamu", "gue", "lo", "yang", "dan", "di", "ke", "dari", "untuk",
    "ini", "itu", "lagi", "udah", "belum", "nggak", "gak", "bukan", "kalau",
    "tapi", "bisa", "ada", "aja", "banget", "kok", "sih", "dong", "deh",
    "nih", "ya", "yuk", "juga", "sama", "buat", "mau", "tahu", "kayak",
    "cuma", "emang", "saja", "adalah",
})


def detect_leaks(clean_text: str) -> dict[str, bool]:
    tag_leak = bool(_TAG_LEAK_RE.search(clean_text))
    json_leak = bool(_JSON_LEAK_RE.search(clean_text))
    english_leak = False
    for sent in _SENTENCE_SPLIT_RE.split(clean_text):
        words = _WORD_RE.findall(sent.lower())
        if not words:
            continue
        en_hits = sum(1 for w in words if w in _EN_WORDS)
        id_hits = sum(1 for w in words if w in _ID_WORDS)
        if (en_hits >= 3 and en_hits > id_hits) or en_hits >= 4:
            english_leak = True
            break
    return {"tag_leak": tag_leak, "json_leak": json_leak, "english_leak": english_leak}


def _fake_tool_result(name: str, args: dict[str, Any]) -> str:
    if name == "set_reminder":
        mins = args.get("minutes", 1)
        note = args.get("note", "minum")
        return f"Reminder set for {mins} minutes from now: '{note}'"
    return "OK"


async def _run_eval(model: str, runs: int) -> Path:
    config.llm_provider = "ollama"
    config.ollama_model = model
    config.gemini_api_key = ""  # Never silently fall back to cloud Gemini

    profile = ProfileManager(None).current  # type: ignore[arg-type]
    fake_person = {
        "tier": "Kenalan Baru",
        "bond": 35.0,
        "trust": 20.0,
        "warmth": 0.10,
        "impression": "Teman ngobrol santai di server",
    }
    heart_state = {
        "dominant": "neutral",
        "valence": 0.25,
        "arousal": 0.35,
        "energy": 0.80,
        "active_emotions": {},
    }

    llm = LLMClient()
    results: list[dict[str, Any]] = []
    try:
        for run_idx in range(1, runs + 1):
            for prompt_text, expects_tool in TEST_MESSAGES:
                messages = assemble_messages(
                    profile=profile,
                    directives="",
                    heart_state=heart_state,
                    person=fake_person,
                    person_notes=[],
                    recalled_memories=[],
                    history=[],
                    speaker_name="Raka",
                    user_text=prompt_text,
                    modality="text",
                    wake_reason="direct_mention",
                )
                tools_called: list[str] = []
                t0 = time.perf_counter()
                resp = await llm.complete_with_tools(messages, tools=TOOL_SCHEMAS, fast_voice=False)
                if resp.tool_calls:
                    messages.append({"role": "assistant", "content": _assistant_tool_turn(resp.text, resp.tool_calls)})
                    for tc in resp.tool_calls:
                        tools_called.append(tc.name)
                        out = _fake_tool_result(tc.name, tc.arguments)
                        messages.append({"role": "user", "content": f"[Tool Result `{tc.name}`]:\n{out}"})
                    resp = await llm.complete_with_tools(messages, tools=None, fast_voice=False)
                if "set_reminder" not in tools_called:
                    parsed_rem = parse_reminder_request(prompt_text)
                    if parsed_rem is not None:
                        _fake_tool_result("set_reminder", {"minutes": parsed_rem[0], "note": parsed_rem[1]})
                        tools_called.append("set_reminder")
                elapsed = time.perf_counter() - t0

                raw = resp.text or ""
                clean, _, _ = extract_tags(raw)
                expected_reg = detect_register(prompt_text)
                clean = normalize_register(_PREFIX_STRIP_RE.sub("", filter_outgoing(clean)).strip(), expected_reg)
                leaks = detect_leaks(clean)
                wrong_pronoun = has_wrong_pronoun(clean, expected_reg)
                results.append({
                    "prompt": prompt_text,
                    "expects_tool": expects_tool,
                    "run": run_idx,
                    "reply": clean,
                    "length": len(clean),
                    "tools_called": tools_called,
                    "time_s": elapsed,
                    "wrong_pronoun": wrong_pronoun,
                    **leaks,
                })
    finally:
        await llm.close()

    total = len(results)
    tag_leaks = sum(1 for r in results if r["tag_leak"])
    json_leaks = sum(1 for r in results if r["json_leak"])
    en_leaks = sum(1 for r in results if r["english_leak"])
    wrong_pronouns = sum(1 for r in results if r["wrong_pronoun"])
    avg_len = sum(r["length"] for r in results) / total if total else 0.0
    avg_time = sum(r["time_s"] for r in results) / total if total else 0.0
    tool_expected = [r for r in results if r["expects_tool"]]
    tool_ok = sum(1 for r in tool_expected if "set_reminder" in r["tools_called"])
    unexpected_tools = sum(1 for r in results if not r["expects_tool"] and r["tools_called"])

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"lang_eval_{safe_model}_{ts}.md"

    lines = [
        f"# Language Eval: `{model}`",
        "",
        f"- **Timestamp:** {ts}",
        f"- **Runs:** {runs} ({total} replies)",
        f"- **LLM_COMPACT_PROMPT:** {getattr(config, 'llm_compact_prompt', False)}",
        f"- **LLM_TEMPERATURE:** {config.llm_temperature}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Tag leaks | {tag_leaks}/{total} |",
        f"| JSON leaks | {json_leaks}/{total} |",
        f"| Total tag/JSON leaks | {tag_leaks + json_leaks}/{total} |",
        f"| English sentence leaks | {en_leaks}/{total} |",
        f"| Wrong pronoun | {wrong_pronouns}/{total} |",
        f"| Average length (chars) | {avg_len:.1f} |",
        f"| Tool call OK (`set_reminder`) | {tool_ok}/{len(tool_expected)} |",
        f"| Unexpected tool calls | {unexpected_tools}/{total - len(tool_expected)} |",
        f"| Average reply time (s) | {avg_time:.2f} |",
        "",
        "## Replies",
        "",
    ]
    for r in results:
        tools_str = ", ".join(r["tools_called"]) if r["tools_called"] else "-"
        lines.extend([
            f"### Run {r['run']} — `{r['prompt']}`",
            f"- **Length:** {r['length']} chars | **Time:** {r['time_s']:.2f}s | **Tools:** {tools_str}",
            f"- **Leaks:** tag={r['tag_leak']}, json={r['json_leak']}, english={r['english_leak']}, wrong_pronoun={r['wrong_pronoun']}",
            "",
            f"> {r['reply'].replace(chr(10), chr(10) + '> ') if r['reply'] else '(empty)'}",
            "",
        ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out_path}")
    print(
        f"Summary [{model}]: tag_leaks={tag_leaks}/{total}, json_leaks={json_leaks}/{total}, "
        f"english_leaks={en_leaks}/{total}, wrong_pronoun={wrong_pronouns}/{total}, avg_len={avg_len:.1f}, "
        f"tool_ok={tool_ok}/{len(tool_expected)}, avg_time={avg_time:.2f}s"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pachinko Indonesian replies on local Ollama.")
    parser.add_argument("--model", default=config.ollama_model, help="Ollama model name (default from .env)")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per prompt (default: 2)")
    args = parser.parse_args()
    asyncio.run(_run_eval(args.model, max(1, args.runs)))


if __name__ == "__main__":
    main()
