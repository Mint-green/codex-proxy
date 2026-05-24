"""
Codex Debug Proxy — OpenAI Responses API ↔ Chat Completions API
Supports any OpenAI-compatible provider (DeepSeek, MiniMax, etc.)
Single port — proxy API + traffic inspector UI.

Usage:
  python proxy_app.py                  # default config (deepseek)
  python proxy_app.py minimax          # configs/minimax.toml
  python proxy_app.py --config path/to/custom.toml
"""
import json
import time
import uuid
import hashlib
import asyncio
import os
import glob
import logging
import threading
import tomllib
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections import deque

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Codex Debug Proxy")
parser.add_argument("config", nargs="?", default="deepseek",
                    help="Config name (e.g. deepseek, minimax) or path to .toml file")
parser.add_argument("--config-dir", default="configs",
                    help="Config directory (default: configs)")
args = parser.parse_args()

# Resolve config path
if args.config.endswith(".toml"):
    config_path = Path(args.config)
else:
    config_path = Path(args.config_dir) / f"{args.config}.toml"

if not config_path.exists():
    print(f"ERROR: Config file not found: {config_path}")
    sys.exit(1)

with open(config_path, "rb") as f:
    cfg = tomllib.load(f)

# ── Config ───────────────────────────────────────────────
PROXY_PORT      = cfg["server"]["port"]
UPSTREAM_BASE   = cfg["upstream"]["base_url"]
UPSTREAM_KEY    = cfg["upstream"]["api_key"]
UPSTREAM_TIMEOUT = cfg["upstream"].get("timeout", 120)
MODEL_MAP       = cfg["models"]
REASONING_MAP   = cfg.get("reasoning", {})
CONFIG_NAME     = config_path.stem
TRACE_DIR       = Path(cfg["traces"]["dir"])
# "field" = upstream uses reasoning_content field (DeepSeek)
# "think_tags" = upstream uses <think> tags in content (MiniMax)
REASONING_FMT   = cfg["upstream"].get("reasoning_format", "field")
EXTRA_PARAMS    = cfg["upstream"].get("extra_params", {})

# ── Reasoning DB ─────────────────────────────────────────
import sqlite3

class ReasoningDB:
    """Persistent key → reasoning store (SQLite). Keys are prefixed:
    "call:{call_id}" for tool-call reasoning, "hash:{int}" for content fingerprint.
    Survives proxy restarts."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # Check for old schema (call_id) and migrate to new (key)
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info('reasoning')")}
        if cols and "call_id" in cols and "key" not in cols:
            self._conn.execute(
                "CREATE TABLE reasoning_new (key TEXT PRIMARY KEY, reasoning TEXT NOT NULL, created_at TEXT)"
            )
            self._conn.execute(
                "INSERT INTO reasoning_new SELECT 'call:' || call_id, reasoning, created_at FROM reasoning"
            )
            self._conn.execute("DROP TABLE reasoning")
            self._conn.execute("ALTER TABLE reasoning_new RENAME TO reasoning")
            self._conn.commit()
        else:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS reasoning "
                "(key TEXT PRIMARY KEY, reasoning TEXT NOT NULL, created_at TEXT)"
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT reasoning FROM reasoning WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def save(self, key: str, reasoning: str):
        if not reasoning or not key:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO reasoning (key, reasoning, created_at) "
            "VALUES (?, ?, ?)",
            (key, reasoning, datetime.now(timezone.utc).isoformat())
        )
        self._conn.commit()

reasoning_db = ReasoningDB(TRACE_DIR / "reasoning.db")

# ── Logging ────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"proxy_{CONFIG_NAME}.log"

from logging.handlers import RotatingFileHandler

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_file_handler = RotatingFileHandler(
    str(LOG_FILE), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))

logger = logging.getLogger("proxy")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)
logger.propagate = False

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from starlette.status import HTTP_200_OK
import uvicorn

# ── FastAPI app ─────────────────────────────────────────
app = FastAPI(title=f"Codex Debug Proxy [{CONFIG_NAME}]")
http_client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT))

# ═══════════════════════════════════════════════════════
#  SessionStore — port of codex-relay/src/session.rs
# ═══════════════════════════════════════════════════════

class SessionStore:
    """Maps response_id → accumulated message history.
    Unified reasoning store: key is "call:{call_id}" or "hash:{int}", backed by SQLite."""
    def __init__(self):
        self._history: dict[str, list[dict]] = {}          # response_id → [ChatMessage]
        self._reasoning: dict[str, str] = {}               # "call:id" | "hash:int" → reasoning

    def store_reasoning(self, call_id: str, reasoning: str):
        if reasoning and call_id:
            key = f"call:{call_id}"
            self._reasoning[key] = reasoning
            reasoning_db.save(key, reasoning)

    def get_reasoning(self, call_id: str) -> Optional[str]:
        key = f"call:{call_id}"
        rc = self._reasoning.get(key)
        if rc:
            return rc
        rc = reasoning_db.get(key)
        if rc:
            self._reasoning[key] = rc
        return rc

    def _fingerprint(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def store_turn_reasoning(self, assistant: dict, reasoning: str):
        """Store reasoning by content fingerprint so it can be recovered when Codex
        replays conversation without previous_response_id. Uses MD5 for stable key
        across restarts (Python hash() is randomized per-process)."""
        if not reasoning:
            return
        content = assistant.get("content")
        if isinstance(content, str) and content:
            key = f"hash:{self._fingerprint(content)}"
            self._reasoning[key] = reasoning
            reasoning_db.save(key, reasoning)
        for tc in (assistant.get("tool_calls") or []):
            cid = tc.get("id", "")
            if cid:
                self.store_reasoning(cid, reasoning)

    def get_turn_reasoning(self, assistant: dict) -> Optional[str]:
        content = assistant.get("content")
        if isinstance(content, str) and content:
            key = f"hash:{self._fingerprint(content)}"
            rc = self._reasoning.get(key)
            if rc:
                return rc
            rc = reasoning_db.get(key)
            if rc:
                self._reasoning[key] = rc
            return rc
        return None

    def get_history(self, response_id: str) -> list[dict]:
        return self._history.get(response_id, [])

    def new_id(self) -> str:
        return f"resp_{uuid.uuid4().hex[:24]}"

    def save_with_id(self, rid: str, messages: list[dict]):
        self._history[rid] = messages

    def save(self, messages: list[dict]) -> str:
        rid = self.new_id()
        self._history[rid] = messages
        return rid


sessions = SessionStore()


# ═══════════════════════════════════════════════════════
#  Usage conversion
# ═══════════════════════════════════════════════════════

def _convert_usage(chat_usage: dict | None) -> dict | None:
    if not chat_usage:
        return None
    u = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }
    if "prompt_tokens_details" in chat_usage:
        u["input_tokens_details"] = chat_usage["prompt_tokens_details"]
    if "completion_tokens_details" in chat_usage:
        u["output_tokens_details"] = chat_usage["completion_tokens_details"]
    return u


def _apply_reasoning(msg: dict, reasoning: str):
    """Apply reasoning to a message in the format the upstream expects."""
    if REASONING_FMT == "think_tags":
        existing = msg.get("content") or ""
        msg["content"] = f"<think>{reasoning}</think>\n\n{existing}" if existing else f"<think>{reasoning}</think>"
    else:
        msg["reasoning_content"] = reasoning


# ═══════════════════════════════════════════════════════
#  translate — port of codex-relay/src/translate.rs
# ═══════════════════════════════════════════════════════

def _value_to_chat_content(content) -> Optional[str | list[dict]]:
    """Convert Responses API content → Chat Completions content."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Check if any part is non-text (e.g. input_image)
        has_non_text = any(
            p.get("type", "") not in ("input_text", "text", "output_text")
            for p in content if isinstance(p, dict)
        )
        if not has_non_text:
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        # Multimodal: map each part
        mapped = []
        for part in content:
            if not isinstance(part, dict):
                mapped.append({"type": "text", "text": str(part)})
                continue
            kind = part.get("type", "")
            if kind in ("input_text", "text", "output_text"):
                mapped.append({"type": "text", "text": part.get("text", "")})
            elif kind == "input_image":
                url = part.get("image_url", "")
                mapped.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "image_url":
                inner = part.get("image_url", "")
                if isinstance(inner, str):
                    inner = {"url": inner}
                mapped.append({"type": "image_url", "image_url": inner})
            else:
                mapped.append(part)
        return mapped
    return str(content)


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Responses API tools → Chat Completions tools."""
    out = []
    for t in tools:
        kind = t.get("type", "")
        if kind == "function":
            # Already Chat format?
            if "function" in t:
                out.append(t)
            else:
                fn = {}
                for k in ("name", "description", "parameters", "strict"):
                    if k in t:
                        fn[k] = t[k]
                out.append({"type": "function", "function": fn})
        elif kind == "namespace":
            # Codex 0.128+ MCP plugin grouping — splice in child functions
            for sub in t.get("tools", []):
                if sub.get("type") == "function":
                    if "function" in sub:
                        out.append(sub)
                    else:
                        fn = {}
                        for k in ("name", "description", "parameters", "strict"):
                            if k in sub:
                                fn[k] = sub[k]
                        out.append({"type": "function", "function": fn})
    return out


def responses_to_chat(body: dict) -> dict:
    """Convert Responses API request → Chat Completions request.
    Port of codex-relay translate::to_chat_request()."""
    from copy import deepcopy

    # Reconstruct history from previous_response_id
    prev_id = body.get("previous_response_id")
    messages: list[dict] = []
    if prev_id:
        messages = deepcopy(sessions.get_history(prev_id))
        # Convert reasoning_content to upstream format
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("reasoning_content"):
                rc = msg.pop("reasoning_content")
                _apply_reasoning(msg, rc)

    model_name = body.get("model", "")
    upstream_model = MODEL_MAP.get(model_name, model_name)

    # Prefer instructions over system (matching codex-relay)
    system_text = body.get("instructions") or body.get("system")
    if system_text:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_text})

    input_items = body.get("input", [])
    if isinstance(input_items, str):
        # Text-only input
        messages.append({"role": "user", "content": input_items})
        return _build_chat_body(upstream_model, messages, body)

    # ── Collect existing call_ids from history (for dedup) ──
    existing_call_ids: set[str] = set()
    for msg in messages:
        for tc in (msg.get("tool_calls") or []):
            cid = tc.get("id", "")
            if cid:
                existing_call_ids.add(cid)
        tid = msg.get("tool_call_id", "")
        if tid:
            existing_call_ids.add(tid)

    existing_tool_responses: set[str] = set()
    for msg in messages:
        tid = msg.get("tool_call_id", "")
        if tid:
            existing_tool_responses.add(tid)

    # ── Process input items ──
    i = 0
    items = input_items
    while i < len(items):
        item = items[i]
        item_type = item.get("type", "")

        if item_type == "function_call":
            call_id = item.get("call_id", "")
            # Skip if already in history (dedup)
            if call_id in existing_call_ids:
                i += 1
                continue

            # Group consecutive function_calls into one assistant message
            grouped: list[dict] = []
            reasoning_content: Optional[str] = None
            while i < len(items):
                cur = items[i]
                if cur.get("type", "") != "function_call":
                    break
                cid = cur.get("call_id", "")
                name = cur.get("name", "")
                args = cur.get("arguments", "{}")
                if reasoning_content is None:
                    reasoning_content = sessions.get_reasoning(cid)
                grouped.append({
                    "id": cid,
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })
                i += 1

            msg: dict = {"role": "assistant", "content": None, "tool_calls": grouped}
            if reasoning_content:
                _apply_reasoning(msg, reasoning_content)
            else:
                # Fallback: try turn-level fingerprint
                rc = sessions.get_turn_reasoning(msg)
                if rc:
                    _apply_reasoning(msg, rc)
            messages.append(msg)

        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            if call_id in existing_tool_responses:
                i += 1
                continue
            output = item.get("output", "")
            messages.append({
                "role": "tool",
                "content": str(output),
                "tool_call_id": call_id,
            })
            i += 1

        elif item_type == "reasoning":
            # Skip — reasoning is recovered from session store
            i += 1

        else:
            # Regular message (user/assistant/developer)
            role = item.get("role", "user")
            role = "system" if role == "developer" else role
            content = _value_to_chat_content(item.get("content"))
            msg = {"role": role, "content": content}

            # Preserve reasoning_content from input item (Codex may include it)
            rc_input = item.get("reasoning_content")
            if rc_input:
                _apply_reasoning(msg, rc_input)
            elif role == "assistant":
                rc = sessions.get_turn_reasoning(msg)
                if rc:
                    _apply_reasoning(msg, rc)

            # System/developer messages must go to front (interleaving fix)
            if role == "system":
                if messages and messages[0].get("role") == "system":
                    messages[0] = msg
                else:
                    messages.insert(0, msg)
            else:
                messages.append(msg)
            i += 1

    return _build_chat_body(upstream_model, messages, body)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Objects are recursed, everything else is overwritten."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _build_chat_body(model: str, messages: list[dict], body: dict) -> dict:
    """Build final Chat Completions request body."""
    chat_body = {"model": model, "messages": messages}

    if "tools" in body and body["tools"]:
        chat_tools = _convert_tools(body["tools"])
        if chat_tools:
            chat_body["tools"] = chat_tools

    if body.get("stream"):
        chat_body["stream"] = True
        chat_body["stream_options"] = {"include_usage": True}

    # Map reasoning.effort → upstream reasoning_effort
    raw_effort = body.get("reasoning_effort")
    if not raw_effort and isinstance(body.get("reasoning"), dict):
        raw_effort = body["reasoning"].get("effort", "")
    if raw_effort:
        if REASONING_MAP:
            mapped = REASONING_MAP.get(raw_effort)
            if mapped:
                chat_body["reasoning_effort"] = mapped
        else:
            chat_body["reasoning_effort"] = raw_effort

    for k in ("temperature", "max_output_tokens", "top_p"):
        if k in body:
            if k == "max_output_tokens":
                chat_body["max_tokens"] = body[k]
            else:
                chat_body[k] = body[k]

    if EXTRA_PARAMS:
        _deep_merge(chat_body, EXTRA_PARAMS)

    return chat_body


# ═══════════════════════════════════════════════════════
#  stream — port of codex-relay/src/stream.rs
# ═══════════════════════════════════════════════════════

def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_think_tags(text: str) -> tuple[str, str]:
    """Extract <think>...</think> tags from text.
    Returns (reasoning, clean_text)."""
    import re
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    matches = think_pattern.findall(text)
    if not matches:
        return "", text
    reasoning = "\n".join(m.strip() for m in matches)
    clean_text = think_pattern.sub("", text).strip()
    return reasoning, clean_text


async def _stream_responses(in_body: dict, chat_body: dict, t0: float):
    """Streaming: Chat SSE chunks → OpenAI Responses API SSE events.
    Port of codex-relay stream::translate_stream()."""
    resp_id = sessions.new_id()
    model = in_body.get("model", "")
    request_messages = chat_body.get("messages", [])

    # ── State ──
    msg_item_id = f"msg_{uuid.uuid4().hex[:16]}"
    accumulated_text = ""
    accumulated_reasoning = ""
    tool_calls: dict[int, dict] = {}  # index → {id, name, arguments}
    emitted_message_item = False
    emitted_reasoning_item = False
    reasoning_item_id = f"rs_{uuid.uuid4().hex[:16]}"
    msg_output_index = 0
    reasoning_output_index = -1
    stream_done = False
    all_chunks = []
    sse_events: list[dict] = []

    def _emit(event: str, data: dict):
        sse_events.append({"event": event, "data": data})
        return _sse(event, data)

    def _tool_calls_sorted():
        return [tool_calls[k] for k in sorted(tool_calls.keys())]

    try:
        # Build response reasoning echo — Codex reads this to know reasoning is present
        resp_reasoning = {}
        req_reasoning = in_body.get("reasoning") or {}
        if isinstance(req_reasoning, dict):
            resp_reasoning = {"effort": req_reasoning.get("effort", "medium")}
        # Always include summary since we emit reasoning items
        resp_reasoning["summary"] = "detailed"

        # response.created
        yield _emit("response.created", {
            "type": "response.created",
            "response": {
                "id": resp_id,
                "status": "in_progress",
                "model": model,
                "reasoning": resp_reasoning,
            }
        })

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST", f"{UPSTREAM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {UPSTREAM_KEY}",
                         "Content-Type": "application/json"},
                json=chat_body,
            ) as upstream:
                # Check for HTTP error
                if upstream.status_code >= 400:
                    body_text = await upstream.aread()
                    yield _emit("response.failed", {
                        "type": "response.failed",
                        "response": {"id": resp_id, "status": "failed",
                                     "error": {"code": str(upstream.status_code),
                                              "message": body_text.decode()[:500]}}
                    })
                    return

                async for line in upstream.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        stream_done = True
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    all_chunks.append(chunk)

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})

                        # Reasoning content (DeepSeek thinking mode)
                        rc = delta.get("reasoning_content", "")
                        if rc:
                            if not emitted_reasoning_item:
                                reasoning_output_index = 0
                                msg_output_index = 1
                                yield _emit("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {
                                        "type": "reasoning",
                                        "id": reasoning_item_id,
                                        "summary": []
                                    }
                                })
                                # Open summary part — fires ONCE per reasoning block
                                yield _emit("response.reasoning_summary_part.added", {
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": reasoning_item_id,
                                    "output_index": 0,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text", "text": ""}
                                })
                                emitted_reasoning_item = True
                            accumulated_reasoning += rc
                            yield _emit("response.reasoning_summary_text.delta", {
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": reasoning_item_id,
                                "output_index": 0,
                                "summary_index": 0,
                                "delta": rc
                            })

                        # Text content
                        content = delta.get("content", "")
                        if content:
                            if not emitted_message_item:
                                # Close reasoning item before starting message
                                if emitted_reasoning_item:
                                    yield _emit("response.reasoning_summary_text.done", {
                                        "type": "response.reasoning_summary_text.done",
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "summary_index": 0,
                                        "text": accumulated_reasoning
                                    })
                                    yield _emit("response.reasoning_summary_part.done", {
                                        "type": "response.reasoning_summary_part.done",
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "summary_index": 0,
                                        "part": {"type": "summary_text", "text": accumulated_reasoning}
                                    })
                                    yield _emit("response.output_item.done", {
                                        "type": "response.output_item.done",
                                        "output_index": 0,
                                        "item": {
                                            "type": "reasoning",
                                            "id": reasoning_item_id,
                                            "summary": [{"type": "summary_text", "text": accumulated_reasoning}]
                                        }
                                    })
                                yield _emit("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": msg_output_index,
                                    "item": {
                                        "type": "message",
                                        "id": msg_item_id,
                                        "role": "assistant",
                                        "status": "in_progress",
                                        "content": []
                                    }
                                })
                                emitted_message_item = True
                            accumulated_text += content
                            yield _emit("response.output_text.delta", {
                                "type": "response.output_text.delta",
                                "item_id": msg_item_id,
                                "output_index": msg_output_index,
                                "delta": content
                            })

                        # Tool call deltas
                        for tc_item in (delta.get("tool_calls") or []):
                            idx = tc_item.get("index", 0)
                            entry = tool_calls.setdefault(idx, {
                                "id": "", "name": "", "arguments": ""
                            })
                            if tc_item.get("id"):
                                entry["id"] = tc_item["id"]
                            fn = tc_item.get("function") or {}
                            if fn.get("name"):
                                entry["name"] += fn["name"]
                            if fn.get("arguments"):
                                entry["arguments"] += fn["arguments"]

        # Extract <think> tags from accumulated text (MiniMax returns thinking in content)
        if not accumulated_reasoning and accumulated_text:
            extracted_reasoning, clean_text = _extract_think_tags(accumulated_text)
            if extracted_reasoning:
                accumulated_reasoning = extracted_reasoning
                accumulated_text = clean_text

        # Emit reasoning item if accumulated but not yet sent
        # (reasoning-only or tool-call-only turn where no text content was streamed)
        if accumulated_reasoning and not emitted_reasoning_item and not emitted_message_item:
            reasoning_output_index = 0
            msg_output_index = 1
            yield _emit("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": accumulated_reasoning}]
                }
            })
            yield _emit("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": accumulated_reasoning}]
                }
            })
            emitted_reasoning_item = True

        # ── Close message item ──
        if emitted_message_item:
            yield _emit("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": msg_output_index,
                "item": {
                    "type": "message",
                    "id": msg_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": accumulated_text}]
                }
            })

        # ── Emit function_call items ──
        base_index = (msg_output_index + 1) if emitted_message_item else (1 if emitted_reasoning_item else 0)
        fc_items: list[dict] = []
        for rel_idx, tc in enumerate(_tool_calls_sorted()):
            fc_item_id = f"fc_{uuid.uuid4().hex[:16]}"
            output_index = base_index + rel_idx

            yield _emit("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": fc_item_id,
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": "",
                    "status": "in_progress"
                }
            })

            if tc["arguments"]:
                yield _emit("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": fc_item_id,
                    "output_index": output_index,
                    "delta": tc["arguments"]
                })

            yield _emit("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": fc_item_id,
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed"
                }
            })

            fc_items.append({
                "type": "function_call",
                "id": fc_item_id,
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
                "status": "completed"
            })

        if stream_done or accumulated_text or tool_calls:
            # ── Store reasoning for next turn ──
            for tc in _tool_calls_sorted():
                cid = tc.get("id", "")
                if cid:
                    sessions.store_reasoning(cid, accumulated_reasoning)

            # Build assistant message for session history
            assistant_tool_calls = None
            if tool_calls:
                assistant_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]}
                    }
                    for tc in _tool_calls_sorted()
                ]

            assistant_msg = {
                "role": "assistant",
                "content": accumulated_text if accumulated_text else None,
            }
            if accumulated_reasoning:
                assistant_msg["reasoning_content"] = accumulated_reasoning
            if assistant_tool_calls:
                assistant_msg["tool_calls"] = assistant_tool_calls

            # Store turn reasoning for fingerprint recovery
            if accumulated_reasoning:
                sessions.store_turn_reasoning(assistant_msg, accumulated_reasoning)

            # Save full conversation to session store
            full_history = list(request_messages) + [assistant_msg]
            sessions.save_with_id(resp_id, full_history)

            # Build output array
            output_items: list[dict] = []
            if accumulated_reasoning:
                output_items.append({
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": accumulated_reasoning}]
                })
            if emitted_message_item:
                output_items.append({
                    "type": "message",
                    "id": msg_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": accumulated_text}]
                })
            output_items.extend(fc_items)

            # Usage from last chunk
            usage = None
            if all_chunks:
                usage = _convert_usage(all_chunks[-1].get("usage"))

            yield _emit("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": resp_id,
                    "status": "completed",
                    "model": model,
                    "reasoning": resp_reasoning,
                    "output": output_items,
                    "usage": usage,
                }
            })

            logger.info(f"STREAM DONE resp_id={resp_id} text_len={len(accumulated_text)} "
                         f"tool_calls={len(tool_calls)} output_items={len(output_items)}")
            resp_entry = {"output": output_items}
            if usage:
                resp_entry["usage"] = usage
            await _capture_entry(in_body, chat_body, 200, resp_entry, t0, sse_events, all_chunks)
        else:
            # Stream incomplete
            logger.warning(f"STREAM INCOMPLETE resp_id={resp_id} — no [DONE]")
            yield _emit("response.failed", {
                "type": "response.failed",
                "response": {
                    "id": resp_id,
                    "status": "failed",
                    "error": {
                        "code": "stream_incomplete",
                        "message": "stream disconnected before completion"
                    }
                }
            })
            await _capture_entry(in_body, chat_body, 0, {"error": "stream_incomplete"}, t0, sse_events, all_chunks)

    except Exception as e:
        logger.error(f"STREAM ERROR resp_id={resp_id}: {e}", exc_info=True)
        yield _emit("response.failed", {
            "type": "response.failed",
            "response": {
                "id": resp_id,
                "status": "failed",
                "error": {"code": "proxy_error", "message": str(e)}
            }
        })
        await _capture_entry(in_body, chat_body, 502, {"error": str(e)}, t0, sse_events, all_chunks)


# ═══════════════════════════════════════════════════════
#  Capture store & WebSocket broadcast
# ═══════════════════════════════════════════════════════

captures: deque[dict] = deque(maxlen=200)
trace_entries: deque[dict] = deque(maxlen=200)  # claude-tap entry format for SSE
ws_clients: set[WebSocket] = set()
sse_queues: list[asyncio.Queue] = []
trace_lock = threading.Lock()


async def broadcast(cap: dict):
    # WebSocket clients — cap format (for inspector)
    ws_payload = json.dumps(cap, ensure_ascii=False, default=str)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(ws_payload)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)
    # SSE clients — entry format (for claude-tap)
    if trace_entries:
        sse_payload = json.dumps(trace_entries[-1], ensure_ascii=False, default=str)
        for q in sse_queues:
            try:
                q.put_nowait(sse_payload)
            except asyncio.QueueFull:
                pass


async def _capture_entry(in_body: dict, out_body: dict, res_status: int,
                         res_body: dict, t0: float,
                         sse_events: list[dict] | None = None,
                         upstream_chunks: list[dict] | None = None):
    cap = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "POST",
        "path": "/v1/responses",
        "incoming": {"format": "responses", "body": in_body},
        "outgoing": {"format": "chat_completions",
                     "url": f"{UPSTREAM_BASE}/chat/completions",
                     "body": out_body},
        "response": {"status": res_status, "body": res_body},
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }
    captures.append(cap)

    # Build claude-tap entry format (for SSE + trace file)
    resp_entry_body = dict(res_body)
    if "output_items" in resp_entry_body:
        resp_entry_body["output"] = resp_entry_body.pop("output_items")

    entry = {
        "request_id": cap["id"],
        "timestamp": cap["timestamp"],
        "turn": str(len(captures)),
        "duration_ms": cap["duration_ms"],
        "transport": "http",
        "upstream_base_url": UPSTREAM_BASE,
        "config_name": CONFIG_NAME,
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "body": cap["incoming"]["body"],
        },
        "proxy_outgoing": {
            "url": f"{UPSTREAM_BASE}/chat/completions",
            "body": out_body,
        },
        "response": {
            "status": cap["response"]["status"],
            "body": resp_entry_body,
        },
    }
    if sse_events:
        entry["response"]["sse_events"] = sse_events
        entry["response"]["sse_text"] = "".join(
            _sse(ev["event"], ev["data"]) for ev in sse_events
        )
    if upstream_chunks:
        entry["response"]["upstream_chunks"] = upstream_chunks

    trace_entries.append(entry)
    await broadcast(cap)

    # Persist to .traces/YYYY-MM-DD/trace.jsonl
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trace_dir = TRACE_DIR / today
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "trace.jsonl"

        with trace_lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("Trace write failed", exc_info=True)


# ═══════════════════════════════════════════════════════
#  Proxy routes
# ═══════════════════════════════════════════════════════

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_v1(request: Request, path: str):
    """Main proxy entry point."""
    t0 = time.time()

    # Parse body for write methods
    in_body = {}
    if request.method in ("POST", "PUT", "PATCH"):
        # Read raw bytes first to handle encoding issues
        raw_body = await request.body()
        if raw_body:
            try:
                in_body = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                return JSONResponse(
                    {"error": f"body parse error: {e}"}, status_code=422
                )

    # ── GET /v1/models ──
    if path == "models" and request.method == "GET":
        try:
            resp = await http_client.get(
                f"{UPSTREAM_BASE}/models",
                headers={"Authorization": f"Bearer {UPSTREAM_KEY}"}
            )
            data = resp.json()
            # Rewrite model names back to Codex aliases
            reverse_map = {v: k for k, v in MODEL_MAP.items()}
            model_list = data.get("data") or data.get("models") or []
            for m in model_list:
                if m.get("id") in reverse_map:
                    m["id"] = reverse_map[m["id"]]
            return JSONResponse({
                "object": "list",
                "data": model_list,
                "models": model_list,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)

    # ── POST /v1/responses ──
    if path == "responses":
        logger.info(f"REQ prev_id={in_body.get('previous_response_id','')[:30]} "
                     f"model={in_body.get('model','')} "
                     f"input_items={len(in_body.get('input',[]))} "
                     f"tools={len(in_body.get('tools',[]))} "
                     f"stream={in_body.get('stream')}")
        chat_body = responses_to_chat(in_body)
        logger.info(f"CHAT model={chat_body.get('model')} msgs={len(chat_body.get('messages',[]))} "
                     f"tools={len(chat_body.get('tools',[]))} stream={chat_body.get('stream')}")

        if in_body.get("stream"):
            return StreamingResponse(
                _stream_responses(in_body, chat_body, t0),
                media_type="text/event-stream",
                headers={
                    "x-request-id": str(uuid.uuid4()),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming
        try:
            resp = await http_client.post(
                f"{UPSTREAM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {UPSTREAM_KEY}",
                         "Content-Type": "application/json"},
                json=chat_body,
            )
            chat_data = resp.json()

            if resp.status_code >= 400:
                await _capture_entry(in_body, chat_body, resp.status_code,
                                    chat_data, t0)
                return JSONResponse(chat_data, status_code=resp.status_code)

            choice = chat_data.get("choices", [{}])[0]
            msg = choice.get("message", {})

            # Build output items
            output_items: list[dict] = []
            reasoning = msg.get("reasoning_content")
            text = msg.get("content")

            # Extract <think> tags from content (MiniMax returns thinking in content)
            if not reasoning and text:
                extracted_reasoning, clean_text = _extract_think_tags(text)
                if extracted_reasoning:
                    reasoning = extracted_reasoning
                    text = clean_text

            if reasoning:
                output_items.append({
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:16]}",
                    "summary": [{"type": "summary_text", "text": reasoning}]
                })
            if text:
                output_items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]
                })
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                output_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                })

            # Save session
            assistant_msg = {
                "role": "assistant",
                "content": text if text else None,
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            if msg.get("tool_calls"):
                assistant_msg["tool_calls"] = msg["tool_calls"]
                for tc in msg["tool_calls"]:
                    sessions.store_reasoning(tc.get("id", ""), reasoning)
            sessions.store_turn_reasoning(assistant_msg, reasoning)

            full_history = list(chat_body.get("messages", [])) + [assistant_msg]
            response_id = sessions.save(full_history)

            # Build reasoning echo
            nonstream_reasoning = {}
            req_reasoning_ns = in_body.get("reasoning") or {}
            if isinstance(req_reasoning_ns, dict):
                nonstream_reasoning = {"effort": req_reasoning_ns.get("effort", "medium")}
            nonstream_reasoning["summary"] = "detailed"

            response_obj = {
                "id": response_id,
                "object": "response",
                "model": in_body.get("model", ""),
                "created_at": int(time.time()),
                "status": "completed",
                "reasoning": nonstream_reasoning,
                "output": output_items,
                "usage": _convert_usage(chat_data.get("usage")),
            }

            await _capture_entry(in_body, chat_body, resp.status_code,
                                response_obj, t0)
            return JSONResponse(response_obj)

        except Exception as e:
            await _capture_entry(in_body, chat_body, 502, {"error": str(e)}, t0)
            return JSONResponse({"error": str(e)}, status_code=502)

    # ── Passthrough for other /v1/* paths ──
    url = f"{UPSTREAM_BASE}/{path}"
    headers = {"Authorization": f"Bearer {UPSTREAM_KEY}"}
    try:
        if request.method in ("GET", "DELETE"):
            resp = await http_client.request(request.method, url, headers=headers)
        else:
            resp = await http_client.request(request.method, url, headers=headers, json=in_body)
        body = resp.json() if resp.content else {}
        return JSONResponse(body, status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ── WebSocket for live inspector ──

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    for cap in list(captures):
        try:
            await ws.send_text(json.dumps(cap, ensure_ascii=False, default=str))
        except Exception:
            break
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


# ═══════════════════════════════════════════════════════
#  claude-tap viewer API + SSE
# ═══════════════════════════════════════════════════════

@app.get("/api/dates")
async def api_dates():
    """List available trace dates (claude-tap compatible)."""
    dates = []
    if TRACE_DIR.exists():
        for d in sorted(TRACE_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d / "trace.jsonl").exists():
                dates.append(d.name)
    return JSONResponse({"dates": dates, "has_legacy": False})


@app.get("/api/traces/{date}")
async def api_traces_get(date: str):
    """Get trace entries for a date as JSON array."""
    trace_file = TRACE_DIR / date / "trace.jsonl"
    if not trace_file.exists():
        return Response("not found", status_code=404)
    entries = []
    with trace_lock:
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return JSONResponse(entries)


@app.delete("/api/traces/{date}")
async def api_traces_delete(date: str):
    """Delete trace file for a date."""
    trace_file = TRACE_DIR / date / "trace.jsonl"
    deleted = 0
    with trace_lock:
        if trace_file.exists():
            trace_file.unlink()
            deleted = 1
            # Remove parent dir if empty
            parent = trace_file.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
    return JSONResponse({"deleted_files": deleted})


@app.get("/events")
async def sse_events(request: Request):
    """SSE endpoint for live trace streaming."""
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    sse_queues.append(q)

    async def event_stream():
        try:
            # Replay existing trace entries on connect
            for entry in list(trace_entries):
                payload = json.dumps(entry, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"
            # Stream new captures
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in sse_queues:
                sse_queues.remove(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── JS snippet to render proxy_outgoing (Chat Completions) in claude-tap ──
_PROXY_OUTGOING_JS = r"""
<script>
(function(){
  var _origRD = renderDetail;
  renderDetail = function(e) {
    // Preprocess request input: inject reasoning_content into assistant message content
    // so claude-tap's renderMessages() can render it as a thinking block.
    var input = e && e.request && e.request.body && e.request.body.input;
    if (Array.isArray(input)) {
      input.forEach(function(item) {
        if (item.role === 'assistant' && item.reasoning_content) {
          var rc = item.reasoning_content, tc = {type: 'thinking', thinking: rc};
          if (Array.isArray(item.content)) {
            item.content.unshift(tc);
          } else if (typeof item.content === 'string') {
            item.content = item.content.trim() ? [tc, {type: 'text', text: item.content}] : [tc];
          } else {
            item.content = [tc];
          }
          delete item.reasoning_content;
        }
      });
    }
    _origRD(e);
    var po = e && e.proxy_outgoing;
    if (!po || !po.body) return;
    var body = po.body, d = document.getElementById('detail');
    if (!d) return;
    var html = '';
    var sys = extractSystem(body);
    if (sys) html += section('Proxy Out → System', renderSystemPrompt(sys), false, sys);

    // Pre-inject reasoning_content into content BEFORE getMessages() runs,
    // because getMessages → normalizeChatMessageForDisplay drops reason_content.
    if (Array.isArray(body.messages)) {
      body.messages.forEach(function(m) {
        if (m.role === 'assistant' && m.reasoning_content) {
          var rc = m.reasoning_content, tc = {type: 'thinking', thinking: rc};
          if (Array.isArray(m.content)) {
            m.content.unshift(tc);
          } else if (typeof m.content === 'string') {
            m.content = m.content.trim() ? [tc, {type: 'text', text: m.content}] : [tc];
          } else {
            m.content = [tc];
          }
          delete m.reasoning_content;
        }
      });
    }
    var msgs = getMessages(body);
    if (msgs && msgs.length) html += section('Proxy Out → Messages (Chat API)', renderMessages(msgs), false, null, msgs.length + ' msgs');
    var tools = getRequestTools(body);
    if (tools && tools.length) html += section('Proxy Out → Tools', renderTools(tools), false, null, tools.length + ' tools');
    if (!html) return;
    var sections = d.querySelectorAll('.section');
    var jsonSec = sections[sections.length - 1];
    if (jsonSec) {
      var wrapper = document.createElement('div');
      var proxiedUrl = po.url || '';
      var host = (e.upstream_base_url || '').replace(/^https?:\/\//, '').split('/')[0] || (proxiedUrl.replace(/^https?:\/\//, '').split('/')[0]) || '';
      var fwdLabel = e.config_name || host || 'upstream';
      wrapper.innerHTML = '<div style="margin:12px 0;padding:6px 12px;background:var(--blue-bg);border-radius:6px;font-size:11px;color:var(--blue);font-weight:600;">\u{1f4e4} Forwarded to ' + fwdLabel + ' → ' + proxiedUrl + '</div>' + html;
      bindSections(wrapper);
      while (wrapper.firstChild) jsonSec.parentNode.insertBefore(wrapper.firstChild, jsonSec);
    }
  };
})();
</script>
"""

@app.get("/tap")
async def tap_viewer():
    """Serve claude-tap viewer in live mode."""
    viewer_path = Path("viewer.html")
    if not viewer_path.exists():
        return HTMLResponse("<h1>viewer.html not found</h1>")
    with open(viewer_path, encoding="utf-8") as f:
        html = f.read()
    # LIVE_MODE in head, proxy_outgoing extension before </body>
    html = html.replace("</head>", "<script>var LIVE_MODE=true;</script>\n</head>")
    html = html.replace("</body>", _PROXY_OUTGOING_JS + "\n</body>")
    return HTMLResponse(html)


@app.get("/tap/{date}")
async def tap_viewer_with_trace(date: str):
    """Serve claude-tap viewer with embedded historical trace data."""
    viewer_path = Path("viewer.html")
    if not viewer_path.exists():
        return HTMLResponse("<h1>viewer.html not found</h1>")
    trace_file = TRACE_DIR / date / "trace.jsonl"
    if not trace_file.exists():
        return Response(f"no trace data for {date}", status_code=404)

    with open(viewer_path, encoding="utf-8") as f:
        html = f.read()

    # Parse trace entries as JSON array
    entries = []
    with trace_lock:
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    entries_json = json.dumps(entries, ensure_ascii=False, default=str)
    init_script = (
        f"<script>var EMBEDDED_TRACE_DATA={entries_json};"
        "document.addEventListener('DOMContentLoaded',()=>{"
        "setTimeout(()=>{if(typeof fetchDates==='function')fetchDates();},100);"
        "});</script>"
    )
    html = html.replace("</head>", init_script + "\n</head>")
    html = html.replace("</body>", _PROXY_OUTGOING_JS + "\n</body>")
    return HTMLResponse(html)


# ═══════════════════════════════════════════════════════
#  Inspector UI
# ═══════════════════════════════════════════════════════

INSPECTOR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Codex Proxy — Inspector</title>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;--bdr:#30363d;--bdr2:#21262d;--tx:#c9d1d9;--tx2:#8b949e;--tx3:#484f58;--blu:#58a6ff;--grn:#7ee787;--red:#f85149;--amb:#f0883e;--prp:#bc8cff;--cya:#39d2c0}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,monospace;background:var(--bg);color:var(--tx);display:flex;height:100vh;overflow:hidden}
#sidebar{width:320px;min-width:260px;background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column}
#sidebar h2{padding:14px 16px;font-size:13px;color:var(--blu);border-bottom:1px solid var(--bdr);flex-shrink:0}
#req-list{flex:1;overflow-y:auto}
.req-item{padding:10px 14px;border-bottom:1px solid var(--bdr2);cursor:pointer;font-size:12px;transition:background .12s;border-left:3px solid transparent}
.req-item:hover{background:var(--bg3)}
.req-item.active{background:#1f2a37;border-left-color:var(--blu)}
.req-item .r1{display:flex;align-items:center;gap:6px}
.req-item .method{color:var(--grn);font-weight:700;font-size:11px}
.req-item .path{color:var(--tx2);font-size:11px}
.req-item .status{font-weight:700;font-size:12px}
.req-item .r2{color:var(--tx3);font-size:10px;margin-top:3px}
.req-item .model-badge{font-size:9px;padding:1px 5px;border-radius:3px;background:#1f2a37;color:var(--blu);margin-left:auto}
.s-2xx{color:var(--grn)}.s-4xx{color:var(--red)}.s-5xx{color:var(--amb)}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#tabs{display:flex;border-bottom:1px solid var(--bdr);background:var(--bg2);flex-shrink:0}
#tabs button{padding:9px 16px;background:none;border:none;color:var(--tx2);cursor:pointer;font-size:12px;border-bottom:2px solid transparent;white-space:nowrap}
#tabs button.active{color:#f0f6fc;border-bottom-color:#f78166}
#tabs button:hover{color:var(--tx)}
.panel{display:none;flex:1;overflow-y:auto;padding:14px}
.panel.active{display:block}
/* Dual JSON view */
#dual-view{display:flex;gap:10px;height:100%}
#dual-view .col{flex:1;display:flex;flex-direction:column;overflow:hidden}
.col-label{font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:5px 10px;border-radius:4px 4px 0 0;font-weight:700}
.col-label.responses{background:#1f2a37;color:var(--blu)}.col-label.chat{background:#2a1f1f;color:#f78166}
.col-body{flex:1;overflow-y:auto;background:var(--bg2);border:1px solid var(--bdr);border-radius:0 0 6px 6px;padding:10px;font-size:11px}
/* JSON syntax */
.json-block{background:var(--bg2);border:1px solid var(--bdr);border-radius:6px;padding:12px;overflow:auto;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all;max-height:75vh}
.jk{color:#79c0ff}.js{color:#a5d6ff}.jn{color:#79c0ff}
/* Messages view */
.msg-wrap{display:flex;flex-direction:column;gap:8px}
.msg{border-radius:8px;padding:10px 14px;font-size:13px;line-height:1.55;border:1px solid var(--bdr2);max-width:85%}
.msg.user{align-self:flex-end;background:#0d2240;border-color:#1f3a5f}
.msg.assistant{align-self:flex-start;background:#0d2d1a;border-color:#1a3a26}
.msg.system{align-self:flex-start;background:#2d2200;border-color:#3d3000}
.msg.tool{align-self:flex-start;background:#1f1338;border-color:#2a1f48}
.msg-role{display:inline-block;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;padding:2px 7px;border-radius:3px;margin-bottom:6px}
.msg.user .msg-role{background:var(--blu);color:#fff}
.msg.assistant .msg-role{background:var(--grn);color:#fff}
.msg.system .msg-role{background:#d29922;color:#fff}
.msg.tool .msg-role{background:var(--prp);color:#fff}
.msg .content{white-space:pre-wrap;word-break:break-word}
.msg .tool-use{font-size:10px;color:var(--cya);background:#0a2e2c;padding:6px 10px;border-radius:4px;margin-top:4px;font-family:monospace}
/* Token bar */
.token-bar{display:flex;gap:14px;font-size:11px;padding:8px 0;flex-wrap:wrap}
.tok-item{display:flex;align-items:center;gap:5px}
.tok-dot{width:7px;height:7px;border-radius:50%}
.tok-label{color:var(--tx2)}.tok-val{font-weight:600;font-family:monospace}
/* Status bar */
#status-bar{background:var(--bg2);border-top:1px solid var(--bdr);padding:5px 14px;font-size:11px;color:var(--tx3);display:flex;gap:14px;flex-shrink:0;align-items:center}
.empty-state{color:var(--tx3);text-align:center;padding:60px 20px}
.info-row{font-size:11px;color:var(--tx2);margin-bottom:12px;display:flex;gap:16px;flex-wrap:wrap}
</style>
</head>
<body>
<div id="sidebar">
<h2>Traffic</h2>
<div id="req-list"><div class="empty-state">Waiting...</div></div>
</div>
<div id="main">
<div id="tabs">
<button class="active" data-tab="msg-panel">Messages</button>
<button data-tab="dual">Dual JSON</button>
<button data-tab="in-panel">Request</button>
<button data-tab="out-panel">Chat Body</button>
<button data-tab="res-panel">Response</button>
</div>
<div id="msg-panel" class="panel active">
<div class="info-row" id="msg-info"></div>
<div id="msg-view"></div>
</div>
<div id="dual" class="panel" style="padding:10px">
<div id="dual-view">
<div class="col"><div class="col-label responses">In — Responses API</div><div class="col-body" id="dual-in"></div></div>
<div class="col"><div class="col-label chat">Out — Chat Completions</div><div class="col-body" id="dual-out"></div></div>
</div>
</div>
<div id="in-panel" class="panel"><div class="json-block" id="in-body"></div></div>
<div id="out-panel" class="panel"><div class="json-block" id="out-body"></div></div>
<div id="res-panel" class="panel"><div class="json-block" id="res-body"></div></div>
<div id="status-bar">
<span id="count">0 requests</span><span id="tok-sum"></span><span id="conn-status" style="color:var(--red)">● Disconnected</span>
</div>
</div>
<script>
const ws=new WebSocket(`ws://${location.host}/ws`);
let captures=[],activeId=null;
ws.onopen=()=>document.getElementById('conn-status').innerHTML='<span style="color:var(--grn)">● Live</span>';
ws.onclose=()=>document.getElementById('conn-status').innerHTML='<span style="color:var(--red)">● Disconnected</span>';
ws.onmessage=(e)=>{
  const cap=JSON.parse(e.data);
  const idx=captures.findIndex(c=>c.id===cap.id);
  if(idx>=0) captures[idx]=cap; else captures.push(cap);
  renderList();updateStats();
  if(!activeId||activeId===cap.id){activeId=cap.id;renderDetail(cap);}
};

function updateStats(){
  document.getElementById('count').textContent=captures.length+' requests';
  let totalIn=0,totalOut=0;
  captures.forEach(c=>{
    const u=(c.response.body||{}).usage||(c.response.body||{}).converted_from_chat?.usage;
    if(!u){ const r=(c.response.body||{}).responses_response; if(r&&r.usage) u=r.usage; }
    if(u){totalIn+=u.input_tokens||0;totalOut+=u.output_tokens||0;}
  });
  document.getElementById('tok-sum').textContent=`In: ${totalIn} Out: ${totalOut}`;
}

function renderList(){
  document.getElementById('req-list').innerHTML=captures.slice().reverse().map(c=>{
    const cls=c.response.status<400?'s-2xx':c.response.status<500?'s-4xx':'s-5xx';
    const model=(c.incoming.body||{}).model||'?';
    return `<div class="req-item${c.id===activeId?' active':''}" onclick="select('${c.id}')">
      <div class="r1"><span class="method">${c.method}</span><span class="path">${c.path}</span><span class="status ${cls}">${c.response.status}</span></div>
      <div class="r2">${new Date(c.timestamp).toLocaleTimeString()} · ${c.duration_ms}ms <span class="model-badge">${model}</span></div>
    </div>`;
  }).join('')||'<div class="empty-state">Waiting...</div>';
}
function select(id){activeId=id;const cap=captures.find(c=>c.id===id);if(cap)renderDetail(cap);renderList();}

function renderDetail(c){
  const ib=c.incoming.body, ob=c.outgoing.body, rb=c.response.body;

  // JSON views
  document.getElementById('dual-in').innerHTML=syntaxJson(ib);
  document.getElementById('dual-out').innerHTML=syntaxJson(ob);
  document.getElementById('in-body').innerHTML=syntaxJson(ib);
  document.getElementById('out-body').innerHTML=syntaxJson(ob);
  document.getElementById('res-body').innerHTML=syntaxJson(rb);

  // Info row
  const usage=(rb.usage||(rb.responses_response||{}).usage||(rb.converted_from_chat||{}).usage||{});
  document.getElementById('msg-info').innerHTML=
    `<span>Model: <b style="color:var(--blu)">${ib.model||'?'}</b></span>`+
    `<span>⏱ ${c.duration_ms}ms</span>`+
    `<span style="color:var(--grn)">In: ${usage.input_tokens||0}</span>`+
    `<span style="color:var(--amb)">Out: ${usage.output_tokens||0}</span>`+
    `${usage.total_tokens?`<span>Total: ${usage.total_tokens}</span>`:''}`;

  // Messages view
  renderMessages(ib, rb);
}

function renderMessages(inBody, resBody){
  const msgs=[];
  const input=inBody.input||[];
  const instructions=inBody.instructions||'';

  if(instructions){
    const preview=instructions.length>500?instructions.slice(0,500)+'...':instructions;
    msgs.push({role:'system',content:preview,label:`System (${(instructions.length/1024).toFixed(1)}KB)`});
  }

  for(const item of input){
    if(!item||typeof item!=='object') continue;
    const t=item.type||'';
    if(t==='message'){
      const role=item.role||'user';
      let content=item.content||'';
      if(typeof content!=='string') content=JSON.stringify(content);
      msgs.push({role,content});
    }else if(t==='function_call'){
      msgs.push({role:'assistant',tool:{name:item.name||'?',call_id:item.call_id||'',args:item.arguments||''}});
    }else if(t==='function_call_output'){
      msgs.push({role:'tool',content:item.output||'',call_id:item.call_id||''});
    }
  }

  // Response output
  const output=(resBody.responses_response||resBody).output||resBody.output_items;
  if(Array.isArray(output)){
    for(const item of output){
      if(!item) continue;
      if(item.type==='message'){
        const parts=(item.content||[]).map(p=>p.text||'').join('\n');
        if(parts) msgs.push({role:'assistant',content:parts});
      }else if(item.type==='reasoning'){
        const parts = item.summary || item.content || [];
        const text = parts.map(p=>p.text||'').join('\n');
        if(text) msgs.push({role:'system',content:text,label:'Reasoning'});
      }else if(item.type==='function_call'){
        msgs.push({role:'assistant',tool:{name:item.name||'?',call_id:item.call_id||'',args:item.arguments||''}});
      }
    }
  }

  // Render
  document.getElementById('msg-view').innerHTML=msgs.length
    ? '<div class="msg-wrap">'+msgs.map(m=>{
        const cls=m.role==='user'?'user':m.role==='assistant'?'assistant':m.role==='tool'?'tool':'system';
        let body='';
        if(m.tool){
          body=`<div class="tool-use">🔧 <b>${m.tool.name}</b><br>call_id: ${m.tool.call_id}<br>${m.tool.args}</div>`;
        }else{
          body=`<div class="content">${escHtml(m.content||'')}</div>`;
        }
        const label=m.label||m.role.toUpperCase();
        return `<div class="msg ${cls}"><div class="msg-role">${label}</div>${body}</div>`;
      }).join('')+'</div>'
    : '<div class="empty-state">No messages</div>';
}

function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function syntaxJson(obj){
  return JSON.stringify(obj,null,2).replace(/("(\\[^]|[^"\\])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,m=>{
    let cls=/^"/.test(m)?(/:$/.test(m)?'jk':'js'):'jn';
    return `<span class="${cls}">${m}</span>`;
  });
}

document.querySelectorAll('#tabs button').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('#tabs button').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    const el=document.getElementById(btn.dataset.tab);
    if(el)el.classList.add('active');
  });
});
</script>
</body>
</html>"""


@app.get("/")
async def inspector():
    return HTMLResponse(INSPECTOR_HTML)


# ═══════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════

def main():
    print(f"\n  Config:     {CONFIG_NAME}  ({config_path})")
    print(f"  Upstream:   {UPSTREAM_BASE}")
    print(f"  Models:     {', '.join(f'{k}→{v}' for k, v in MODEL_MAP.items())}")
    print(f"  Proxy + UI: http://127.0.0.1:{PROXY_PORT}")
    print(f"  Inspector:  http://127.0.0.1:{PROXY_PORT}/")
    print(f"  Trace dir:  {TRACE_DIR.resolve()}\n")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
