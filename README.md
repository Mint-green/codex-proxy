# Codex Proxy

OpenAI Responses API ↔ Chat Completions API 协议转换代理，支持任意兼容 OpenAI Chat API 的上游厂商（DeepSeek、MiniMax 等），附带流量可视化面板。

### 为什么需要这个

Codex CLI 使用 OpenAI Responses API 格式，而大部分第三方 LLM 厂商只提供 Chat Completions API。本代理在两者之间做实时协议转换，让 Codex 可以直接使用 DeepSeek、MiniMax 等厂商的模型。同时记录每次请求的完整 trace（请求体、SSE 事件、响应体），支持 `/tap` 页面实时调试。

## 快速开始

```bash
# 安装依赖
pip install fastapi uvicorn httpx

# 复制示例配置并填入你的 API Key
cp configs/deepseek.example.toml configs/deepseek.toml
# 编辑 configs/deepseek.toml，将 api_key 替换为你的实际 Key

# 默认 DeepSeek
python proxy_app.py

# MiniMax
python proxy_app.py minimax

# 同时启动两个厂商（两个终端）
python proxy_app.py            # → :5555 → DeepSeek
python proxy_app.py minimax    # → :5556 → MiniMax

# 自定义配置
python proxy_app.py --config /path/to/my.toml
```

启动后访问：
- `http://127.0.0.1:5555/` — 简易流量面板（最近 50 条请求）
- `http://127.0.0.1:5555/tap` — claude-tap 可视化面板（请求/响应完整解析、SSE 回放、Token 统计）
- `http://127.0.0.1:5555/tap/2026-05-21` — 指定日期的历史记录

## 配置说明

配置文件使用 TOML 格式，放在 `configs/` 目录下。完整字段：

```toml
[server]
port = 5555                     # 代理监听端口

[upstream]
base_url = "https://api.deepseek.com/v1"   # 上游 Chat API 地址
api_key = "sk-..."                          # 上游 API Key
timeout = 120                               # 请求超时（秒）
# reasoning_format = "field"               # 可选，reasoning 注入格式。默认 "field"（separate field）
                                           # "think_tags" = MiniMax 风格（<think> 标签）
# [upstream.extra_params]                  # 可选，额外参数递归合并到 Chat API 请求体
# thinking = {type = "enabled"}            # 例：MiMo 的 Anthropic 风格 thinking 参数

[models]
"gpt-5.5" = "deepseek-v4-pro"   # Codex 模型名 → 上游模型名
"gpt-5.4" = "deepseek-v4-flash"

[reasoning]                     # reasoning_effort 映射（可选）
low    = "high"                 # OpenAI 值 → 上游值
medium = "high"
high   = "high"
xhigh  = "max"

[traces]
dir = "traces-deepseek"         # trace 文件目录（含 SQLite reasoning 库）
```

字段说明：

| 段 | 字段 | 必填 | 说明 |
|----|------|------|------|
| `[server]` | `port` | 是 | 代理监听端口 |
| `[upstream]` | `base_url` | 是 | 上游 Chat Completions API 地址（需含 `/v1`） |
| `[upstream]` | `api_key` | 是 | 上游 API Key |
| `[upstream]` | `timeout` | 否 | HTTP 超时秒数，默认 120 |
| `[upstream]` | `reasoning_format` | 否 | reasoning 注入格式：`"field"`（默认，separate field，DeepSeek 适用）或 `"think_tags"`（`<think>` 标签，MiniMax 适用） |
| `[upstream.extra_params]` | 键值对 | 否 | 额外参数，递归深度合并到 Chat API 请求体根部。对象递归合并，基本类型/数组覆盖。例：`thinking = {type = "enabled"}`（MiMo 的 Anthropic 风格 thinking 参数） |
| `[models]` | 键值对 | 是 | Codex 模型名 → 上游模型名映射 |
| `[reasoning]` | 键值对 | 否 | reasoning_effort 值映射。不配置则透传；配置了但某档位不映射则丢弃 |
| `[traces]` | `dir` | 是 | Trace 文件 & SQLite 库的存储目录 |

### 添加新厂商

```bash
cp configs/deepseek.example.toml configs/my_provider.toml
# 编辑 my_provider.toml 后：
python proxy_app.py my_provider
```

模型映射的含义：Codex 只认识 `gpt-5.5` / `gpt-5.4` 等自家模型，代理按 `[models]` 表将这两个名映射到上游实际模型名。上游模型必须支持标准 Chat Completions API。

## 协议转换

### 请求转换：Responses → Chat

Codex 发来的 Responses API 请求包含 `input` 数组（items），其中有 `message`、`function_call`、`function_call_output` 等类型的 item。代理将其转换为 Chat API 的 `messages` 数组：

| Responses API item | Chat API message |
|---|---|
| `{type: "message", role: "user"}` | `{role: "user", content: "..."}` |
| `{type: "message", role: "developer"}` | `{role: "system", content: "..."}` |
| `{type: "function_call", call_id, name, arguments}` | `{role: "assistant", content: null, tool_calls: [...]}` |
| `{type: "function_call_output", call_id, output}` | `{role: "tool", tool_call_id, content: "..."}` |
| `{type: "reasoning"}` | 丢弃（reasoning 由 session store 恢复） |

连续多个 `function_call` 会被合并到同一个 `assistant` 消息的 `tool_calls` 数组里。

### 响应转换：Chat → Responses

上游返回 Chat API 响应后，代理将其转为 Responses API 的事件流：

- Chat `choices[0].delta.content` → `response.output_text.delta`
- Chat `choices[0].delta.reasoning_content`（DeepSeek）→ `response.output_item.added (type: reasoning, summary: [...])`（流式）
- Chat `choices[0].delta.content` 中 `<think>` 标签（MiniMax）→ 提取后作为 `response.output_item.added (type: reasoning)` 单独发出，剩余纯文本作为 message 内容
- Chat `choices[0].delta.tool_calls` → `response.output_item.added (type: function_call)` + `response.function_call_arguments.delta` + `response.output_item.done`
- 非流式同理，在 `output` 数组中放置 `reasoning`、`message`、`function_call` 项

### Thinking / Reasoning 处理

**背景：** 不同厂商返回思维链的方式不同：DeepSeek 通过独立的 `reasoning_content` 字段返回，MiniMax 则把思维包在 `<think>...</think>` 标签里随 `content` 返回。对于发生过工具调用的轮次，DeepSeek 要求后续请求中将 `reasoning_content` 回传 API，MiniMax 则需把思维链作为 `<think>` 标签注入 assistant 消息的 content 开头。

**本代理的处理：**

**提取阶段**（响应返回时）：
- **DeepSeek**：`reasoning_content` 字段直接累加为 `accumulated_reasoning`
- **MiniMax**：流结束后调用 `_extract_think_tags()` 从 `accumulated_text` 中分离 `<think>` 标签内容和纯文本

**存储与恢复**（下次请求时）：

统一的 key-value 存储（内存 dict + SQLite），key 前缀区分索引方式：

```
写入：响应返回 → store_reasoning(call_id, reasoning_text)
                ├── _reasoning["call:{call_id}"] = text     （内存）
                ├── _reasoning["hash:{md5}"] = text         （content 指纹，内存）
                ├── ReasoningDB.save("call:{call_id}", text) （SQLite 持久化）
                └── ReasoningDB.save("hash:{md5}", text)     （SQLite 持久化）

查询：构建 Chat 消息时，两种命中路径：
  1. 按 call_id 查：get_reasoning(call_id)
     ├── _reasoning["call:{call_id}"]       → 命中返回
     └── ReasoningDB.get("call:{call_id}")   → 命中回写内存，返回
                                              → 未命中 → 路径 2
  2. 按 content 指纹查（Codex 全量回放 input 项时，无 call_id）：
     ├── _reasoning["hash:{md5(content)}"]   → 命中返回
     └── ReasoningDB.get("hash:{md5}")        → 命中回写内存，返回
                                              → 未命中 → None
```

> 指纹使用 MD5（跨进程稳定），而非 Python `hash()`（进程内随机化）。

恢复的 reasoning 通过 `_apply_reasoning()` 按 `reasoning_format` 配置注入：
- `"field"`（默认）→ 写入 assistant 消息的 `reasoning_content` 字段（DeepSeek）
- `"think_tags"` → 拼入 assistant 消息的 `content` 开头：`<think>...</think>\n\n{content}`（MiniMax）

### reasoning_effort 映射

OpenAI Responses API 的 `reasoning_effort` 值（low/medium/high/xhigh）与各厂商的 effort 值定义不同。配置文件的 `[reasoning]` 段做映射：

- **DeepSeek**：只支持 `high` 和 `max`，因此 low/medium/high 都映射为 `high`，xhigh 映射为 `max`
- **MiniMax**：标准 Chat API 不支持 `reasoning_effort` 参数，无论是否配置 `[reasoning]` 段，该参数都不会传给上游（会被丢弃）。如需使用 MiniMax 的思考能力，应确保上游模型本身就开启了思考模式

## 组件与数据流

### 组件来源

| 组件 | 文件 | 来源 |
|------|------|------|
| 协议转换核心 | `proxy_app.py` | 参考 [codex-relay](https://github.com/MetaFARS/codex-relay)（Rust）的 Python 移植 |
| 流量可视化面板 | `viewer.html` | 基于 [claude-tap](https://github.com/liaohch3/claude-tap) v0.1.74 修改，详见下方 [viewer 修改说明](#viewer-修改说明-vs-claude-tap-原始版本) |

`proxy_app.py` 是一个单文件应用，包含：协议转换、SessionStore、ReasoningDB、Trace 记录、简易流量面板（`/`）、claude-tap 面板（`/tap`）所需的所有后端接口。

### 数据流

```
Codex CLI ──Responses API──→ proxy_app.py
                                 │
                    ┌────────────┤
                    │  to_chat_request()
                    │  1. load history (prev_id)
                    │  2. process input items
                    │  3. inject reasoning via _apply_reasoning()
                    │     (session store / SQLite, fmt-aware)
                    │  4. build Chat API body
                    │     (w/ stream_options if streaming)
                    └────────────┤
                                 │
                    ┌────────────┤
                    │  httpx → upstream
                    │  (DeepSeek / MiniMax / ...)
                    └────────────┤
                                 │
                    ┌────────────┤
                    │  from_chat_response()
                    │  1. convert deltas → SSE events
                    │  2. extract reasoning (rc field or
                    │     <think> tags by _extract_think_tags)
                    │  3. store reasoning (session + SQLite)
                    │  4. save full history by response_id
                    │  5. write trace entry (JSONL)
                    └────────────┤
                                 │
              ┌──────────────────┤
              │  /tap viewer     │
              │  读取 trace JSONL│
              │  JS 预处理 rc    │
              │  渲染 thinking   │
              └──────────────────┘
```

### SessionStore（内存）

参考 codex-relay 的 `session.rs`。API：`store_reasoning` / `get_reasoning` / `store_turn_reasoning` / `get_turn_reasoning` / `get_history` / `save_with_id` / `save`。

统一的 `_reasoning` dict，key 带前缀区分来源：
- **`call:{call_id}`** — 按 tool call ID 精确查找
- **`hash:{md5}`** — 按 assistant 消息 content 的 MD5 指纹查找。用于 Codex 不使用 `previous_response_id` 而是全量回放 `input` 项时的 reasoning 恢复

内存 miss 时自动走 SQLite 回写，重启后指纹索引可从 SQLite 恢复。

### ReasoningDB（SQLite 持久化）

启动时自动在 `[traces].dir/reasoning.db` 创建。表结构：

```sql
CREATE TABLE reasoning (
    key        TEXT PRIMARY KEY,
    reasoning  TEXT NOT NULL,
    created_at TEXT
);
```

key 前缀区分索引方式：`call:{call_id}`（工具调用索引）、`hash:{md5_hex}`（content 指纹索引，MD5 确保跨进程稳定）。两种索引同一张表、同一个内存 dict。

- `INSERT OR REPLACE`，同一 key 幂等写入
- 启动时自动迁移旧表：`call_id` → `call:{call_id}`
- 只在内存 miss 时才走 SQLite，查到后回写内存

### Trace 记录

每次请求自动记录到 `[traces].dir/YYYY-MM-DD/trace.jsonl`。每行一个 JSON 对象：

```json
{
  "call_id": "req_...",
  "timestamp": "2026-05-21T09:33:28+00:00",
  "request":  {"body": {...}},
  "response": {"body": {...}, "sse_events": [...]},
  "proxy_outgoing": {"body": {...}},
  "status": "ok"
}
```

线程安全写入（`threading.Lock`），不同 config 使用不同 `traces.dir`，互不干扰。

### viewer 修改说明（vs claude-tap 原始版本）

基于 [claude-tap](https://github.com/liaohch3/claude-tap) **v0.1.74** 的 `viewer.html` 做了以下修改：

| 修改 | 说明 |
|------|------|
| i18n 注入 | 原始 i18n 依赖构建时注入 `__CLAUDE_TAP_I18N__` 变量，副本缺失。已直接将完整翻译字典（en / zh-CN）写入页面 |
| 新增 Token（New Input） | 顶部统计栏 + 侧边栏条目 + 单轮详情页均新增"新增输入"展示（=`input_tokens - cache_read_input_tokens`），靛蓝色标识 |
| 语言选择器精简 | 原始支持 8 种语言，本副本仅保留 en / zh-CN 两个选项 |
| Forwarded to 动态化 | 原始写死 "Forwarded to DeepSeek"，改为从 URL 提取 hostname 及从 trace 提取 `upstream_base_url` 动态显示 |

核心交互逻辑、SSE 回放、diff 对比等功能未做改动。

## Codex 接入

要使 Codex CLI 通过本代理调用上游模型，在 `.codex/config.toml` 中添加 provider：

```toml
[provider.deepseek]
base_url = "http://127.0.0.1:5555/v1"
api_key  = "any"
```

`api_key` 可以填任意值——真正的鉴权在本代理的配置文件中完成。

## 依赖与安装

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.11+ | `tomllib` 内置模块 |
| fastapi | — | Web 框架 |
| uvicorn | — | ASGI 服务器 |
| httpx | — | 异步上游 HTTP 请求 |
| sqlite3 | Python 标准库（内置） | reasoning 持久化，无需额外安装 |

```bash
pip install fastapi uvicorn httpx
```

首次启动时自动创建：
- `[traces].dir/` — trace 目录
- `[traces].dir/reasoning.db` — SQLite reasoning 库
- `logs/proxy_<config>.log` — 运行日志（1MB×3 自动轮转）

## 项目结构

```
codex-proxy/
├── proxy_app.py              # 主程序（协议转换 + SessionStore + ReasoningDB + 调试面板）
├── viewer.html               # claude-tap 可视化前端（完整副本）
├── configs/
│   ├── deepseek.example.toml # DeepSeek 配置示例
│   ├── minimax.example.toml  # MiniMax 配置示例
│   └── mimo.example.toml     # MiMo 配置示例
├── traces-deepseek/          # DeepSeek trace 目录（自动创建）
│   ├── YYYY-MM-DD/
│   │   └── trace.jsonl       # 按日记录的 trace 文件
│   └── reasoning.db          # SQLite reasoning 持久化库
├── traces-minimax/           # MiniMax trace 目录（自动创建，同上结构）
├── logs/                     # 运行日志（自动创建，1MB×3 轮转）
│   ├── proxy_deepseek.log
│   └── proxy_minimax.log
└── README.md
```

## 更新记录

### v3 (2026-05-23)

- **MiMo（小米）对接**：`thinking` 参数适配（Anthropic 风格 `{type: "enabled"}`），reasoning 格式兼容 DeepSeek 的 `reasoning_content` 字段
- **`[upstream.extra_params]` 配置**：支持任意额外参数递归深度合并到 Chat API 请求体，适配各厂商非标扩展（如 MiMo 的 `thinking`）

### v2 (2026-05-23)

- **MiniMax 对接**：支持 `<think>` 标签式 reasoning（`_extract_think_tags` / `_apply_reasoning`），流结束自动兜底（无需 `[DONE]`），`stream_options` 获取 token 消耗
- **`reasoning_format` 配置**：`[upstream]` 下新增字段，`"field"`（DeepSeek 默认）或 `"think_tags"`（MiniMax）
- **Reasoning 存储统一重构**：内存 dict + SQLite 合并为单一 key-value 存储，key 前缀 `call:` / `hash:`，指纹改为 MD5（跨进程稳定），两种索引均持久化
- **viewer 优化**：Forwarded to 动态显示实际厂商名和 base_url

### v1 (2026-05-21)

- 首次提交，DeepSeek 协议转换，claude-tap 可视化面板
