#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_proxy.py — 图片桥接中转代理
===================================
给不支持视觉的 LLM 接上图片理解能力：

无图请求 → 原样透传给主模型（支持流式 SSE）
有图请求 → 先调用视觉模型分析图片（按用户问题结构化提取文字/报错/界面信息），
          把图片替换为文字分析结果，再转发给主模型

支持两种协议端点：
- OpenAI 兼容：POST /v1/chat/completions（WorkBuddy / Codex / OpenAI SDK / Chat UI）
- Anthropic 兼容：POST /v1/messages（Claude Code / OpenClaw 等）

监听：0.0.0.0:8888
配置：hybrid_proxy.json（含两个上游的 base_url / api_key / model）
运行：python3 hybrid_proxy.py
"""

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlreq
from urllib.error import HTTPError, URLError


def _find_config():
    """配置查找顺序：环境变量 > 脚本同目录 hybrid_proxy.json > /root/hybrid_proxy.json（兜底）"""
    env = os.environ.get("HYBRID_PROXY_CONFIG")
    if env:
        return env
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_proxy.json")
    if os.path.exists(local):
        return local
    return "/root/hybrid_proxy.json"


CONFIG_PATH = _find_config()
LOG_PATH = os.environ.get("HYBRID_PROXY_LOG", "/var/log/hybrid_proxy.log")


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CFG = load_config()
PORT = int(CFG.get("listen_port", 8888))
KIMI_URL = CFG["kimi_base_url"].rstrip("/") + "/chat/completions"
KIMI_KEY = CFG["kimi_api_key"]
KIMI_MODEL = CFG["kimi_model"]
KIMI_TIMEOUT = int(CFG.get("kimi_timeout", 180))
DEEPSEEK_URL = CFG["deepseek_url"]
DEEPSEEK_KEY = CFG["deepseek_api_key"]
DEEPSEEK_TIMEOUT = int(CFG.get("deepseek_timeout", 300))
DEEPSEEK_MODEL = CFG.get("deepseek_model", "deepseek-v4-pro")


def _json_call(url, api_key, payload, timeout, retries=3):
    """POST JSON，返回 (status, headers_dict, body_bytes)。错误时 status=0 且 body 为错误描述。"""
    data = json.dumps(payload).encode("utf-8")
    last = (0, {}, b"")
    for attempt in range(1, retries + 1):
        req = urlreq.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
        })
        try:
            with urlreq.urlopen(req, timeout=timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except Exception as e:  # noqa: BLE001
            last = (0, {}, ("%s: %s" % (type(e).__name__, e)).encode("utf-8", "ignore"))
            if attempt < retries:
                time.sleep(2 * attempt)
    return last


def has_image(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "image_url":
                    return True
    return False


def last_user_text(messages):
    """取最后一条含文本的 user 消息作为『用户问题』。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()[:2000]
        if isinstance(c, list):
            texts = [p.get("text", "") for p in c if p.get("type") == "text"]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined[:2000]
    return ""


def _analyze_single_image(url, user_question):
    """对单张图调 kimi 分析，返回 (analysis_text, error)。error 非空表示失败。"""
    prompt = (
        "请分析这张图片，回答用户的问题。要求："
        "1) 直接输出最终结论，不要输出任何思考过程/推理步骤；"
        "2) 完整提取图中所有可见文字（逐字 OCR，特别注意报错信息、数字、域名、按钮文字、金额）；"
        "3) 说明这是什么页面/界面/类型；"
        "4) 如果用户问了具体问题，直接针对问题给出答案和依据。\n"
        "用户问题：" + (user_question or "（无，请概括图片内容）")
    )
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": url}}]
    kimi_payload = {
        "model": KIMI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 1,
        "max_tokens": 2048,
        "stream": False,
    }
    t0 = time.time()
    status, _, body = _json_call(KIMI_URL, KIMI_KEY, kimi_payload, KIMI_TIMEOUT)
    cost = round(time.time() - t0, 1)
    if status != 200:
        err = body.decode("utf-8", "ignore")[:300]
        log("kimi FAIL status=%s cost=%ss err=%s" % (status, cost, err))
        return "", "kimi 视觉分析失败 HTTP %s: %s" % (status, err)

    try:
        resp = json.loads(body)
        msg = resp["choices"][0]["message"]
        # kimi-k2.6 是推理模型：最终答案在 content，思考过程在 reasoning_content。
        # 若 content 为空（可能被 max_tokens 截断），退而取 reasoning_content 兜底。
        analysis = msg.get("content") or ""
        if not analysis.strip():
            analysis = msg.get("reasoning_content") or ""
        if not analysis.strip():
            analysis = "[kimi 未返回文本内容]"
    except Exception as e:  # noqa: BLE001
        log("kimi parse FAIL: %s body=%s" % (e, body[:200]))
        return "", "kimi 响应解析失败: %s" % e

    log("kimi OK status=%s cost=%ss" % (status, cost))
    return analysis, None


def analyze_images_with_kimi(messages):
    """
    逐图独立交给视觉模型分析（每张图都完整提取文字），
    把消息里的 image_url 部分替换为对应的文字分析结果。
    返回 (new_messages, error)。error 非空表示整批失败。
    """
    new_messages = json.loads(json.dumps(messages))
    user_question = last_user_text(new_messages)

    img_count = 0
    first_err = None
    for m in new_messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        new_parts = []
        for p in c:
            if p.get("type") != "image_url":
                new_parts.append(p)
                continue
            u = p.get("image_url")
            url = u.get("url", "") if isinstance(u, dict) else (u or "")
            if not url:
                new_parts.append(p)
                continue
            img_count += 1
            analysis, err = _analyze_single_image(url, user_question)
            if err:
                if first_err is None:
                    first_err = err
                # 失败时用错误文本替换图片（避免把图原样发给不支持视觉的主模型）
                new_parts.append({"type": "text", "text": "[第 %d 张图片分析失败: %s]" % (img_count, err)})
            else:
                new_parts.append({"type": "text",
                                  "text": "[第 %d 张图片（已由视觉模型 %s 分析）]\n%s" % (img_count, KIMI_MODEL, analysis)})
        m["content"] = new_parts

    if img_count == 0:
        return new_messages, None
    # 只要有一张图分析成功就继续；全部失败才返回错误
    if first_err is not None and img_count == 1:
        return new_messages, first_err
    return new_messages, None


# ---------------------------------------------------------------------------
# Anthropic ↔ OpenAI 格式转换
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(anthropic_messages, system=None):
    """
    把 Anthropic Messages 格式转成 OpenAI Chat 格式。
    - image source base64 → data URI（image_url）
    - image source url → image_url
    - 顶层 system（str 或 text block 列表）→ role=system 的 OpenAI 消息
    """
    out = []
    if system:
        if isinstance(system, str):
            out.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
            joined = "".join(t for t in texts if t).strip()
            if joined:
                out.append({"role": "system", "content": joined})

    for m in anthropic_messages:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": c})
            continue
        if not isinstance(c, list):
            continue
        parts = []
        for p in c:
            t = p.get("type") if isinstance(p, dict) else None
            if t == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif t == "image":
                src = p.get("source") or {}
                if src.get("type") == "base64":
                    media = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    parts.append({"type": "image_url",
                                  "image_url": {"url": "data:%s;base64,%s" % (media, data)}})
                elif src.get("type") == "url":
                    parts.append({"type": "image_url",
                                  "image_url": {"url": src.get("url", "")}})
            else:
                # 其他类型（tool_use 等）原样保留
                parts.append(p)
        out.append({"role": role, "content": parts})
    return out


def openai_text_to_anthropic_message(text, model, input_tokens=0, output_tokens=0):
    """把 OpenAI 的文本回复包成 Anthropic 非流式 message 响应。"""
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def extract_openai_text(obj):
    """从 OpenAI 非流式响应里提取正文文本（content 可能是 str 或 list）。"""
    try:
        msg = obj["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "".join(t for t in texts if t)
    return ""


class AnthropicStreamConverter:
    """把 OpenAI 流式 SSE 逐字节转成 Anthropic 流式 SSE 事件。"""

    def __init__(self, model):
        self.model = model
        self.msg_id = "msg_" + uuid.uuid4().hex
        self.buf = b""
        self.started = False
        self.block_started = False
        self.finished = False
        self.out_tokens = 0

    def _sse(self, event, obj):
        return ("event: %s\ndata: %s\n\n" % (event, json.dumps(obj, ensure_ascii=False))).encode("utf-8")

    def _start(self):
        out = self._sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.msg_id, "type": "message", "role": "assistant",
                "model": self.model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        out += self._sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        self.started = True
        self.block_started = True
        return out

    def close(self):
        """收尾：补齐 content_block_stop / message_delta / message_stop。"""
        if self.finished:
            return b""
        self.finished = True
        out = b""
        if not self.started:
            # 没有任何内容：也发一套空的 message_start 保证客户端拿到完整事件
            out += self._start()
        if self.block_started:
            out += self._sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        out += self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": self.out_tokens},
        })
        out += self._sse("message_stop", {"type": "message_stop"})
        return out

    def feed(self, chunk):
        """输入 OpenAI SSE 原始字节，返回可写出的 Anthropic SSE 字节。"""
        if self.finished:
            return b""
        self.buf += chunk
        out = b""
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                out += self.close()
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            try:
                delta = obj["choices"][0]["delta"]
                finish_reason = obj["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError):
                continue
            if not self.started:
                out += self._start()
            content = delta.get("content")
            if content:
                self.out_tokens += len(content)
                out += self._sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                })
            if finish_reason:
                out += self.close()
        return out


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(b"hybrid_proxy ok")))
        self.end_headers()
        self.wfile.write(b"hybrid_proxy ok")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/v1/messages"):
            self._handle_anthropic()
        else:
            self._handle_openai()

    # ---- 共享核心：有图分析 → 得到最终转发 payload ----

    def _prepare_payload(self, payload):
        """有图则先做视觉分析替换；强制主模型名。返回 (final_payload, error)。"""
        messages = payload.get("messages") or []
        if has_image(messages):
            new_msgs, err = analyze_images_with_kimi(messages)
            if err:
                return None, err
            payload["messages"] = new_msgs
        payload["model"] = DEEPSEEK_MODEL
        return payload, None

    def _stream_forward(self, payload, on_chunk, on_end=None):
        """
        流式转发到主模型。on_chunk(raw_bytes) 对每个读到的块做处理并返回要写出的字节；
        on_end()（可选）在流结束后调用，返回收尾字节（如 Anthropic 的 message_stop）。
        返回 True 表示成功（已写响应），否则已通过异常分支处理。
        """
        payload["stream"] = True
        data = json.dumps(payload).encode("utf-8")
        req = urlreq.Request(DEEPSEEK_URL, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + DEEPSEEK_KEY,
            "Accept": self.headers.get("Accept", "application/json"),
        })
        try:
            resp = urlreq.urlopen(req, timeout=DEEPSEEK_TIMEOUT)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                out = on_chunk(chunk)
                if out:
                    self.wfile.write(out)
                    self.wfile.flush()
            if on_end:
                tail = on_end()
                if tail:
                    self.wfile.write(tail)
                    self.wfile.flush()
            return True
        except HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False
        except Exception as e:  # noqa: BLE001
            self._respond_json(502, {"error": "upstream stream error: %s" % e})
            return False

    # ---- OpenAI 端点 ----

    def _handle_openai(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            self._respond_json(400, {"error": "invalid json body"})
            return

        t0 = time.time()
        final, err = self._prepare_payload(payload)
        if err:
            self._respond_json(502, {"error": err})
            return

        want_stream = bool(final.get("stream"))
        if want_stream:
            log("REQ openai stream start")
            self._stream_forward(final, lambda chunk: chunk)
            log("REQ openai stream done total=%ss" % round(time.time() - t0, 1))
            return

        status, hdrs, body = _json_call(DEEPSEEK_URL, DEEPSEEK_KEY, final, DEEPSEEK_TIMEOUT)
        total = round(time.time() - t0, 1)
        log("REQ passthrough status=%s total=%ss" % (status, total))
        ct = hdrs.get("Content-Type", "application/json")
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- Anthropic 端点 ----

    def _handle_anthropic(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw)
        except Exception:  # noqa: BLE001
            self._respond_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid json body"}})
            return

        # Anthropic 请求结构：{model, max_tokens, system?, messages, stream?}
        anthropic_messages = req.get("messages") or []
        system = req.get("system")
        openai_messages = anthropic_to_openai_messages(anthropic_messages, system)

        openai_payload = {
            "model": DEEPSEEK_MODEL,
            "messages": openai_messages,
            "max_tokens": req.get("max_tokens", 2048),
            "stream": bool(req.get("stream")),
        }
        # 透传温度等可选参数
        for k in ("temperature", "top_p"):
            if k in req:
                openai_payload[k] = req[k]

        t0 = time.time()
        final, err = self._prepare_payload(openai_payload)
        if err:
            self._respond_json(502, {"type": "error", "error": {"type": "api_error", "message": err}})
            return

        want_stream = bool(final.get("stream"))
        if want_stream:
            conv = AnthropicStreamConverter(DEEPSEEK_MODEL)
            self._stream_forward(final, conv.feed, conv.close)
            log("REQ anthropic stream done total=%ss" % round(time.time() - t0, 1))
            return

        status, hdrs, body = _json_call(DEEPSEEK_URL, DEEPSEEK_KEY, final, DEEPSEEK_TIMEOUT)
        total = round(time.time() - t0, 1)
        log("REQ anthropic status=%s total=%ss" % (status, total))
        if status != 200:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        try:
            openai_resp = json.loads(body)
            text = extract_openai_text(openai_resp)
            usage = openai_resp.get("usage") or {}
            anth = openai_text_to_anthropic_message(
                text, DEEPSEEK_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            anth_body = json.dumps(anth, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(anth_body)))
            self.end_headers()
            self.wfile.write(anth_body)
        except Exception as e:  # noqa: BLE001
            log("anthropic parse FAIL: %s" % e)
            self._respond_json(502, {"type": "error", "error": {"type": "api_error", "message": "response parse failed: %s" % e}})

    def _respond_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log("hybrid_proxy listening on 0.0.0.0:%d  kimi=%s  deepseek=%s" % (PORT, KIMI_MODEL, DEEPSEEK_URL))
    srv.serve_forever()


if __name__ == "__main__":
    main()
