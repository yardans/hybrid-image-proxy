# hybrid-image-proxy 🖼️→📝

**给不支持视觉的 LLM 接上图片理解能力** 的 OpenAI 兼容中转代理（单文件、零依赖、可跑在 OpenWrt 软路由/树莓派/任何有 Python3 的机器上）。

```
无图请求  → 原样透传给主模型（支持流式 SSE）
有图请求  → 先调用视觉模型分析图片（逐字 OCR 提取文字/报错/界面信息），
           把图片替换为文字分析结果，再转发给主模型
```

典型场景：你的主力模型不支持视觉（比如某些文本模型 / 内部网关），但你又想让它在对话里"看懂"截图、报错、UI 界面。这个代理在中间做一次桥接——图由视觉模型消化成文字，文字再交给你的主力模型。

## ✨ 特性

- **OpenAI 兼容**：`POST /v1/chat/completions`，现有客户端（OpenAI SDK、curl、各种 Chat UI）零改动接入
- **流式透传**：无图请求原样转发 SSE，首字延迟不受影响
- **多图支持**：有图请求逐图交给视觉模型分析，每张图独立完整提取
- **推理模型兜底**：视觉模型只输出 `content` 不输出时，自动退回 `reasoning_content`
- **单文件零依赖**：只用 Python3 标准库，无 pip 依赖
- **Key 不落盘**：API Key 放独立配置文件，不入代码、不入 git

## 🚀 快速开始

### 1. 准备配置

```bash
cp hybrid_proxy.json.example hybrid_proxy.json
vi hybrid_proxy.json
```

```json
{
  "listen_port": 8888,
  "kimi_base_url": "https://api.moonshot.cn/v1",
  "kimi_api_key": "你的视觉模型 API Key",
  "kimi_model": "kimi-k2.6",
  "kimi_timeout": 180,
  "deepseek_url": "https://你的主模型网关/v1/chat/completions",
  "deepseek_api_key": "你的主模型 API Key",
  "deepseek_model": "deepseek-v4-pro",
  "deepseek_timeout": 300
}
```

| 字段 | 说明 |
|---|---|
| `listen_port` | 监听端口，默认 8888 |
| `kimi_*` | 视觉模型（用于分析图片），需支持 OpenAI 兼容图片输入 |
| `deepseek_*` | 主模型（最终生成回复），可为任意 OpenAI 兼容接口 |
| `deepseek_model` | 主模型名；若你的网关不支持该模型名，可在此改写 |

### 2. 启动

```bash
python3 hybrid_proxy.py
# 配置文件默认读 /root/hybrid_proxy.json，
# 可用环境变量覆盖：HYBRID_PROXY_CONFIG=/path/to/hybrid_proxy.json HYBRID_PROXY_LOG=/path/to/log python3 hybrid_proxy.py
```

### 3. 调用

```bash
# 无图（纯文本，流式）
curl -N http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"any","stream":true,"messages":[{"role":"user","content":"你好"}]}'

# 有图（视觉模型先分析，再交给主模型）
curl -N http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "stream": true,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "这个报错怎么解决？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/screenshot.png"}}
      ]
    }]
  }'
```

返回内容和 OpenAI 完全一致，客户端无需任何改动。

## 🖥️ 部署到 OpenWrt 软路由

`openwrt/` 目录提供 procd 自启脚本：

```bash
cp openwrt/hybrid_proxy.init /etc/init.d/hybrid_proxy
chmod +x /etc/init.d/hybrid_proxy
/etc/init.d/hybrid_proxy enable
/etc/init.d/hybrid_proxy start
```

> ⚠️ OpenWrt 的 BusyBox **没有 `nohup` 命令**，别在启动命令里写 `nohup`，直接后台运行即可（脚本已处理）。

## 📝 日志

每行一条，含各阶段耗时，方便定位慢在哪一环：

```
[2026-08-17 08:21:02] kimi OK status=200 cost=38.2s
[2026-08-17 08:21:04] REQ image→deepseek stream status=200 total=40.0s
[2026-08-17 08:14:48] REQ passthrough status=200 total=1.1s
```

- `kimi OK cost=...`：视觉模型分析耗时
- `REQ passthrough`：无图请求总耗时
- `REQ image→deepseek`：有图请求总耗时（= kimi 耗时 + 主模型耗时）

## ⚡ 性能提示

- **视觉模型选型直接决定有图请求速度**：推理型视觉模型（如 kimi-k2.6）单图分析约 30~45 秒；换非推理型视觉模型（如 gpt-4o-mini 类）通常 5~10 秒
- 有图请求是**串行**的：先等视觉模型分析完，再等主模型生成，两段耗时相加
- 若主/视觉模型 API 域名是海外 IP 但服务在国内，记得在你的代理工具里把它们设为直连，避免绕路

## 🔒 安全

- API Key 只存在于 `hybrid_proxy.json`，该文件已被 `.gitignore` 排除
- 建议：服务只监听内网（`0.0.0.0:8888` 是内网地址），或在上游加一层鉴权
- 本项目不收集任何日志外的数据

## 📄 License

MIT
