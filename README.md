# hybrid-image-proxy 🖼️→📝

**给不支持视觉的 LLM 接上图片理解能力** 的兼容中转代理（单文件、零依赖、可跑在 OpenWrt 软路由/树莓派/任何有 Python3 的机器上）。同时支持 OpenAI 与 Anthropic 两种协议。

```
无图请求  → 原样透传给主模型（支持流式 SSE）
有图请求  → 先调用视觉模型分析图片（逐字 OCR 提取文字/报错/界面信息），
           把图片替换为文字分析结果，再转发给主模型
```

典型场景：你的主力模型不支持视觉（比如某些文本模型 / 内部网关），但你又想让它在对话里"看懂"截图、报错、UI 界面。这个代理在中间做一次桥接——图由视觉模型消化成文字，文字再交给你的主力模型。

**可接入任意主流客户端**（WorkBuddy / Claude Code / Codex / OpenClaw 等 CLI 编码代理，或任意 OpenAI SDK / Chat UI）：只需新建一个自定义模型，把 API 地址指向本服务，即可让原本"看不到图"的模型具备看图能力。详见下方 [接入客户端](#4-接入客户端新建自定义模型)。

## ✨ 特性

- **双协议支持**：同时提供 OpenAI（`/v1/chat/completions`）与 Anthropic（`/v1/messages`）两个端点，覆盖 WorkBuddy / Codex / Claude Code / OpenClaw 等主流客户端
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

配置查找顺序：**环境变量 `HYBRID_PROXY_CONFIG` > 脚本同目录的 `hybrid_proxy.json` > `/root/hybrid_proxy.json`**。把配置文件放到脚本同目录，直接运行即可：

```bash
python3 hybrid_proxy.py
# 可选：环境变量覆盖配置/日志路径
# HYBRID_PROXY_CONFIG=/path/to/hybrid_proxy.json HYBRID_PROXY_LOG=/path/to/log python3 hybrid_proxy.py
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

返回内容和 OpenAI 完全一致。

### 4. 接入客户端（新建自定义模型）

中转服务本身是一个 OpenAI 兼容接口，**你需要在客户端里新建一个「自定义模型」，把 API 地址指向它**，才能真正用上。核心就三行配置：

| 配置项 | 值 |
|---|---|
| API 地址 / Base URL | `http://<部署机器IP>:8888/v1` |
| API Key | 随便填（如 `sk-xxx`，本服务不校验） |
| 模型名 | 随便填（如 `hybrid`，代理会强制替换成配置里的主模型名） |

> 模型名填什么都能用，因为 `hybrid_proxy.py` 会忽略你传入的模型名，统一走配置里的 `deepseek_model`（视觉模型同理自动触发）。

**以 CLI 编码代理为例**（这是本工具最典型的使用场景——给不带视觉的编码模型接看图能力）。本服务同时提供两个协议端点，两类客户端都能直接接入：

- **OpenAI 兼容端点**（`/v1/chat/completions`）→ **Codex、WorkBuddy、OpenAI SDK** 及各种 Chat UI：新增一个 OpenAI 兼容的「自定义模型」，Base URL 填 `http://<部署机器IP>:8888/v1`，API Key 随便填，模型名随便填即可。
- **Anthropic 兼容端点**（`/v1/messages`）→ **Claude Code、OpenClaw** 等：把模型供应商指向本服务。以 Claude Code 为例，设置环境变量：
  ```bash
  export ANTHROPIC_BASE_URL=http://<部署机器IP>:8888
  export ANTHROPIC_API_KEY=sk-xxx    # 随便填，本服务不校验
  ```
  之后选用任意模型名，发文字走透传，发图片自动走「视觉分析 → 主模型」桥接。

> 两种协议下，模型名都随便填——代理会忽略你传入的模型名，统一走配置里的 `deepseek_model`（视觉模型同理自动触发）。

> 若客户端要求 HTTPS：中转服务本身只提供 HTTP。可在前面套一层反代（Nginx/Caddy）或走内网直连；局域网内直接 HTTP 即可。

## 🖥️ 跨平台部署

### macOS

```bash
# 1. 安装 Python3（若未装）
brew install python3

# 2. 下载本仓库两个文件到同一目录：hybrid_proxy.py + hybrid_proxy.json.example
#    复制 example 为 hybrid_proxy.json 并填写 Key

# 3. 启动（自动读取同目录的 hybrid_proxy.json）
python3 hybrid_proxy.py

# 4. 局域网访问：http://你的Mac的IP:8888/v1/chat/completions
```

### Windows（PowerShell）

```powershell
# 1. 安装 Python3：https://www.python.org/downloads/ ，安装时勾选 "Add Python to PATH"
# 2. 下载本仓库两个文件到同一目录：hybrid_proxy.py + hybrid_proxy.json.example
#    复制 example 为 hybrid_proxy.json 并填写 Key（文件名必须叫 hybrid_proxy.json）

# 3. 启动（自动读取同目录的 hybrid_proxy.json）
python hybrid_proxy.py
```

> Windows 注意：
> - 命令用 `python`（不是 `python3`）
> - 首次运行弹防火墙提示时点「允许访问」，否则局域网其他设备连不上 8888 端口
> - 日志默认只输出到终端；想落盘可加环境变量：`$env:HYBRID_PROXY_LOG = "$PWD\hybrid_proxy.log"`
> - `openwrt/` 目录的脚本是软路由专用的，Windows 直接忽略

### OpenWrt 软路由

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
