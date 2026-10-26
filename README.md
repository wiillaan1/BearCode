# BearCode

BearCode 是一个基于 Python 的本地 Agent Runtime，提供命令行入口和网页工作台。

## 环境要求

- Python **3.11 及以上**。已实测 3.11.15 与 3.13.9；3.12 未实测。
- macOS 自带的 `python3` 可能是 3.9，不受支持（测试会在导入阶段失败）。请用 3.11 或更高版本创建虚拟环境，例如把下面的 `python3` 换成 `python3.11`。
- 运行网页工作台和测试不需要 Node.js。

## 1. 安装依赖

在仓库根目录执行：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

## 2. 配置模型（需要你自己的 API key）

```bash
test -f .env || cp .env.example .env
```

用编辑器打开 `.env`，填入你自己的密钥、接口地址和模型 ID。下面两种方式选择一种；使用服务专用变量时，删除或注释模板里已启用的 `APIKEY`、`API`、`MINI_CLAUDE_MODEL`，避免混用配置。

OpenAI-compatible 接口：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-provider.example/v1
MODEL=your-model-id
```

Anthropic-compatible 接口：

```env
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=https://your-provider.example/anthropic
MODEL=your-model-id
```

| 不需要 API key | 需要有效的 API key |
| --- | --- |
| 安装依赖、运行测试、`python -m agents.main --help`、启动网页工作台并打开页面 | 在命令行或网页工作台里与模型对话（交互模式、一次性任务、网页里发起任务） |

## 3. 命令行

```bash
# 交互模式
python -m agents.main

# 执行一次任务
python -m agents.main "总结这个项目的目录结构和核心模块"

# 查看全部参数和内置命令
python -m agents.main --help
```

## 4. 网页工作台

```bash
python -m agents.web
```

看到 `Bear Code Web → http://127.0.0.1:8765` 后，在浏览器打开 <http://127.0.0.1:8765>。服务只监听本机 `127.0.0.1`；停止服务在终端按 `Ctrl+C`。需要换端口时：

```bash
python -m agents.web --port 8766
```

前端由后端直接提供，不需要 `npm install` 或构建；不要直接双击 `web/index.html` 打开，页面需要同源的 API 和会话 Cookie。

## 5. 运行测试

```bash
python -m unittest discover -s tests -q
```

测试使用模拟模型，不需要 API key，也不访问网络。

许可证见 LICENSE。
