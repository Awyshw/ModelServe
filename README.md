# ModelServe
这是一个为了兼容 OpenAI API 的调用方式来搭建本地模型(大语言模型、向量模型、rerank 模型)服务的代码仓库

## 安装 uv
* uv 操作详见：https://uv.oaix.tech/getting-started/features/#_2
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install xxx  # 安装 python 指定版本（例如 3.12）
uv python pin xxx #  指定 uv 使用的 python 版本
```
* 虚拟环境
```bash
uv venv  # 创建 python 虚拟环境
source .venv/bin/activate  # 激活当前虚拟环境
```

* 安装依赖
```bash
uv init  # 创建一个新的 Python 项目
uv add -r requirements.txt  # 向项目添加依赖项
```