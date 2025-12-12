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

## Embedding
* 下载模型  
目前使用的是 huggingface 上的模型，具体模型下载可参考国内镜像站 https://hf-mirror.com/
目前使用的向量模型有 bge-m3、bce 等
模型下载后，需要将模型文件放置在 models 目录下，并修改 embedding/config.py 中的配置，包括模型路径、模型名称、模型类型等
例如目录列表为
```bash
.
├── models
│   ├── bge-m3
│   └── bce-embedding-base_v1
|   └── bce-reranker-base_v1
|   └── bge-reranker-v2-m3
```

* 配置文件  
修改 embedding/config.py 中的配置，包括模型路径、模型名称、模型类型等

* 启动 Embedding 服务
```bash
USE_CUDA=True python -u embedding/main.py
```

* 调用 Embedding 服务
1. 直接调用 API（兼容 OpenAI 客户端）
```python
import openai

# 配置 OpenAI 客户端指向本地服务
openai.api_base = "http://localhost:9090/v1"
openai.api_key = "sk-local-embedding-api-key"  # 如果启用了 API Key 验证

# 生成嵌入（与 OpenAI API 完全一致）
response = openai.Embedding.create(
    input=["这是一个测试文本", "本地向量模型真好用"],
    model="bge-m3"  # 映射到本地的 bge-m3
)

# 处理结果
embeddings = [item["embedding"] for item in response["data"]]
print(f"嵌入维度: {len(embeddings[0])}")
print(f"第一个向量: {embeddings[0][:5]}...")

# reranker 模型 调用 Rerank API
response = openai.Rerank.create(
    query="介绍人工智能的应用",
    documents=[
        "机器学习是人工智能的一个重要分支，研究计算机如何学习",
        "人工智能在医疗领域的应用包括疾病诊断、药物研发等",
        "Python 是一种流行的编程语言，常用于数据科学",
        "自动驾驶汽车是人工智能在交通领域的典型应用",
        "人工智能可以帮助企业优化供应链管理，提高效率"
    ],
    model="bge-rerakner-v2-m3",
    return_scores=True
)

# 处理结果
print(f"模型: {response.model}")
print("排序结果:")
for idx, item in enumerate(response.data):
    print(f"排名 {idx+1}: 分数={item.score:.4f}, 文本: {item.document}")
    print(f"  原始索引: {item.original_index}")
    print()
```

2. 使用 Curl 调用
```bash
# embedding
curl http://localhost:9090/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-embedding-api-key" \
  -d '{
    "input": ["Hello world", "本地向量模型"],
    "model": "bge-m3",
    "encoding_format": "float"
  }'
```
```bash
# reranker
curl http://localhost:9090/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-embedding-api-key" \
  -d '{
    "query": "介绍人工智能的应用",
    "documents": [
      "机器学习是人工智能的一个重要分支",
      "人工智能在医疗领域有广泛应用",
      "编程语言包括 Python、Java 等",
      "人工智能可以用于自动驾驶汽车"
    ],
    "model": "bce-reranker-base_v1",
    "return_scores": true,
    "return_documents": true
  }'
```
```bash
# 返回精简结果（仅返回索引和分数）
curl http://localhost:9090/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "介绍人工智能的应用",
    "documents": [
      "机器学习是人工智能的一个重要分支",
      "人工智能在医疗领域有广泛应用",
      "编程语言包括 Python、Java 等"
    ],
    "model": "bge-reranker-v2-m3",
    "return_scores": true,
    "return_documents": false
  }'
```


# PaperModels
PaperModels 是基于对大模型论文中提到的一些技术核心点，进行简单的实现，不涉及到真实场景中的复杂实现，仅作为技术分享和交流使用。

## [DeepSeek V3.2](paper_models/deepseek_v3.2/deepseek_v3.2.md)