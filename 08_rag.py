"""RAG 检索：向 LLM 提问笔记本配置问题。

逻辑：
- 若 Chroma 知识库存在 → 精确检索：计算全部文档相关度，取最接近的一条；
  若相关度低于阈值（知识库中没有相关内容）则交由大模型回答
- 若 Chroma 知识库不存在 → 直接让大模型回答

前置步骤：先运行 07_chroma_split.py 生成知识库
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 本地 Ollama 服务配置
BASE_URL = "http://192.168.3.140:11434/v1"
MODEL_NAME = "qwen2.5:0.5b-instruct"
API_KEY = "ollama"  # Ollama 本地服务不校验密钥，占位即可

# 知识库配置
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "laptop_kb"
# 相关度阈值（0~1，越大越相关）：低于该值视为知识库中无相关内容，交给大模型回答
RELEVANCE_THRESHOLD = 0.35


class LocalEmbeddings(Embeddings):
    """将 chromadb 的 DefaultEmbeddingFunction 适配为 langchain Embeddings 接口。"""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, emb)) for emb in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._fn([text])[0]))


def chroma_exists() -> bool:
    """检查 Chroma 知识库是否存在且有数据。"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def best_match_from_kb(question: str) -> tuple[str, float] | None:
    """精确检索：计算知识库中全部文档与问题的相关度，返回最接近的一条及其分数。

    返回 (文档全文, 相关度分数)；知识库为空时返回 None。
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=LocalEmbeddings(),
    )
    # 返回 (Document, relevance_score)，score 越大越相关
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=10
    )
    if not docs_with_scores:
        return None

    # 按相关度降序排序，取最接近的一条
    docs_with_scores.sort(key=lambda item: item[1], reverse=True)
    best_doc, best_score = docs_with_scores[0]
    return best_doc.page_content, best_score


def ask_llm(question: str) -> str:
    """直接让大模型回答。"""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
    )
    response = llm.invoke([HumanMessage(content=question)])
    return response.content


def main():
    # question = "ThinkPad T14p 2023 酷睿版的处理器型号是什么？"
    question = "ThinkPad T14p 2023 的生产商是谁"
    print(f"提问：{question}\n")

    if chroma_exists():
        # 知识库存在：精确检索最接近的一条
        best = best_match_from_kb(question)
        if best is None:
            print("知识库中没有文档，由大模型直接回答：\n")
            print(ask_llm(question))
            return

        content, score = best
        print(f"最接近的文档（相关度 {score:.2f}，阈值 {RELEVANCE_THRESHOLD}）：\n")

        if score >= RELEVANCE_THRESHOLD:
            # 相关度达标：返回该文档全文
            print(content)
        else:
            # 相关度不足：知识库无相关内容，交给大模型回答
            print("相关度低于阈值，知识库中可能没有相关内容，改由大模型回答：\n")
            print(ask_llm(question))
    else:
        # 知识库不存在：由大模型直接回答
        print("知识库不存在，由大模型直接回答：\n")
        answer = ask_llm(question)
        print(answer)


if __name__ == "__main__":
    main()
