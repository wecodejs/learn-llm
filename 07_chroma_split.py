"""知识库向量化：将 txts/ 下的所有 txt 分割并存入 Chroma 向量数据库。

- 使用 Chroma 内置的本地 embedding 模型（all-MiniLM-L6-v2），无需外部 API
- 向量库持久化到 chroma_db/ 目录
"""

from pathlib import Path

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 路径配置
TXT_DIR = Path(__file__).parent / "txts"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "laptop_kb"


class LocalEmbeddings(Embeddings):
    """将 chromadb 的 DefaultEmbeddingFunction 适配为 langchain Embeddings 接口。"""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, emb)) for emb in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._fn([text])[0]))


def main():
    # 1. 读取所有 txt 文件
    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"未在 {TXT_DIR} 下找到任何 txt 文件")
        return

    documents = []
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8")
        documents.append(content)
        print(f"读取：{txt_file.name}（{len(content)} 字符）")

    # 2. 文本分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.create_documents(documents)
    print(f"分割完成：共 {len(chunks)} 个文本块")

    # 3. 向量化并持久化到 Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=LocalEmbeddings(),  # 本地 embedding，无需 API
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"向量化完成：已入库 {count} 条，保存到 {CHROMA_DIR}")


if __name__ == "__main__":
    main()
