"""流式输出示例：通过 langchain 流式打印本地 Ollama 大模型的回答。

Ollama 服务地址：http://192.168.3.140:11434
模型：qwen3.5:latest
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 本地 Ollama 服务配置
BASE_URL = "http://192.168.3.140:11434/v1"
MODEL_NAME = "qwen2.5:0.5b-instruct"
API_KEY = "ollama"  # Ollama 本地服务不校验密钥，占位即可


def main():
    # 初始化模型客户端（开启流式）
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
        streaming=True,
    )

    # 提问并流式获取回答
    question = "如何让大模型回答问题时输出的是json结构，使用的是langchain库"
    print(f"提问：{question}\n")
    print("回答：")

    for chunk in llm.stream([HumanMessage(content=question)]):
        print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
