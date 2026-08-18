"""基础 Prompt 示例：通过 langchain 调用本地 Ollama 大模型。

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
    # 初始化模型客户端
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
    )

    # 提问并获取回答
    question = "python中如何定义class"
    print(f"提问：{question}\n")

    response = llm.invoke([HumanMessage(content=question)])

    # 打印回答内容
    print("回答：")
    print(response.content)


if __name__ == "__main__":
    main()
