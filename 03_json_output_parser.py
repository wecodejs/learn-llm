"""JSON 输出示例：使用 JsonOutputParser 让 langchain 返回结构化 JSON。

Ollama 服务地址：http://192.168.3.140:11434
模型：qwen2.5:latest
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 本地 Ollama 服务配置
BASE_URL = "http://192.168.3.140:11434/v1"
MODEL_NAME = "qwen2.5"
API_KEY = "ollama"  # Ollama 本地服务不校验密钥，占位即可


# 定义期望的 JSON 结构
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


def main():
    # 初始化模型客户端
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    # 创建 JsonOutputParser，并传入结构定义
    parser = JsonOutputParser(pydantic_object=Person)

    # 构造提示词模板，附带 JSON 输出格式说明
    prompt = PromptTemplate(
        template="回答用户的提问。\n{format_instructions}\n问题：{query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 组装链：prompt -> llm -> parser
    chain = prompt | llm | parser

    question = "张三今年25岁，请返回他的姓名和年龄"
    print(f"提问：{question}\n")

    result = chain.invoke({"query": question})

    # 打印解析后的结构化结果
    print("回答：")
    print(result)
    print(f"\n类型：{type(result).__name__}")
    print(f"name={result['name']}, age={result['age']}")


if __name__ == "__main__":
    main()
