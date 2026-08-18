# 1. 导入必要的模块
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# 2. 定义期望返回的 JSON 结构 (Pydantic 模型)
class CodeAnalysis(BaseModel):
    language: str = Field(description="推荐使用的编程语言")
    complexity: str = Field(description="任务复杂度评估，如：低、中、高")
    key_steps: list[str] = Field(description="实现该需求的核心步骤列表")


# 3. 初始化本地 Ollama 模型（已修改为局域网连接）
llm = ChatOllama(
    model="qwen2.5:0.5b-instruct",
    base_url="http://192.168.3.140:11434",  # 指向局域网内的 Ollama 服务
    temperature=0.3,  # 降低温度，让 JSON 输出更稳定
    format="json"  # 强制 Ollama 输出 JSON 格式
)

# 4. 构建聊天提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的软件架构师。请根据用户的需求，分析并提取关键信息。\n{format_instructions}"),
    ("human", "需求描述：{user_requirement}")
])

# 5. 创建 JSON 解析器
parser = JsonOutputParser(pydantic_object=CodeAnalysis)

# 6. 使用 LCEL 管道语法构建执行链
chain = prompt | llm | parser

# 7. 运行并获取结果
if __name__ == "__main__":
    try:
        result = chain.invoke({
            "user_requirement": "帮我写一个 Python 脚本，每天定时抓取某网站的新闻标题，并保存到本地 TXT 文件中。",
            "format_instructions": parser.get_format_instructions()
        })

        # 此时 result 已经是一个干净的 Python 字典，无需手动解析
        print("分析结果：")
        print(f"推荐语言：{result['language']}")
        print(f"复杂度：{result['complexity']}")
        print("核心步骤：")
        for step in result['key_steps']:
            print(f"  - {step}")

    except Exception as e:
        print(f"解析失败，大模型可能未严格遵循格式: {e}")