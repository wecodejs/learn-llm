import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

# 1. 定义结构化输出的 Schema (使用 Pydantic)
class CodeReview(BaseModel):
    issues: List[str] = Field(description="代码中发现的问题列表")
    severity_score: int = Field(description="代码严重度评分，1-10分，10分为最严重")
    suggestion: str = Field(description="具体的修复建议和最佳实践")

# 2. 初始化组件
load_dotenv()
llm = ChatOllama(
    model="qwen2.5:0.5b-instruct",
    base_url="http://192.168.3.140:11434",  # 指向局域网内的 Ollama 服务
    temperature=0.3,  # 降低温度，让 JSON 输出更稳定
    format="json"  # 强制 Ollama 输出 JSON 格式
)
parser = JsonOutputParser(pydantic_object=CodeReview)

# 3. 构建提示词模板 (替代 PipelinePromptTemplate)
# 使用 {format_instructions} 让 LLM 知道必须输出合法的 JSON
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位拥有10年经验的资深软件工程师，专注于代码质量和安全审查。\n{format_instructions}"),
    ("human", "请审查以下代码：\n```python\n{code}\n```")
])

# 4. 自定义业务逻辑函数 (无缝接入 LCEL 管道)
def save_review_report(review_data):
    """将审查结果自动保存为本地 JSON 文件"""
    try:
        with open("code_review_report.json", "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=4)
        print("✅ 审查报告已成功保存至 code_review_report.json")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")
    return review_data  # 必须返回数据，以便在链中继续传递

# 5. 使用 LCEL 管道符 `|` 构建复杂链路
chain = (
    {
        "code": RunnablePassthrough(),  # 透传用户的代码输入
        "format_instructions": lambda _: parser.get_format_instructions()  # 动态注入格式指令
    }

    | prompt        # 组装提示词
    | llm           # 调用大模型
    | parser        # 解析为结构化 JSON
    | save_review_report  # 自定义函数入链，保存文件
)

# 6. 调用 Chain
test_code = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

try:
    result = chain.invoke(test_code)
    print("🤖 结构化审查结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    print("❌ 链路执行失败:", e)