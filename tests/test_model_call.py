import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.environ.get("LLM_API_KEY", "")
base_url = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1")

if not api_key:
    print("错误: 未找到 LLM_API_KEY，请在 .env 文件中配置")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
)

response = client.chat.completions.create(
    model="Pro/zai-org/GLM-4.7",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好，请介绍一下你自己"},
    ],
)

print(response.choices[0].message.content)
