"""
Configuration: LLM client and external API credentials.
All secrets are read from environment variables — never hardcode keys here.
"""

import os

from langchain_openai import ChatOpenAI

# ============ LLM CONFIG ===========

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")

LLM = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="openai/gpt-oss-120b",
)

# ============ EXTERNAL API CREDENTIALS ===========

MIREYE_TOKEN = os.getenv("MIREYE_TOKEN")
EIA_API_KEY = os.getenv("EIA_API_KEY")

# ============ DATABASE ===========

DATABASE_URL = os.getenv("DATABASE_URL")
