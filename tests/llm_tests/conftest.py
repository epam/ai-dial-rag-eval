import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

CACHE_PATH = f"{Path(__file__).parent.parent}/data/cache/cache.db"


def pytest_configure(config):
    if config.getoption("--llm-mode") and config.getoption("--llm-mode")[0] == "real":
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)


class PromptSQLiteCache(SQLiteCache):
    def lookup(self, prompt: str, llm_string: str) -> Optional[Any]:
        prompt = json.loads(prompt)[0]["kwargs"]["content"]
        return super().lookup(prompt, "")

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        prompt = json.loads(prompt)[0]["kwargs"]["content"]
        super().update(prompt, "", return_val)


def _make_real_llm(cache: PromptSQLiteCache) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        model="gemini-3.1-flash-lite-preview",
        api_key=SecretStr(os.environ.get("DIAL_API_KEY", "")),
        azure_endpoint=os.environ.get("DIAL_URL", ""),
        api_version="2023-03-15-preview",
        max_tokens=2048,
        timeout=600,
        temperature=0,
        seed=3227,
        cache=cache,
    )


@pytest.fixture
def llm(request) -> BaseChatModel:
    llm_mode = (
        request.config.getoption("--llm-mode")[0]
        if request.config.getoption("--llm-mode")
        else "fake"
    )
    cache = PromptSQLiteCache(CACHE_PATH)
    if llm_mode == "real":
        return _make_real_llm(cache)
    else:
        if not os.path.exists(CACHE_PATH):
            raise FileNotFoundError(
                f"Cache not found at {CACHE_PATH}. "
                "Run tests with --llm-mode real to populate the cache."
            )
        return FakeChatModel(cache=cache)
