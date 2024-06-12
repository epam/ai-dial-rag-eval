from importlib.metadata import version
from typing import Optional

PKG_NAME = "aidial-rag-eval"


def get_tools_versions(tools: Optional[list[str]] = None) -> dict[str, str]:
    tools = tools or [PKG_NAME]
    return {tool: version(tool) for tool in tools}
