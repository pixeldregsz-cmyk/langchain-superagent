"""LangChain tools for the HivemindOS SuperAgent API."""

from langchain_superagent.client import (
    DEFAULT_BASE_URL,
    SuperAgentAPIError,
    SuperAgentClient,
)
from langchain_superagent.toolkit import SuperAgentToolkit
from langchain_superagent.tools import create_superagent_tools

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BASE_URL",
    "SuperAgentAPIError",
    "SuperAgentClient",
    "SuperAgentToolkit",
    "create_superagent_tools",
    "__version__",
]
