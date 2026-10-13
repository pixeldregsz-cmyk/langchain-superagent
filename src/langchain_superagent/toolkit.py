"""LangChain toolkit bundling the SuperAgent tools."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict, Field

from langchain_superagent.client import SuperAgentClient
from langchain_superagent.tools import create_superagent_tools

API_KEY_ENV_VAR = "HIVEMINDOS_API_KEY"


class SuperAgentToolkit(BaseToolkit):
    """Toolkit exposing the HivemindOS SuperAgent API as LangChain tools.

    Gives an agent: service discovery (list/search), generic operation
    invocation with automatic idempotency keys, credit-balance and pricing
    reads, and managed memory write/read.

    Example:
        >>> toolkit = SuperAgentToolkit.from_api_key()  # reads HIVEMINDOS_API_KEY
        >>> tools = toolkit.get_tools()
        >>> agent = create_react_agent(model, tools)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SuperAgentClient = Field(description="Configured SuperAgent API client.")

    def get_tools(self) -> list[BaseTool]:
        """Return the SuperAgent tools bound to this toolkit's client."""
        return create_superagent_tools(self.client)

    @classmethod
    def from_api_key(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **client_kwargs: Any,
    ) -> "SuperAgentToolkit":
        """Build a toolkit from an API key (or the HIVEMINDOS_API_KEY env var).

        Raises:
            ValueError: If no API key is provided and the env var is unset.
        """
        key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not key:
            raise ValueError(
                f"No SuperAgent API key provided. Pass api_key or set the "
                f"{API_KEY_ENV_VAR} environment variable."
            )
        kwargs: dict[str, Any] = dict(client_kwargs)
        if base_url:
            kwargs["base_url"] = base_url
        if app_id:
            kwargs["app_id"] = app_id
        if project_id:
            kwargs["project_id"] = project_id
        return cls(client=SuperAgentClient(api_key=key, **kwargs))
