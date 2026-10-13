"""LangChain tools wrapping the HivemindOS SuperAgent API.

Built by :func:`create_superagent_tools`, which binds a shared
:class:`~langchain_superagent.client.SuperAgentClient` into each tool so the
toolkit can hand them to any LangChain agent or chain.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from langchain_superagent.client import SuperAgentClient

MEMORY_SERVICE_ID = "cloud-superbrain"
"""Service id that hosts the managed memory operations (memory.create, ...)."""


class ListServicesArgs(BaseModel):
    name_filter: str = Field(
        default="",
        description=(
            "Keep only services whose id or name contains this text "
            "(case-insensitive). Empty string returns all services."
        ),
    )


class SearchActionsArgs(BaseModel):
    query: str = Field(
        default="",
        description="Free text matched against action names and descriptions.",
    )
    service_id: str = Field(
        default="",
        description="Narrow to one service, e.g. 'cloud-superbrain'.",
    )
    mode: str = Field(
        default="",
        description="One of 'read', 'write', 'execute'. Empty means all modes.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Max actions to return (1-50).",
    )


class InvokeServiceArgs(BaseModel):
    """Arguments for invoking a managed service operation."""

    service_id: str = Field(
        description=(
            "Managed service id, e.g. 'cloud-superbrain', 'managed-wallets', "
            "'answer-engine'. Discover ids with superagent_list_services."
        )
    )
    operation_id: str = Field(
        description=(
            "Operation to run: the short capability id (e.g. 'memory.create') "
            "or the full id from superagent_search_actions "
            "(e.g. 'services.invoke.cloud-superbrain.memory.create')."
        )
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON-serializable input object for the operation. Field names are "
            "service-specific; check the service catalog at "
            "https://api.hivemindos.app/openapi.json or via "
            "superagent_search_actions first."
        ),
    )
    idempotency_key: str = Field(
        default="",
        description=(
            "Optional idempotency key (8-200 chars, [A-Za-z0-9:._-]). A fresh "
            "UUID is generated automatically when omitted. Reusing a key with "
            "different input is rejected by the API."
        ),
    )


class MemoryWriteArgs(BaseModel):
    text: str = Field(
        description="The memory content (secrets are refused by the service)."
    )
    memory_type: str = Field(
        default="note",
        description="Label like 'note', 'fact', 'preference', 'decision'.",
    )
    tags: str = Field(
        default="",
        description="Comma-separated tag list, e.g. 'food,preferences'.",
    )
    project: str = Field(
        default="",
        description="Optionally scope the memory to a project.",
    )


class MemoryReadArgs(BaseModel):
    query: str = Field(
        description="The question or keywords to find memories for."
    )
    tags: str = Field(
        default="",
        description="Comma-separated tags; every tag must match.",
    )
    project: str = Field(
        default="",
        description="Narrow the search to a project.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max memories to return (1-50).",
    )


def _compact_services(services: list[dict[str, Any]]) -> str:
    lines = []
    for s in services:
        ops = s.get("operationCount", s.get("operations", "?"))
        lines.append(
            f"- {s.get('id')}: {s.get('name')} "
            f"[category={s.get('category')}, operations={ops}]"
        )
        desc = (s.get("description") or "").strip()
        if desc:
            lines.append(f"  {desc[:160]}")
    return "\n".join(lines)


def _compact_actions(actions: list[dict[str, Any]]) -> str:
    lines = []
    for a in actions:
        aid = a.get("actionId") or a.get("operationId") or a.get("id")
        desc = (a.get("description") or a.get("summary") or "").strip()
        lines.append(f"- {aid} (service: {a.get('serviceId')}, mode: {a.get('mode')})")
        if desc:
            lines.append(f"  {desc[:200]}")
    return "\n".join(lines) or "No actions found."


def create_superagent_tools(client: SuperAgentClient) -> list[BaseTool]:
    """Create the SuperAgent LangChain tools bound to ``client``."""

    @tool(args_schema=ListServicesArgs)
    def superagent_list_services(name_filter: str = "") -> str:
        """List the HivemindOS managed services this API key can use.

        Returns each service's id, name, category, operation count and a short
        description.
        """
        services = client.list_services()
        if name_filter:
            needle = name_filter.lower()
            services = [
                s
                for s in services
                if needle in str(s.get("id", "")).lower()
                or needle in str(s.get("name", "")).lower()
            ]
        return _compact_services(services)

    @tool(args_schema=SearchActionsArgs)
    def superagent_search_actions(
        query: str = "",
        service_id: str = "",
        mode: str = "",
        limit: int = 20,
    ) -> str:
        """Search the operations (actions) available to this API key.

        Returns action ids you can pass as operation_id to
        superagent_invoke_service.
        """
        if mode and mode not in ("read", "write", "execute"):
            return f"Invalid mode {mode!r}: expected 'read', 'write' or 'execute'."
        actions = client.search_actions(
            query=query or None,
            service_id=service_id or None,
            mode=mode or None,
            limit=max(1, min(limit, 50)),
        )
        return _compact_actions(actions)

    @tool(args_schema=InvokeServiceArgs)
    def superagent_invoke_service(
        service_id: str,
        operation_id: str,
        input_data: Optional[dict[str, Any]] = None,
        idempotency_key: str = "",
    ) -> str:
        """Invoke a managed service operation on the SuperAgent API.

        This is the workhorse tool: it runs any of the 400+ operations across
        the managed services (models, memory, wallets, trading, media,
        automations, nft-agents, ...). Mutations automatically get a unique
        Idempotency-Key header unless you supply your own. Charged calls spend
        API credits; check superagent_get_credit_balance first.
        """
        result = client.invoke(
            service_id=service_id,
            operation_id=operation_id,
            input=dict(input_data or {}),
            idempotency_key=idempotency_key or None,
        )
        return json.dumps(result, indent=2, default=str)[:8000]

    @tool
    def superagent_get_credit_balance() -> str:
        """Read the SuperAgent account's credit balance.

        Returns balanceCredits plus lifetime credited/debited totals. Invoke
        and other spending operations are refused when the balance cannot cover
        them, so check this before running paid work.
        """
        bal = client.credit_balance()
        return json.dumps(bal, indent=2, default=str)

    @tool
    def superagent_get_pricing() -> str:
        """Read server-owned service pricing for the SuperAgent API."""
        return json.dumps(client.pricing(), indent=2, default=str)[:8000]

    @tool(args_schema=MemoryWriteArgs)
    def superagent_memory_write(
        text: str,
        memory_type: str = "note",
        tags: str = "",
        project: str = "",
    ) -> str:
        """Save one memory (fact, decision, preference, note) to managed memory.

        Runs cloud-superbrain memory.create. Free for the first 200 memories,
        then 1 credit each.
        """
        input_data: dict[str, Any] = {"text": text, "type": memory_type}
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if tag_list:
            input_data["tags"] = tag_list
        if project:
            input_data["project"] = project
        result = client.invoke(
            service_id=MEMORY_SERVICE_ID,
            operation_id="memory.create",
            input=input_data,
        )
        return json.dumps(result, indent=2, default=str)[:8000]

    @tool(args_schema=MemoryReadArgs)
    def superagent_memory_read(
        query: str,
        tags: str = "",
        project: str = "",
        limit: int = 10,
    ) -> str:
        """Search saved memories by meaning and keywords.

        Runs cloud-superbrain memory.recall. Returns matching memories, not
        an answer. Costs 1 credit per recall when semantic mode applies.
        """
        input_data: dict[str, Any] = {"query": query, "limit": max(1, min(limit, 50))}
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if tag_list:
            input_data["tags"] = tag_list
        if project:
            input_data["project"] = project
        result = client.invoke(
            service_id=MEMORY_SERVICE_ID,
            operation_id="memory.recall",
            input=input_data,
        )
        return json.dumps(result, indent=2, default=str)[:8000]

    return [
        superagent_list_services,
        superagent_search_actions,
        superagent_invoke_service,
        superagent_get_credit_balance,
        superagent_get_pricing,
        superagent_memory_write,
        superagent_memory_read,
    ]
