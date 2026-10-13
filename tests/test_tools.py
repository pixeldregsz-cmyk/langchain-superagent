"""Tests for langchain-superagent.

No API credits or network access required: HTTP is faked with
httpx.MockTransport, and tool schemas are validated statically.
Run with: pytest
"""

import json
import re

import httpx
import pytest

from langchain_superagent import (
    SuperAgentClient,
    SuperAgentToolkit,
    create_superagent_tools,
)
from langchain_superagent.client import BROWSER_USER_AGENT

IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9:._-]+$")


def _mock_client(handler) -> SuperAgentClient:
    return SuperAgentClient(
        api_key="hmos_test_key_do_not_use",
        transport=httpx.MockTransport(handler),
    )


def _json_response(payload, status=200):
    return httpx.Response(status, json=payload)


# ------------------------------------------------------------------------- #
# packaging / toolkit wiring
# ------------------------------------------------------------------------- #

def test_toolkit_builds_seven_tools():
    toolkit = SuperAgentToolkit(client=_mock_client(lambda r: _json_response({})))
    tools = toolkit.get_tools()
    names = sorted(t.name for t in tools)
    assert names == [
        "superagent_get_credit_balance",
        "superagent_get_pricing",
        "superagent_invoke_service",
        "superagent_list_services",
        "superagent_memory_read",
        "superagent_memory_write",
        "superagent_search_actions",
    ]


def test_toolkit_requires_api_key(monkeypatch):
    monkeypatch.delenv("HIVEMINDOS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="HIVEMINDOS_API_KEY"):
        SuperAgentToolkit.from_api_key()


def test_toolkit_reads_env_key(monkeypatch):
    monkeypatch.setenv("HIVEMINDOS_API_KEY", "hmos_from_env")
    # the sandbox ships proxy/no_proxy vars that this httpx version cannot
    # parse (e.g. a '*[::1]' no_proxy entry); a real client build must not
    # depend on them
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    toolkit = SuperAgentToolkit.from_api_key()
    assert toolkit.client.api_key == "hmos_from_env"


# ------------------------------------------------------------------------- #
# schema validity (no credits needed)
# ------------------------------------------------------------------------- #

def test_tool_schemas_are_valid():
    toolkit = SuperAgentToolkit(client=_mock_client(lambda r: _json_response({})))
    for t in toolkit.get_tools():
        assert t.name.startswith("superagent_"), t.name
        assert t.description and len(t.description.strip()) > 20, t.name
        schema = t.args_schema.model_json_schema()
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties", {}), dict)
        # every declared arg must have a description so agents use it well
        for prop, spec in schema["properties"].items():
            assert spec.get("description"), f"{t.name}.{prop} lacks a description"


def test_invoke_tool_args_schema():
    toolkit = SuperAgentToolkit(client=_mock_client(lambda r: _json_response({})))
    invoke_tool = next(
        t for t in toolkit.get_tools() if t.name == "superagent_invoke_service"
    )
    schema = invoke_tool.args_schema.model_json_schema()
    assert set(schema["required"]) == {"service_id", "operation_id"}
    assert "input_data" in schema["properties"]
    assert "idempotency_key" in schema["properties"]


def test_no_secrets_leak_into_schemas():
    toolkit = SuperAgentToolkit(client=_mock_client(lambda r: _json_response({})))
    for t in toolkit.get_tools():
        blob = t.description + json.dumps(t.args_schema.model_json_schema())
        assert "hmos_" not in blob
        assert "Bearer" not in blob


# ------------------------------------------------------------------------- #
# client behaviour (mocked HTTP)
# ------------------------------------------------------------------------- #

def test_client_sends_browser_user_agent_and_auth():
    seen = {}

    def handler(request: httpx.Request):
        seen["ua"] = request.headers["user-agent"]
        seen["auth"] = request.headers["authorization"]
        return _json_response({"ok": True, "services": []})

    _mock_client(handler).list_services()
    assert seen["ua"] == BROWSER_USER_AGENT
    assert seen["auth"] == "Bearer hmos_test_key_do_not_use"


def test_invoke_attaches_idempotency_key_and_operation():
    seen = {}

    def handler(request: httpx.Request):
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content.decode())
        seen["path"] = request.url.path
        return _json_response({"ok": True, "result": "done"})

    client = _mock_client(handler)
    out = client.invoke(
        service_id="cloud-superbrain",
        operation_id="memory.create",
        input={"text": "hello"},
    )
    assert out == {"ok": True, "result": "done"}
    assert seen["path"] == "/v1/services/cloud-superbrain/invoke"
    assert seen["body"]["operationId"] == "memory.create"
    assert seen["body"]["input"] == {"text": "hello"}
    assert seen["idem"] and IDEMPOTENCY_PATTERN.match(seen["idem"])
    assert 8 <= len(seen["idem"]) <= 200


def test_invoke_respects_caller_supplied_idempotency_key():
    seen = {}

    def handler(request: httpx.Request):
        seen["idem"] = request.headers.get("idempotency-key")
        return _json_response({"ok": True})

    _mock_client(handler).invoke(
        service_id="cloud-superbrain",
        operation_id="memory.recall",
        input={"query": "x"},
        idempotency_key="my-unique-key-123",
    )
    assert seen["idem"] == "my-unique-key-123"


def test_idempotency_keys_are_unique():
    keys = {SuperAgentClient.new_idempotency_key() for _ in range(1000)}
    assert len(keys) == 1000
    assert all(IDEMPOTENCY_PATTERN.match(k) for k in keys)


def test_api_error_surfaces_status_and_message():
    def handler(request: httpx.Request):
        return httpx.Response(402, json={"error": "insufficient credits"})

    from langchain_superagent import SuperAgentAPIError

    with pytest.raises(SuperAgentAPIError) as exc_info:
        _mock_client(handler).credit_balance()
    assert exc_info.value.status_code == 402
    assert "insufficient credits" in str(exc_info.value)


def test_list_services_parses_live_shape():
    payload = {
        "ok": True,
        "services": [
            {
                "id": "cloud-superbrain",
                "name": "Cloud Superbrain",
                "description": "Managed memory and knowledge.",
                "category": "agent",
                "operationCount": 29,
            }
        ],
    }

    def handler(request: httpx.Request):
        return _json_response(payload)

    tools = create_superagent_tools(_mock_client(handler))
    list_tool = next(t for t in tools if t.name == "superagent_list_services")
    out = list_tool.invoke({"name_filter": "superbrain"})
    assert "cloud-superbrain" in out
    assert "Cloud Superbrain" in out
    # filter excludes non-matching services
    out_all = list_tool.invoke({"name_filter": "zzz-no-match"})
    assert "cloud-superbrain" not in out_all


def test_credit_balance_parses_live_shape():
    payload = {
        "ok": True,
        "credits": {
            "accountId": "pagw_acct_test",
            "balanceCredits": 0,
            "totalCreditedCredits": 0,
            "totalDebitedCredits": 0,
        },
    }

    def handler(request: httpx.Request):
        return _json_response(payload)

    tools = create_superagent_tools(_mock_client(handler))
    bal_tool = next(t for t in tools if t.name == "superagent_get_credit_balance")
    out = json.loads(bal_tool.invoke({}))
    assert out["balanceCredits"] == 0


def test_memory_tools_target_cloud_superbrain():
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.url.path, json.loads(request.content.decode())))
        return _json_response({"ok": True, "memory": {"id": "mem_1"}})

    tools = create_superagent_tools(_mock_client(handler))
    by_name = {t.name: t for t in tools}
    by_name["superagent_memory_write"].invoke(
        {"text": "Pixel likes decaf.", "tags": "food, prefs", "project": "hive"}
    )
    by_name["superagent_memory_read"].invoke({"query": "coffee", "limit": 3})

    assert seen[0][0] == "/v1/services/cloud-superbrain/invoke"
    assert seen[0][1]["operationId"] == "memory.create"
    assert seen[0][1]["input"]["text"] == "Pixel likes decaf."
    assert seen[0][1]["input"]["tags"] == ["food", "prefs"]
    assert seen[1][1]["operationId"] == "memory.recall"
    assert seen[1][1]["input"]["query"] == "coffee"
