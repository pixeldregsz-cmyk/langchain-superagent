# langchain-superagent

LangChain tools + toolkit for the [HivemindOS SuperAgent API](https://hivemindos.app/superagent-api/docs/) — 40+ managed services and 400+ operations (models, memory, wallets, trading, media, automations, nft-agents, …) callable from any LangChain agent.

Built for the HivemindOS **25,000-credit integration bounty** ("Publish a LangChain tool for the SuperAgent API, merged or released where people install it"). Bounty claims are filed on the HivemindOS Hall Rewards page with a link to the merged PR / published package.

## Install

```bash
pip install langchain-superagent
```

Or from source:

```bash
git clone <repo-url>
cd langchain-superagent
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

## Quickstart

```python
import os
os.environ["HIVEMINDOS_API_KEY"] = "hmos_..."  # never commit this

from langchain_superagent import SuperAgentToolkit

toolkit = SuperAgentToolkit.from_api_key()  # reads HIVEMINDOS_API_KEY
tools = toolkit.get_tools()

# plug into any agent, e.g.:
# agent = create_react_agent(model, tools)
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a runnable walkthrough.

## Tools

| Tool | What it does |
|---|---|
| `superagent_list_services` | List managed services the key can use (id, category, operation count). Optional `name_filter`. |
| `superagent_search_actions` | Search the 400+ operations by free text, `service_id`, and `mode` (`read`/`write`/`execute`). Returns action ids for invoking. |
| `superagent_invoke_service` | Invoke any operation: `service_id` + `operation_id` (short like `memory.create` or full like `services.invoke.cloud-superbrain.memory.create`) + `input_data` dict. Auto-generates a unique `Idempotency-Key` header per call unless you pass `idempotency_key`. |
| `superagent_get_credit_balance` | Read the account's credit balance (paid calls are refused without credits — check first). |
| `superagent_get_pricing` | Read server-owned service pricing. |
| `superagent_memory_write` | Save a fact/decision/preference/note to managed memory (`cloud-superbrain` → `memory.create`). Free for the first 200 memories, then 1 credit each. |
| `superagent_memory_read` | Semantic + keyword search over saved memories (`memory.recall`). |

Typical agent loop: `superagent_list_services` → `superagent_search_actions` → `superagent_invoke_service`.

## Auth & safety notes

- **API key**: pass `api_key=` or set `HIVEMINDOS_API_KEY`. The key is sent as a `Bearer` token and is never logged or embedded in tool schemas.
- **Idempotency**: every mutation automatically gets a unique `Idempotency-Key` header (required by the API; reusing a key with different input is rejected). Supply your own for safe retries.
- **Credits**: spending operations need a funded account — top up at the HivemindOS console billing page. `superagent_get_credit_balance` is the pre-flight check.
- **Cloudflare**: the API edge blocks default Python user-agents (HTTP 403, error 1010). This client sends a browser-like `User-Agent` so plain scripts don't get bot-blocked.
- **App revenue share**: pass `app_id="app_<12 hex>"` to `SuperAgentToolkit.from_api_key` to attach `X-HivemindOS-App` (Hall-registered apps earn a share of charged calls).

## API reference

- Docs: https://hivemindos.app/superagent-api/docs/
- OpenAPI 3.1 spec: https://api.hivemindos.app/openapi.json
- Remote MCP (alternative integration path): https://api.hivemindos.app/mcp

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests use `httpx.MockTransport` — no API key, credits, or network needed.

## Publishing checklist

- [ ] Create a PyPI account and an API token
- [ ] `python -m build && twine upload dist/*`
- [ ] Tag a GitHub release (`v0.1.0`)
- [ ] File the bounty claim on the HivemindOS Hall Rewards page with the PyPI link

## License

MIT — see [LICENSE](LICENSE).
