"""Basic usage example for langchain-superagent.

Reads the API key from the HIVEMINDOS_API_KEY environment variable.
Only read-only calls run by default; the paid examples (invoke / memory)
are shown commented out — uncomment them once your account holds credits.
"""

import os
import sys

from langchain_superagent import SuperAgentToolkit

if not os.environ.get("HIVEMINDOS_API_KEY"):
    sys.exit("Set HIVEMINDOS_API_KEY first: export HIVEMINDOS_API_KEY='hmos_...'")

toolkit = SuperAgentToolkit.from_api_key()
tools = {t.name: t for t in toolkit.get_tools()}
print("tools:", sorted(tools))

# 1. Discover services -----------------------------------------------------
print("\n=== services (filtered to 'memory') ===")
print(tools["superagent_list_services"].invoke({"name_filter": "brain"}))

# 2. Search actions ---------------------------------------------------------
print("\n=== actions: memory ===")
print(
    tools["superagent_search_actions"].invoke(
        {"query": "memory", "service_id": "cloud-superbrain", "limit": 5}
    )
)

# 3. Credit balance (read-only) ---------------------------------------------
print("\n=== credit balance ===")
print(tools["superagent_get_credit_balance"].invoke({}))

# 4. Paid examples (need credits) -------------------------------------------
# print(tools["superagent_memory_write"].invoke({
#     "text": "The beekeeper prefers decaf after 2pm.",
#     "memory_type": "preference",
#     "tags": "food,preferences",
# }))
# print(tools["superagent_memory_read"].invoke({"query": "coffee preferences"}))
# print(tools["superagent_invoke_service"].invoke({
#     "service_id": "answer-engine",
#     "operation_id": "searches.create",
#     "input_data": {"query": "latest Base chain news"},
# }))
