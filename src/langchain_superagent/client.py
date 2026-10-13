"""Low-level HTTP client for the HivemindOS SuperAgent API.

Docs: https://hivemindos.app/superagent-api/docs/
OpenAPI: https://api.hivemindos.app/openapi.json

Authentication is a Bearer API key. Every mutation / spending call must carry a
unique ``Idempotency-Key`` header (reusing a key with different input is
rejected server-side), so this client generates one automatically unless the
caller supplies its own.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "https://api.hivemindos.app/v1"

# The API sits behind Cloudflare bot management. Default python HTTP clients
# (python-requests, plain httpx) get an HTTP 403 "error code: 1010" bot block;
# a browser-like User-Agent passes. Verified 2026-10-03.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class SuperAgentAPIError(Exception):
    """Raised when the SuperAgent API returns an error response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        body: Any = None,
    ) -> None:
        super().__init__(f"SuperAgent API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body


class SuperAgentClient:
    """Synchronous HTTP client for the SuperAgent REST API.

    Args:
        api_key: HivemindOS SuperAgent API key (``hmos_...``). Prefer passing it
            via the ``HIVEMINDOS_API_KEY`` environment variable and never commit
            it to source control.
        base_url: API base URL. Defaults to the production endpoint.
        app_id: Optional Hall app id (``app_<12 hex>``). Sent as
            ``X-HivemindOS-App``; its owner earns a share of charged calls and
            charged responses carry an ``X-HivemindOS-Receipt-Id``.
        project_id: Optional ``X-HivemindOS-Project`` selector for
            account-level keys.
        timeout: Request timeout in seconds.
        transport: Optional httpx transport (used by tests with MockTransport).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        app_id: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "An API key is required. Pass api_key or set HIVEMINDOS_API_KEY."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_id = app_id
        self.project_id = project_id
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "application/json",
            },
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def new_idempotency_key() -> str:
        """Generate a fresh idempotency key (uuid4 hex, fits the API pattern)."""
        return uuid.uuid4().hex

    def _mutation_headers(
        self, idempotency_key: Optional[str] = None
    ) -> dict[str, str]:
        headers = {"Idempotency-Key": idempotency_key or self.new_idempotency_key()}
        if self.app_id:
            headers["X-HivemindOS-App"] = self.app_id
        if self.project_id:
            headers["X-HivemindOS-Project"] = self.project_id
        return headers

    def _handle(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                body = response.json()
            except Exception:  # noqa: BLE001
                body = response.text
            message = (
                body.get("error") or body.get("message")
                if isinstance(body, dict)
                else str(body)
            )
            raise SuperAgentAPIError(response.status_code, str(message), body)
        if not response.content:
            return None
        try:
            return response.json()
        except Exception:  # noqa: BLE001
            return response.text

    def get(
        self, path: str, params: Optional[dict[str, Any]] = None
    ) -> Any:
        """GET a path relative to the API base URL."""
        return self._handle(self._http.get(path, params=params))

    def post(
        self,
        path: str,
        json: Optional[dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        """POST a path, always attaching an Idempotency-Key header."""
        return self._handle(
            self._http.post(path, json=json, headers=self._mutation_headers(idempotency_key))
        )

    # ------------------------------------------------------------------ #
    # API conveniences
    # ------------------------------------------------------------------ #
    def list_services(self) -> list[dict[str, Any]]:
        """List the managed services this API key can use."""
        data = self.get("/services")
        return data.get("services", []) if isinstance(data, dict) else data

    def search_actions(
        self,
        query: Optional[str] = None,
        action_id: Optional[str] = None,
        service_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search actions available to this API key.

        mode is one of: read, write, execute.
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if action_id:
            params["actionId"] = action_id
        if service_id:
            params["serviceId"] = service_id
        if mode:
            params["mode"] = mode
        data = self.get("/actions", params=params)
        if isinstance(data, dict):
            return data.get("actions", data.get("items", []))
        return data

    def invoke(
        self,
        service_id: str,
        operation_id: Optional[str] = None,
        path: Optional[str] = None,
        method: str = "POST",
        path_parameters: Optional[dict[str, str]] = None,
        query: Optional[dict[str, Any]] = None,
        input: Optional[dict[str, Any]] = None,  # noqa: A002 - matches API field
        connection_id: Optional[str] = None,
        approval_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        """Invoke a managed service operation.

        Provide either ``operation_id`` (short, e.g. ``memory.create``, or the
        full ``services.invoke.<service>.<op>`` id from /actions or the
        catalog) or a raw ``path`` (+ method) for path-style capabilities.
        """
        body: dict[str, Any] = {"method": method}
        if operation_id:
            body["operationId"] = operation_id
        if path:
            body["path"] = path
        if path_parameters:
            body["pathParameters"] = path_parameters
        if query:
            body["query"] = query
        if input is not None:
            body["input"] = input
        if connection_id:
            body["connectionId"] = connection_id
        if approval_id:
            body["approvalId"] = approval_id
        return self.post(
            f"/services/{service_id}/invoke",
            json=body,
            idempotency_key=idempotency_key,
        )

    def credit_balance(self) -> dict[str, Any]:
        """Read the account credit balance."""
        data = self.get("/credits/balance")
        return data.get("credits", data) if isinstance(data, dict) else data

    def pricing(self) -> Any:
        """Read server-owned service pricing."""
        return self.get("/pricing")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SuperAgentClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
