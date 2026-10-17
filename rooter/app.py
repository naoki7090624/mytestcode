
import ipaddress
import os

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask


# ============================================================
# Configuration: values are supplied by compose.yaml
# ============================================================

LITELLM_URL = os.environ.get(
    "LITELLM_URL", "http://litellm:4000"
).rstrip("/")

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")

MOE_MODEL_NAME = os.environ["MOE_MODEL_NAME"]
DENSE_MODEL_NAME = os.environ["DENSE_MODEL_NAME"]
DENSE_A_MODEL_NAME = os.environ["DENSE_A_MODEL_NAME"]
DENSE_B_MODEL_NAME = os.environ["DENSE_B_MODEL_NAME"]

app = FastAPI(title="LLM Router")


# ============================================================
# HTTP client lifecycle
# ============================================================

@app.on_event("startup")
async def startup():
    app.state.http_client = httpx.AsyncClient(
        timeout=None,
        follow_redirects=False,
    )


@app.on_event("shutdown")
async def shutdown():
    await app.state.http_client.aclose()


# ============================================================
# Routing helpers
# ============================================================

def get_client_ipv4(request: Request) -> ipaddress.IPv4Address:
    """Return the client's IPv4 address."""

    if request.client is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot determine client IP address",
        )

    host = request.client.host

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Client IP address is not a valid IP address",
        )

    if isinstance(address, ipaddress.IPv6Address):
        # Support IPv4-mapped IPv6 addresses, if present.
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped

        raise HTTPException(
            status_code=400,
            detail="IPv4 client address is required for Dense routing",
        )

    return address


def select_model(request: Request, requested_model: str) -> str:
    """
    Map the public model name to the model name registered in LiteLLM.

    local-moe   -> local-moe
    local-dense -> dense-a or dense-b, selected by client IPv4 parity
    """

    if requested_model == MOE_MODEL_NAME:
        return MOE_MODEL_NAME

    if requested_model == DENSE_MODEL_NAME:
        client_ip = get_client_ipv4(request)

        if int(str(client_ip).split(".")[-1]) % 2 == 0:
            return DENSE_A_MODEL_NAME

        return DENSE_B_MODEL_NAME

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{requested_model}'. "
            f"Use '{MOE_MODEL_NAME}' or '{DENSE_MODEL_NAME}'."
        ),
    )


# Headers that must not be forwarded hop-by-hop.
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def filtered_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


# ============================================================
# Proxy requests to LiteLLM
# ============================================================

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_request(path: str, request: Request):
    body = await request.body()
    headers = filtered_headers(request.headers)

    # Use the configured key for the Router -> LiteLLM connection.
    # If no key is configured, preserve the incoming Authorization header.
    if LITELLM_API_KEY:
        headers["authorization"] = f"Bearer {LITELLM_API_KEY}"

    # Route chat completion requests according to the requested model.
    if path == "v1/chat/completions" and body:
        try:
            import json
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Request body must be valid JSON",
            )

        requested_model = payload.get("model")

        if not isinstance(requested_model, str):
            raise HTTPException(
                status_code=400,
                detail="A valid 'model' field is required",
            )

        payload["model"] = select_model(request, requested_model)

        import json
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"

    # Preserve the incoming query string.
    upstream_url = f"{LITELLM_URL}/{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    upstream_request = app.state.http_client.build_request(
        method=request.method,
        url=upstream_url,
        headers=headers,
        content=body,
    )

    try:
        upstream_response = await app.state.http_client.send(
            upstream_request,
            stream=True,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to LiteLLM: {exc}",
        )

    response_headers = filtered_headers(upstream_response.headers)

    # Keep streaming responses streaming (e.g. stream=true).
    if "text/event-stream" in upstream_response.headers.get(
        "content-type", ""
    ).lower():
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_response.aclose),
        )

    # For non-streaming responses, read the complete body and close upstream.
    try:
        response_body = await upstream_response.aread()
        return Response(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )
    finally:
        await upstream_response.aclose()
