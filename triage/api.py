"""The web API. One endpoint routes a message; the provider is loaded once at start-up."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import get_settings
from .providers import Provider, build_provider
from .service import route


class RouteRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class RouteResponse(BaseModel):
    category: str | None
    intent: str | None
    order_number: str | None
    needs_review: bool
    problems: list[str]
    raw_output: str
    provider: str
    latency_ms: int


def create_app(provider: Provider | None = None) -> FastAPI:
    """Passing a provider lets the tests run the API without loading a model."""
    app = FastAPI(title="Support message triage", version="1.0")
    state: dict[str, Provider] = {}

    def get_provider() -> Provider:
        if "provider" not in state:
            state["provider"] = provider or build_provider(get_settings())
        return state["provider"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "provider": get_provider().name}

    @app.post("/route", response_model=RouteResponse)
    def route_message(request: RouteRequest) -> RouteResponse:
        routed = asdict(route(get_provider(), request.message))
        routed.pop("message")
        return RouteResponse(**routed)

    return app


app = create_app()
