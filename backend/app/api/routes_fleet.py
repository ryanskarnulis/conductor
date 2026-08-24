"""One narrow proxy: a page in conductor's UI acting on a fleet app directly.

Most of what conductor does goes through the model — that is the point of it.
This route is the exception, and it exists for one shape of work: a person
answering the *same question* dozens of times. Music's sorting pass is ~150
questions, and a click that waits for a local 12B to read a sentence and re-emit
it as a tool call is a slow way of typing, not a button
(``../future-plans/music-agent.md``, Phase 2.6).

**Why proxy at all, rather than let the page dial the app?** Because that makes
the app writable by another origin. The browser would need CORS on music, music
would become the first app in the fleet a foreign page can write to, and every
app that ever wants a panel would follow. Proxying keeps the page same-origin
with conductor and leaves the apps exactly as headless as they were.

**Only what an app declares.** ``agent.actions`` in the manifest names one path
prefix, and nothing outside it is reachable: a generic "forward anything to any
app" route is an SSRF surface with a friendly name. An app without the key has
no proxy at all, which is every app but one today.

Deliberately not the delegate client: that speaks the delegate contract with a
300 s read timeout for cold model loads. Nothing here waits on a model, so a
button that hangs a page for five minutes would be a bug rather than patience.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.rate_limit import rate_limit
from app.fleet.manifests import Fleet, FleetApp

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/fleet", tags=["fleet"])

# No model in the path, so these are local HTTP calls that either answer at once
# or have failed. A person is watching a button.
_CONNECT_TIMEOUT = 5.0
_READ_TIMEOUT = 30.0

# A page's action is a small JSON body; anything larger is not one.
_MAX_BODY_BYTES = 64 * 1024

# What conductor forwards *to* the app. Not the browser's headers — a proxy that
# passes those on hands one origin's cookies and auth to another service.
_FORWARDED_CONTENT_TYPE = "application/json"


async def get_action_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """The HTTP client the proxy forwards with; tests override this dependency."""
    timeout = httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield client


def get_fleet(request: Request) -> Fleet:
    """The fleet discovered at startup; tests override this dependency."""
    fleet: Fleet | None = getattr(request.app.state, "fleet", None)
    if fleet is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the fleet has not been discovered yet",
        )
    return fleet


def _app_with_actions(fleet: Fleet, name: str) -> FleetApp:
    """The named app, if it exists *and* opened a prefix to be proxied.

    One 404 for both cases on purpose: from the browser's side "no such app" and
    "that app publishes nothing to act on" are the same answer, and telling them
    apart only maps the fleet for whoever is asking.
    """
    app = fleet.get(name)
    if app is None or app.agent is None or not app.agent.actions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no app named {name!r} publishes actions",
        )
    return app


def _target(app: FleetApp, path: str) -> str:
    """The URL to call, or a 400 if the path tries to leave the declared prefix.

    ``path`` arrives from the browser, so it is checked rather than trusted:
    every segment must be an ordinary name. ``..`` is the whole attack — httpx
    would happily normalize ``/api/sorting/../agent/conversations`` into a
    surface the app never opened.
    """
    trimmed = path.strip("/")
    segments = [segment for segment in trimmed.split("/") if segment]
    if any(segment == ".." for segment in segments):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="an action path cannot climb out of the declared prefix",
        )
    base = app.actions_base_url
    return f"{base}/{'/'.join(segments)}" if segments else base


@router.api_route(
    "/{app_name}/actions/{path:path}",
    methods=["GET", "POST"],
    dependencies=[Depends(rate_limit("fleet_actions", per_min_attr="fleet_actions_per_min"))],
)
async def proxy_action(
    app_name: str,
    path: str,
    request: Request,
    fleet: Fleet = Depends(get_fleet),
    client: httpx.AsyncClient = Depends(get_action_client),
) -> Response:
    """Forward one call to an app's declared actions prefix and hand back its answer.

    The app's status code and body pass through untouched: a 422 saying "that
    artist is already sorted" is the app's answer to the person, and rewriting
    it here would only make the page guess at what happened.
    """
    app = _app_with_actions(fleet, app_name)
    url = _target(app, path)
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="that is too large to be a page's action",
        )

    try:
        upstream = await client.request(
            request.method,
            url,
            params=dict(request.query_params),
            content=body or None,
            headers={"content-type": _FORWARDED_CONTENT_TYPE} if body else None,
        )
    except httpx.HTTPError as exc:
        logger.warning("fleet_action_unreachable", app=app_name, path=path, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{app_name} did not answer",
        ) from exc

    logger.info(
        "fleet_action",
        app=app_name,
        method=request.method,
        path=path,
        upstream_status=upstream.status_code,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
