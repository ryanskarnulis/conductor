"""Fleet delegation: discover per-app agents and route to them over REST.

Conductor's job lives here. :mod:`app.fleet.manifests` scans the sibling apps'
``app.yaml`` manifests to learn who is in the fleet and which apps ship an
agent; :mod:`app.fleet.delegate` is the typed httpx client for the workspace
delegate contract (``../agent-standard/delegate-api.md``);
:mod:`app.fleet.context` holds the per-run delegation state (thread map + call
budget + audit); and :mod:`app.fleet.tools` turns each discovered agent into an
``ask_<app>`` tool on the shared registry, plus the local ``list_agents`` tool
and the system-prompt fleet layer.
"""
