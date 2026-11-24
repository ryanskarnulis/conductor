"""A card's tap in tests (#251): the one way anything is approved."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def cards(client: TestClient, conversation: int) -> list[dict[str, Any]]:
    """The cards waiting on the person, oldest first."""
    response = client.get(f"/api/agent/conversations/{conversation}/cards")
    assert response.status_code == 200, response.text
    return list(response.json())


def tap(
    client: TestClient,
    conversation: int,
    *,
    answer: str = "yes",
    option: int | None = None,
    which: int | str = -1,
    status: int = 200,
) -> dict[str, Any]:
    """Tap a card: the newest by default, the one at ``which``, or every open
    one (``"all"``: a group's "Approve all")."""
    waiting = cards(client, conversation)
    assert waiting, "no card is waiting"
    ids = [card["id"] for card in waiting] if which == "all" else [waiting[int(which)]["id"]]
    body: dict[str, Any] = {"cards": ids, "answer": answer}
    if option is not None:
        body["option"] = option
    response = client.post(f"/api/agent/conversations/{conversation}/cards", json=body)
    assert response.status_code == status, response.text
    return dict(response.json())


def reply_of(tapped: dict[str, Any]) -> dict[str, Any]:
    """The assistant message of the turn a tap resumed."""
    (exchange,) = tapped["exchanges"]
    return dict(exchange["assistant_message"])
