"""Recall by meaning (#147): the stored vectors, the hybrid ranking, the bar,
and the service being optional.

Over ledger rows and a fake embedder (``tests/fake_embedder.py``): CI has no
embeddings service. The route-level path (a turn embedded as it closes, a
match found by meaning) is in ``test_memory.py``; the real service's numbers
behind :data:`app.ai.memory.LIFT_ALONE` are ``test_recall_cases_live.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

from app.ai import incognito, memory
from app.ai.embeddings import (
    QUERY_TIMEOUT,
    Embedder,
    EmbeddingsUnavailable,
    document_text,
    query_text,
)
from app.db.models import Conversation, LedgerTurn, LedgerTurnVector
from app.services import memory as memory_service
from app.services import turn_vectors
from tests.fake_embedder import FakeEmbedder, use

CORPUS = (
    "for the garage shelves, let's go with the steel rack",
    "the fence needs two coats of paint",
    "move my dentist appointment to Thursday",
    "add milk and eggs to the shopping list",
    "start a chess game against the computer",
    "make a playlist for the road trip",
)
# The paraphrases the fake knows: "carport storage" means "garage shelves".
SAME = {"carport": "garage", "storage": "shelve", "tooth": "dentist", "doctor": "appointment"}


def _seed(db: Session, said: Sequence[str] = CORPUS, **flags: object) -> list[LedgerTurn]:
    conversation = Conversation(title="old")
    db.add(conversation)
    db.flush()
    turns = [
        LedgerTurn(conversation_id=conversation.id, words=words, status="ended", **flags)
        for words in said
    ]
    db.add_all(turns)
    db.flush()
    memory_service.index_pending(db)
    return turns


def _embed_all(db: Session) -> int | None:
    return memory_service.embed_pending(db, limit=100, oldest_first=True, patient=True)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    return use(monkeypatch, FakeEmbedder(SAME))


# --- the pure pieces --------------------------------------------------------------------------


def test_lift_is_over_the_median_and_fusion_only_orders() -> None:
    lifted = memory.lifts({1: 0.60, 2: 0.58, 3: 0.80, 4: 0.59, 5: 0.57})
    assert round(lifted[3], 3) == 0.21 and round(lifted[4], 3) == 0.0
    fused = memory.fuse([1, 2], [2, 3])
    assert fused[2] > fused[1] > fused[3]  # in both rankings beats first in one
    assert memory.meaning_text("grab  Odessa", ["Odessa", "", " "]) == "grab Odessa. Odessa"
    assert memory.meaning_text("", ["Odessa", "Sun"]) == "Odessa, Sun"


def test_vectors_are_stored_as_float32_and_read_back() -> None:
    blob = turn_vectors.pack([0.5, -0.25, 1.0])
    assert len(blob) == 12
    assert list(turn_vectors.unpack(blob)) == [0.5, -0.25, 1.0]


def test_the_client_turns_every_failure_into_unavailable() -> None:
    def serve(answer: httpx.Response | Exception) -> Embedder:
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(answer, Exception):
                raise answer
            assert json.loads(request.content) == {"input": [query_text("hi")]}
            return answer

        return Embedder("http://embed/v1/", httpx.Client(transport=httpx.MockTransport(handler)))

    good = {
        "model": "gemma",
        "data": [{"index": 0, "embedding": [0.6, 0.8]}],
    }
    assert serve(httpx.Response(200, json=good)).embed(
        [query_text("hi")], timeout=QUERY_TIMEOUT
    ).vectors == [[0.6, 0.8]]
    for broken in (
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"model": "gemma", "data": []}),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
    ):
        with pytest.raises(EmbeddingsUnavailable):
            serve(broken).embed([query_text("hi")], timeout=QUERY_TIMEOUT)
    assert query_text("a  b") == "task: search result | query: a b"
    assert document_text("a\nb") == "title: none | text: a b"


def test_off_by_switch_means_no_client() -> None:
    # conftest switches it off: CI has no service, and a dev box's is never dialled.
    assert turn_vectors.embedder() is None


# --- recall, ranked together ------------------------------------------------------------------


def test_a_paraphrase_is_found_by_meaning_and_words_alone_miss_it(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    assert memory_service.recall(db_session, "carport storage").hits == ()
    fake = use(monkeypatch, FakeEmbedder(SAME))
    assert _embed_all(db_session) == len(CORPUS)
    found = memory_service.recall(db_session, "carport storage")
    assert [hit.words for hit in found.hits] == [CORPUS[0]]
    assert fake.sent[-1] == query_text("carport storage")


def test_an_unrelated_question_finds_nothing_never_a_guess(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session)
    _embed_all(db_session)
    with capture_logs() as logs:
        found = memory_service.recall(db_session, "boat trailer hitch")
    assert found.hits == () and fake.calls == 2  # the corpus, then the query
    assert memory.record(found, date(2026, 11, 11))[0] == "Nothing on record matches."
    assert not [log for log in logs if log["event"] == "memory_recalled"]


def test_a_word_hit_still_counts_and_meaning_orders_with_it(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session)
    _embed_all(db_session)
    # "steel" is said once; its meaning is the same turn: found, first.
    found = memory_service.recall(db_session, "steel shelves for the carport")
    assert found.hits[0].words == CORPUS[0]
    with capture_logs() as logs:
        memory_service.recall(db_session, "the fence paint")
    (log,) = [log for log in logs if log["event"] == "memory_recalled"]
    assert log["source"] == "hybrid" and log["by_meaning_alone"] == 0


def test_a_hit_by_meaning_alone_never_names_what_a_step_passes_on(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session)
    _embed_all(db_session)
    found = memory_service.recall(db_session, "tooth doctor")
    assert [hit.words for hit in found.hits] == [CORPUS[2]]
    assert found.best is None  # an embedding ranks; it never decides {sN}


# --- the service is optional ------------------------------------------------------------------


def test_a_service_that_is_down_costs_meaning_never_the_recall(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session)
    _embed_all(db_session)
    fake.down = "ConnectError: refused"
    with capture_logs() as logs:
        words = memory_service.recall(db_session, "garage shelves")
        missed = memory_service.recall(db_session, "carport storage")
    assert [hit.words for hit in words.hits] == [CORPUS[0]]  # word overlap alone
    assert missed.hits == ()
    down = [log for log in logs if log["event"] == "memory_embeddings_unavailable"]
    assert len(down) == 1 and "refused" in down[0]["reason"]
    assert fake.calls == 2  # the second recall left the service alone
    recalled = [log for log in logs if log["event"] == "memory_recalled"]
    assert recalled[0]["source"] == "words"
    # It comes back: after the rest, meaning is used again.
    turn_vectors._reset()
    fake.down = None
    assert memory_service.recall(db_session, "carport storage").hits


def test_indexing_that_fails_is_retried_later(db_session: Session, fake: FakeEmbedder) -> None:
    _seed(db_session)
    fake.down = "ReadTimeout: slow"
    assert memory_service.embed_pending(db_session) is None
    assert db_session.execute(select(LedgerTurnVector)).first() is None
    turn_vectors._reset()
    fake.down = None
    # A later close embeds the newest few; the backfill the rest, once.
    assert memory_service.embed_pending(db_session, limit=2) == 2
    assert _embed_all(db_session) == len(CORPUS) - 2
    assert _embed_all(db_session) == 0


def test_too_few_turns_or_another_model_means_words_alone(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session, CORPUS[:3])
    _embed_all(db_session)
    calls = fake.calls
    assert turn_vectors.cosines(db_session, "carport storage") is None
    assert fake.calls == calls  # too few to say anything: the service isn't asked
    _seed(db_session, CORPUS[3:])
    _embed_all(db_session)
    fake.model = "another-model"
    with capture_logs() as logs:
        assert turn_vectors.cosines(db_session, "carport storage") is None
    assert [log["event"] for log in logs] == ["memory_embeddings_stale"]
    assert turn_vectors.drop_other_models(db_session, "another-model") == len(CORPUS)
    assert _embed_all(db_session) == len(CORPUS)
    assert turn_vectors.cosines(db_session, "carport storage") is not None


# --- privacy ----------------------------------------------------------------------------------


def test_incognito_and_forgotten_turns_are_never_sent(
    db_session: Session, fake: FakeEmbedder
) -> None:
    turns = _seed(db_session)
    _seed(db_session, ["the gate code is 4417"], incognito=True)
    forgotten = _seed(db_session, ["the safe combination is 2219"])[0]
    memory_service.unindex(db_session, forgotten.id)
    assert _embed_all(db_session) == len(CORPUS)
    assert not fake.said("4417") and not fake.said("2219")
    # Forgetting an embedded turn deletes its vector.
    assert db_session.get(LedgerTurnVector, turns[0].id) is not None
    memory_service.unindex(db_session, turns[0].id)
    assert db_session.get(LedgerTurnVector, turns[0].id) is None
    assert memory_service.recall(db_session, "carport storage").hits == ()


def test_an_incognito_turns_words_are_never_a_query(
    db_session: Session, fake: FakeEmbedder
) -> None:
    _seed(db_session)
    _embed_all(db_session)
    calls = fake.calls
    with incognito.turn(True):
        assert memory_service.recall(db_session, "carport storage").hits == ()
        conversation = Conversation(title="now")
        db_session.add(conversation)
        db_session.flush()
        assert memory_service.matches(db_session, "that carport storage", conversation.id) == []
    assert fake.calls == calls


def test_words_asking_to_forget_are_never_a_query(db_session: Session, fake: FakeEmbedder) -> None:
    _seed(db_session)
    _embed_all(db_session)
    conversation = db_session.execute(select(Conversation)).scalars().first()
    assert conversation is not None
    memory_service.matches(db_session, "forget that the gate code is 4417", conversation.id + 1)
    memory_service.recall(db_session, "forget the gate code 4417")
    assert not fake.said("4417")
