"""The recall-by-meaning cases (#147, ``tests/recall_cases.py``).

In CI: the data holds (no query shares a memory word with any turn, so word
overlap finds nothing and every hit is meaning's). Opt-in,
``CONDUCTOR_EMBEDDINGS_LIVE=1``: the real recall over those turns against the
embeddings service on 8600 (CPU only, no model server), printing each query's
lift; every control must find nothing.

    CONDUCTOR_EMBEDDINGS_LIVE=1 pytest tests/test_recall_cases.py -v -s
"""

from __future__ import annotations

import os
import statistics

import pytest
from sqlalchemy.orm import Session

from app.ai import memory
from app.config import get_settings
from app.db.models import Conversation, LedgerTurn
from app.services import memory as memory_service
from app.services import turn_vectors
from tests.recall_cases import CONTROLS, PARAPHRASES, TURNS

LIVE = os.environ.get("CONDUCTOR_EMBEDDINGS_LIVE") == "1"


def test_no_query_shares_a_word_with_any_turn() -> None:
    said = set().union(*(memory.terms(turn) for turn in TURNS))
    for query in [query for query, _ in PARAPHRASES] + list(CONTROLS):
        wanted = memory.terms(query)
        assert wanted, query
        assert not wanted & said, f"{query!r} shares {sorted(wanted & said)}"
    assert all(0 <= target < len(TURNS) for _, target in PARAPHRASES)


@pytest.mark.skipif(not LIVE, reason="CONDUCTOR_EMBEDDINGS_LIVE=1: the embeddings service on 8600")
def test_the_bar_against_the_real_service(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "conductor_embeddings", True)
    turn_vectors._reset()
    conversation = Conversation(title="household")
    db_session.add(conversation)
    db_session.flush()
    turns = [
        LedgerTurn(conversation_id=conversation.id, words=words, status="ended") for words in TURNS
    ]
    db_session.add_all(turns)
    db_session.flush()
    memory_service.index_pending(db_session)
    embedded = memory_service.embed_pending(db_session, limit=100, oldest_first=True, patient=True)
    assert embedded == len(TURNS), "the embeddings service on 8600 didn't answer"

    def lifts(query: str) -> dict[int, float]:
        cosines = turn_vectors.cosines(db_session, query)
        assert cosines is not None, query
        return memory.lifts(cosines)

    found = 0
    target_lifts: list[float] = []
    print(f"\nbar: lift >= {memory.LIFT_ALONE} over the median cosine ({len(TURNS)} turns)")
    print("paraphrases: target lift, its rank, the best other turn's lift, cited?")
    for query, target in PARAPHRASES:
        lifted = lifts(query)
        mine = lifted[turns[target].id]
        rank = 1 + sum(1 for lift in lifted.values() if lift > mine)
        other = max(lift for turn_id, lift in lifted.items() if turn_id != turns[target].id)
        cited = [hit.turn_id for hit in memory_service.recall(db_session, query).hits]
        hit = turns[target].id in cited
        found += hit
        target_lifts.append(mine)
        print(
            f"  {mine:+.3f} #{rank:<2} other {other:+.3f} {'cited' if hit else 'missed':6} "
            f"{len(cited)} hit(s)  {query}"
        )
    control_lifts: list[float] = []
    wrong: list[str] = []
    print("controls: the best turn's lift, what recall cited")
    for query in CONTROLS:
        lifted = lifts(query)
        best = max(lifted, key=lambda turn_id: lifted[turn_id])
        guessed = memory_service.recall(db_session, query).hits
        control_lifts.append(lifted[best])
        if guessed:
            wrong.append(query)
        said = next(turn.words for turn in turns if turn.id == best)
        print(f"  {lifted[best]:+.3f} {len(guessed)} hit(s)  {query!r} (nearest: {said!r})")
    print(
        f"paraphrases cited {found}/{len(PARAPHRASES)}; target lifts "
        f"min {min(target_lifts):+.3f} median {statistics.median(target_lifts):+.3f}; "
        f"controls cited {len(wrong)}/{len(CONTROLS)}, best lift max {max(control_lifts):+.3f}"
    )
    assert not wrong, f"recall guessed for {wrong}"
    # The bar trades recall for honesty: a third of the paraphrases is the floor.
    assert 3 * found >= len(PARAPHRASES)
