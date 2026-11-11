"""Embed every indexed turn that has no vector yet (#147): the one-time backfill.

    python scripts/embed_turns.py          # from backend/, against DATABASE_URL
    docker compose exec backend python scripts/embed_turns.py

Idempotent: a turn with a vector is skipped, so a rerun embeds only what a
closing turn couldn't (the service was down) and exits 0 with nothing to do.
Indexes any turn not yet indexed first, oldest first, as a closing turn would.
Never sends an incognito or forgotten turn. If the service now runs another
model, the old vectors are dropped and every turn is embedded again. Exits 1
when the service doesn't answer, 2 when CONDUCTOR_EMBEDDINGS is off.
"""

from __future__ import annotations

import sys

from app.ai.embeddings import BATCH_TIMEOUT, query_text
from app.config import get_settings
from app.db.session import SessionLocal
from app.services import memory, turn_vectors

# Turns per commit.
CHUNK = 64


def main() -> int:
    if not get_settings().conductor_embeddings:
        print("CONDUCTOR_EMBEDDINGS is off: nothing to do", file=sys.stderr)
        return 2
    probe = turn_vectors.embed([query_text("probe")], timeout=BATCH_TIMEOUT, patient=True)
    if probe is None:
        print(f"no answer from {get_settings().embeddings_base_url}", file=sys.stderr)
        return 1
    with SessionLocal() as db:
        indexed = 0
        while batch := memory.index_pending(db):
            indexed += batch
            db.commit()
        dropped = turn_vectors.drop_other_models(db, probe.model)
        db.commit()
        embedded = 0
        while True:
            done = memory.embed_pending(db, limit=CHUNK, oldest_first=True, patient=True)
            if done is None:
                print(f"the service stopped answering after {embedded} turns", file=sys.stderr)
                return 1
            if done == 0:
                break
            db.commit()
            embedded += done
    print(
        f"indexed {indexed}, dropped {dropped} vectors from another model, embedded {embedded} "
        f"turns ({probe.model})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
