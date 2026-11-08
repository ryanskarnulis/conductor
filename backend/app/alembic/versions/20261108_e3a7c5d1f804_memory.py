"""memory

Memory over the ledger (#80). Additive:

- ``ledger_topics``: quiet automatic topics, built by code from the words and
  entities turns share; never shown or managed by the person.
- ``ledger_turns.terms``/``topic_id``: the words recall matches a turn on and
  the topic it joined. Null until code indexes the turn: existing turns are
  indexed by the app (``app/services/memory.py``) as turns close, oldest
  first, so the clustering lives in one place and this revision stays plain DDL.
- ``ledger_entities.terms``: the words an entity was seen with (the registry
  that resolves "that Caribou song" across days).
- ``ledger_receipts.fact``/``cites_turn_id``: a recall's receipt, the
  recalled words quoted with the turn they come from.

Revision ID: e3a7c5d1f804
Revises: b7d2e4a91c36
Create Date: 2026-11-08 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e3a7c5d1f804"
down_revision: Union[str, None] = "b7d2e4a91c36"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ledger_topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("terms", sa.JSON(), nullable=False),
        sa.Column("turns", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.add_column(sa.Column("terms", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("topic_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_ledger_turns_topic_id"), ["topic_id"], unique=False)
        batch_op.create_foreign_key(
            "fk_ledger_turns_topic_id_ledger_topics", "ledger_topics", ["topic_id"], ["id"]
        )
    with op.batch_alter_table("ledger_entities", schema=None) as batch_op:
        batch_op.add_column(sa.Column("terms", sa.JSON(), nullable=True))
    with op.batch_alter_table("ledger_receipts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("fact", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("cites_turn_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_ledger_receipts_cites_turn_id_ledger_turns",
            "ledger_turns",
            ["cites_turn_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("ledger_receipts", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_ledger_receipts_cites_turn_id_ledger_turns", type_="foreignkey"
        )
        batch_op.drop_column("cites_turn_id")
        batch_op.drop_column("fact")
    with op.batch_alter_table("ledger_entities", schema=None) as batch_op:
        batch_op.drop_column("terms")
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.drop_constraint("fk_ledger_turns_topic_id_ledger_topics", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_ledger_turns_topic_id"))
        batch_op.drop_column("topic_id")
        batch_op.drop_column("terms")
    op.drop_table("ledger_topics")
