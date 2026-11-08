"""memory facts

Memory you can see (#81). Additive:

- ``memory_facts``: what the person asked conductor to remember, in their own
  words, with the turn it came from; soft-deleted when they forget it.
- ``ledger_turns.forgotten_at``: a turn whose fact was forgotten (or that
  forgot one), out of memory's index, recall and every model replay.

Revision ID: f4c2a8d6e913
Revises: e3a7c5d1f804
Create Date: 2026-11-08 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4c2a8d6e913"
down_revision: Union[str, None] = "e3a7c5d1f804"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("source_turn_id", sa.Integer(), nullable=True),
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
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_turn_id"],
            ["ledger_turns.id"],
            name="fk_memory_facts_source_turn_id_ledger_turns",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("memory_facts", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_memory_facts_source_turn_id"), ["source_turn_id"], unique=False
        )
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.add_column(sa.Column("forgotten_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.drop_column("forgotten_at")
    with op.batch_alter_table("memory_facts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_memory_facts_source_turn_id"))
    op.drop_table("memory_facts")
