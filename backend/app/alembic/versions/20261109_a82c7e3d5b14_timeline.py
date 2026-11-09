"""timeline

One timeline and incognito (#82). Additive:

- ``conversations.kind``: ``timeline``, ``thread`` or ``legacy``. Every
  existing row becomes ``legacy`` (reachable only through recall, #58 (c));
  the timeline row is created on first use and starts clean.
- ``conversations.carries``, ``closed_at``, ``result``: a side thread's
  columns (#139), nullable and unused for now.
- ``ledger_turns.incognito``: a turn whose words and reply were never
  written; existing turns are not.

Revision ID: a82c7e3d5b14
Revises: f4c2a8d6e913
Create Date: 2026-11-09 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a82c7e3d5b14"
down_revision: Union[str, None] = "f4c2a8d6e913"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("kind", sa.String(), server_default="legacy", nullable=False))
        batch_op.add_column(sa.Column("carries", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("closed_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("result", sa.String(), nullable=True))
        batch_op.create_index(batch_op.f("ix_conversations_kind"), ["kind"], unique=False)
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("incognito", sa.Boolean(), server_default="0", nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.drop_column("incognito")
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_conversations_kind"))
        batch_op.drop_column("result")
        batch_op.drop_column("closed_at")
        batch_op.drop_column("carries")
        batch_op.drop_column("kind")
