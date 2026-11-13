"""heard

Barge-in (#77): ``ledger_turns.heard`` is how much of a turn's spoken reply
the person heard before they spoke over it, as the voice client reported it
with the next turn. Additive: a nullable JSON column; existing turns read as
heard to the end.

Revision ID: bd521652ab2a
Revises: d4b8e2f61a97
Create Date: 2026-11-12 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "bd521652ab2a"
down_revision: Union[str, None] = "d4b8e2f61a97"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.add_column(sa.Column("heard", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("ledger_turns", schema=None) as batch_op:
        batch_op.drop_column("heard")
