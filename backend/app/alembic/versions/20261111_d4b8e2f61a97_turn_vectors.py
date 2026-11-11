"""turn vectors

Recall by meaning (#147). Additive:

- ``ledger_turn_vectors``: one row per indexed turn, its vector from the
  shared embeddings service (little-endian float32) and the model that made
  it. Rows are written by the app as turns close; existing turns are covered
  by ``scripts/embed_turns.py`` (idempotent), so this revision stays plain DDL.

Revision ID: d4b8e2f61a97
Revises: a82c7e3d5b14
Create Date: 2026-11-11 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4b8e2f61a97"
down_revision: Union[str, None] = "a82c7e3d5b14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ledger_turn_vectors",
        sa.Column("turn_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
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
        sa.ForeignKeyConstraint(["turn_id"], ["ledger_turns.id"]),
        sa.PrimaryKeyConstraint("turn_id"),
    )


def downgrade() -> None:
    op.drop_table("ledger_turn_vectors")
