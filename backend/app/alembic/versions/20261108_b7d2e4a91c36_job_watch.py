"""job watch

Jobs (#76): the watcher's columns on ``ledger_jobs``. ``title``/``untrusted``
are the job's name as last reported, ``polled_at`` its last poll, and
``noted_at``/``note_message_id`` the one note its conversation got when it
ended (the idempotency marker). Additive. Jobs already on the ledger are
marked noted: only jobs started after this revision get a note, so an upgrade
never posts notes for old work.

Revision ID: b7d2e4a91c36
Revises: 5189491c0978
Create Date: 2026-11-08 09:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7d2e4a91c36"
down_revision: Union[str, None] = "5189491c0978"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ledger_jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("title", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column("untrusted", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(sa.Column("polled_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("noted_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("note_message_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_ledger_jobs_noted_at"), ["noted_at"], unique=False)
        batch_op.create_foreign_key(
            "fk_ledger_jobs_note_message_id_conversation_messages",
            "conversation_messages",
            ["note_message_id"],
            ["id"],
        )
    op.execute("UPDATE ledger_jobs SET noted_at = updated_at WHERE noted_at IS NULL")


def downgrade() -> None:
    with op.batch_alter_table("ledger_jobs", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_ledger_jobs_note_message_id_conversation_messages", type_="foreignkey"
        )
        batch_op.drop_index(batch_op.f("ix_ledger_jobs_noted_at"))
        batch_op.drop_column("note_message_id")
        batch_op.drop_column("noted_at")
        batch_op.drop_column("polled_at")
        batch_op.drop_column("untrusted")
        batch_op.drop_column("title")
