"""thread per ask

#262: every ask but an answer to the app's own question opens a fresh app
thread, so ``delegate_threads`` holds one row per thread conductor opened
and hasn't deleted yet, many per window and app. The unique
``(master_conversation_id, app_name)`` goes; rows and the index stay.

Revision ID: c3f1a9d27e5b
Revises: bd521652ab2a
Create Date: 2026-11-24 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "c3f1a9d27e5b"
down_revision: Union[str, None] = "bd521652ab2a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The constraint was created unnamed; batch mode names it by this convention
# when it reflects the table, so it can be dropped by that name.
_NAMING = {"uq": "uq_%(table_name)s_%(column_0_name)s"}
_UNIQUE = "uq_delegate_threads_master_conversation_id"


def upgrade() -> None:
    with op.batch_alter_table(
        "delegate_threads", schema=None, naming_convention=_NAMING
    ) as batch_op:
        batch_op.drop_constraint(_UNIQUE, type_="unique")


def downgrade() -> None:
    # Back to one row per window and app: keep each pair's newest thread.
    op.execute(
        "DELETE FROM delegate_threads WHERE id NOT IN "
        "(SELECT MAX(id) FROM delegate_threads GROUP BY master_conversation_id, app_name)"
    )
    with op.batch_alter_table(
        "delegate_threads", schema=None, naming_convention=_NAMING
    ) as batch_op:
        batch_op.create_unique_constraint(_UNIQUE, ["master_conversation_id", "app_name"])
