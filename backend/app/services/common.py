"""Shared service-layer helpers.

Trimmed from PCC's: conductor has no trash/restore views or purge, so only the
soft-delete filter and marker are needed.
"""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy import Select, select

from app.db.models import SoftDeleteMixin, utcnow

ModelT = TypeVar("ModelT", bound=SoftDeleteMixin)


def active(model: type[ModelT]) -> Select[tuple[ModelT]]:
    """Select non-soft-deleted rows of ``model`` (``deleted_at IS NULL``)."""
    return select(model).where(model.deleted_at.is_(None))


def soft_delete(obj: SoftDeleteMixin) -> None:
    """Mark a row deleted. Caller is responsible for committing."""
    obj.deleted_at = utcnow()
