"""add hashed_password, display_name, is_active to users

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add hashed_password — empty default so existing rows are not rejected
    op.add_column(
        "users",
        sa.Column("hashed_password", sa.String(128), nullable=False, server_default=""),
    )
    # Remove server_default now that backfill is done (keeps column clean going forward)
    op.alter_column("users", "hashed_password", server_default=None)

    # Add display_name — defaults to empty string
    op.add_column(
        "users",
        sa.Column("display_name", sa.String(128), nullable=False, server_default=""),
    )
    op.alter_column("users", "display_name", server_default=None)

    # Add is_active flag — defaults to TRUE so existing accounts stay active
    op.add_column(
        "users",
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "is_active")
    op.drop_column("users", "display_name")
    op.drop_column("users", "hashed_password")
