"""initial schema: users, sessions, modality_features, predictions

Revision ID: 0001
Revises:
Create Date: 2026-07-28
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="participant"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("context", sa.String(64), nullable=False, server_default="unspecified"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("consent_recording", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("faces_blurred", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("session_metadata", JSONB, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "modality_features",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), primary_key=True,
                  server_default=sa.func.now()),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id"),
                  primary_key=True),
        sa.Column("modality", sa.String(32), primary_key=True),
        sa.Column("feature_vector", JSONB, nullable=False),
        sa.Column("quality", sa.Float, nullable=False, server_default="1.0"),
    )
    op.create_index(
        "ix_modality_features_session_time", "modality_features", ["session_id", "time"]
    )
    # Convert to a TimescaleDB hypertable partitioned on `time`.
    # migrate_data=True is safe here since the table is freshly created.
    op.execute(
        "SELECT create_hypertable('modality_features', 'time', "
        "if_not_exists => TRUE, migrate_data => TRUE);"
    )

    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), primary_key=True,
                  server_default=sa.func.now()),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id"),
                  primary_key=True),
        sa.Column("emotion_label", sa.String(32), nullable=False),
        sa.Column("emotion_scores", JSONB, nullable=False),
        sa.Column("stress", sa.Float, nullable=False),
        sa.Column("engagement", sa.Float, nullable=False),
        sa.Column("attention", sa.Float, nullable=False),
        sa.Column("fatigue", sa.Float, nullable=False),
        sa.Column("shap_weights", JSONB, nullable=False, server_default="{}"),
        sa.Column("explanation_text", sa.String(512), nullable=False, server_default=""),
    )
    op.create_index("ix_predictions_session_time", "predictions", ["session_id", "time"])
    op.execute(
        "SELECT create_hypertable('predictions', 'time', "
        "if_not_exists => TRUE, migrate_data => TRUE);"
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("modality_features")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
