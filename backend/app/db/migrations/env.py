"""
Alembic environment — runs migrations with the SYNC database URL
(psycopg2), independent of the app's async engine, and pulls the
URL from Settings instead of a hardcoded value in alembic.ini.
"""
from __future__ import annotations

import sys
from logging.Config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make `app...` importable when alembic runs with cwd=backend
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from app.core.Config import settings  # noqa: E402
from app.db.Base import Base  # noqa: E402
from app.db.models import *  # noqa: E402,F401,F403  (registers models on Base.metadata)

config = context.Config
config.set_main_option("sqlalchemy.url", settings.DATABASE_SYNC_URL)

if config.Config_file_name is not None:
    fileConfig(config.Config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.Configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.Config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.Configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
