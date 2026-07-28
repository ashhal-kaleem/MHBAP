-- Runs once when the Postgres container is first created.
-- Alembic migrations create the tables; this just enables the extension
-- so create_hypertable() calls in the migration succeed.
CREATE EXTENSION IF NOT EXISTS timescaledb;
