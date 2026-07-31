"""Alembic migration versions package.

Each migration module is prefixed with an underscore so it is a valid
Python identifier (Alembic file names start with a digit).  Import them
here so the package is easily importable in tests.
"""
from app.db.migrations.versions import _0002_add_auth_fields_to_users  # noqa: F401
