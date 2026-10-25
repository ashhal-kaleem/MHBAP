"""Alembic migration versions package.

Migration files start with a digit (e.g. 0002_add_auth_fields_to_users.py), so
they are not importable by normal syntax. Tests load them with importlib.
There used to be an underscore-prefixed copy of 0002 here for tests, but Alembic
scans every file in this folder, so the copy created a second revision "0002"
(two heads). It was removed.
"""
