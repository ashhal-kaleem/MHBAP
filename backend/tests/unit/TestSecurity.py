"""Tests for security.py — JWT and password hashing."""
from __future__ import annotations

import time
from datetime import timedelta

import pytest

from app.core.Security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


# ── Password hashing ───────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("secret")
        assert h != "secret"

    def test_verify_correct(self):
        h = hash_password("hunter2")
        assert verify_password("hunter2", h) is True

    def test_verify_wrong(self):
        h = hash_password("hunter2")
        assert verify_password("wrong", h) is False

    def test_different_hashes_same_password(self):
        # bcrypt salts; two hashes must differ
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_empty_password(self):
        h = hash_password("")
        assert verify_password("", h) is True

# ── JWT creation ───────────────────────────────────────────────────────────────

class TestCreateAccessToken:
    def test_returns_three_part_token(self):
        token = create_access_token("user123")
        assert token.count(".") == 2

    def test_subject_round_trips(self):
        token = create_access_token("alice")
        payload = decode_access_token(token)
        assert payload["sub"] == "alice"

    def test_extra_claims_round_trip(self):
        token = create_access_token("bob", extra_claims={"role": "admin"})
        payload = decode_access_token(token)
        assert payload["role"] == "admin"

    def test_iat_set(self):
        before = int(time.time()) - 1
        token = create_access_token("u1")
        payload = decode_access_token(token)
        assert payload["iat"] >= before

    def test_exp_in_future(self):
        token = create_access_token("u1")
        payload = decode_access_token(token)
        assert payload["exp"] > time.time()

    def test_custom_expiry(self):
        token = create_access_token("u1", expires_delta=timedelta(seconds=3600))
        payload = decode_access_token(token)
        # exp should be ~3600 s from now
        assert payload["exp"] - time.time() > 3590


# ── JWT verification ───────────────────────────────────────────────────────────

class TestDecodeAccessToken:
    def test_tampered_signature_rejected(self):
        token = create_access_token("u1")
        parts = token.split(".")
        parts[2] = parts[2][:-3] + "AAA"
        with pytest.raises(ValueError, match="Invalid signature"):
            decode_access_token(".".join(parts))

    def test_malformed_token_rejected(self):
        with pytest.raises(ValueError, match="Malformed"):
            decode_access_token("not.a.valid.jwt.here")

    def test_expired_token_rejected(self):
        token = create_access_token("u1", expires_delta=timedelta(seconds=-1))
        with pytest.raises(ValueError, match="expired"):
            decode_access_token(token)

    def test_tampered_payload_rejected(self):
        import base64, json
        token = create_access_token("u1")
        h, b, s = token.split(".")
        payload = json.loads(base64.urlsafe_b64decode(b + "=="))
        payload["sub"] = "hacker"
        new_b = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        with pytest.raises(ValueError):
            decode_access_token(f"{h}.{new_b}.{s}")
