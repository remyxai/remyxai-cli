"""Tests for the secret-free key hint (code-scanning alert #2).

`remyxai/api/__init__.py` used to log `REMYXAI_API_KEY[:8]`, which put real key
material into log files — for a 41-char prefixed key that is roughly a fifth of
it, and logs travel much further than the shell that exported the key. These
tests pin the replacement: a hint that identifies *which* key is in use while
carrying none of it.
"""
import hashlib
import logging

import pytest

from remyxai.api import key_fingerprint, key_hint

# Shaped like a real key (prefix + body) but not one.
_KEY = "rmx_test_" + "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"


def test_fingerprint_is_stable():
    assert key_fingerprint(_KEY) == key_fingerprint(_KEY)


def test_fingerprint_matches_a_truncated_sha256():
    expected = hashlib.sha256(_KEY.encode()).hexdigest()[:8]
    assert key_fingerprint(_KEY) == f"sha256:{expected}"


def test_fingerprint_distinguishes_different_keys():
    assert key_fingerprint(_KEY) != key_fingerprint(_KEY + "x")
    assert key_fingerprint("aaa") != key_fingerprint("bbb")


def test_fingerprint_of_no_key():
    assert key_fingerprint("") == "none"
    assert key_fingerprint(None) == "none"


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_fingerprint_leaks_no_run_of_the_key(n):
    """The regression guard: no n-char run of the secret may survive."""
    fp = key_fingerprint(_KEY)
    runs = {_KEY[i:i + n] for i in range(len(_KEY) - n + 1)}
    assert not any(r in fp for r in runs)


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_hint_leaks_no_run_of_the_key(n):
    hint = key_hint(_KEY)
    runs = {_KEY[i:i + n] for i in range(len(_KEY) - n + 1)}
    leaked = [r for r in runs if r in hint]
    assert not leaked, f"hint leaked key material: {leaked}"


def test_hint_names_the_source_and_the_fingerprint():
    hint = key_hint(_KEY)
    assert "REMYXAI_API_KEY" in hint
    assert key_fingerprint(_KEY) in hint
    assert str(len(_KEY)) in hint


def test_hint_reports_a_custom_source():
    assert "REMYX_API_KEY_ADMIN" in key_hint(_KEY, source="REMYX_API_KEY_ADMIN")


def test_hint_when_no_key_is_set():
    hint = key_hint("", source="REMYXAI_API_KEY")
    assert "no API key" in hint
    assert "REMYXAI_API_KEY not set" in hint


def test_hint_distinguishes_two_keys_for_support_comparison():
    """The point of the hint: telling 'same key' from 'different key'."""
    assert key_hint(_KEY) != key_hint(_KEY + "x")


def test_import_time_log_carries_no_key_material(monkeypatch, caplog):
    """Re-import the module with a key set and inspect what it logged."""
    import importlib

    import remyxai.api as api

    monkeypatch.setenv("REMYXAI_API_KEY", _KEY)
    with caplog.at_level(logging.DEBUG):  # root logger
        importlib.reload(api)
    text = caplog.text
    assert _KEY not in text
    assert _KEY[:8] not in text, "the old behaviour: logged a slice of the key"
    assert key_fingerprint(_KEY) in text

    # Leave the module as the rest of the suite expects.
    monkeypatch.delenv("REMYXAI_API_KEY", raising=False)
    importlib.reload(api)


def test_source_file_does_not_slice_the_key():
    """Belt and braces: the old pattern must not reappear."""
    from pathlib import Path

    src = Path("remyxai/api/__init__.py").read_text()
    assert "REMYXAI_API_KEY[:" not in src
    assert "Using API Key:" not in src
