"""Tests for `outrider init` two-tier default + per-tier model config.

`init` provisions server-side via the Remyx GitHub App. Two-tier (a cheap
daily drafter + a capable weekly refiner) is the default; a single
`--provider`/`--model` applies to both tiers, per-tier flags override one,
and `--single-tier` opts out. The tier flags translate to the engine's
`phases` body on POST /interests/<id>/provision-action.
"""
from unittest.mock import patch

import click
import pytest

from remyxai.cli.outrider_actions import _build_init_phases
from remyxai.api import interests


# ─── phase-config translation ───────────────────────────────────────────────

def test_default_is_two_tier_no_hardcoded_vendor():
    # No flags, no connected provider known → two-tier with provider unset on
    # each tier, so the engine follows the account's connected provider. No
    # vendor (anthropic or otherwise) is baked in.
    phases = _build_init_phases(False, None, None, None, None, None, None)
    assert phases["mode"] == "two_tier"
    assert phases["drafter"] == {"model": ""}
    assert phases["refiner"] == {"model": ""}
    assert "provider" not in phases["drafter"]
    assert "provider" not in phases["refiner"]


def test_default_provider_fills_both_tiers():
    # The caller's connected provider (any of the three) fills unset tiers.
    phases = _build_init_phases(False, None, None, None, None, None, None,
                                default_provider="moonshot")
    assert phases["drafter"]["provider"] == "moonshot"
    assert phases["refiner"]["provider"] == "moonshot"


def test_single_provider_applies_to_both_tiers():
    phases = _build_init_phases(False, "zai", None, None, None, None, None)
    assert phases["drafter"]["provider"] == "zai"
    assert phases["refiner"]["provider"] == "zai"


def test_explicit_provider_wins_over_connected_default():
    phases = _build_init_phases(False, "zai", None, None, None, None, None,
                                default_provider="anthropic")
    assert phases["drafter"]["provider"] == "zai"
    assert phases["refiner"]["provider"] == "zai"


def test_single_model_applies_to_both_tiers():
    phases = _build_init_phases(False, "moonshot", "kimi-k3",
                                None, None, None, None)
    assert phases["drafter"]["model"] == "kimi-k3"
    assert phases["refiner"]["model"] == "kimi-k3"


def test_per_tier_override_wins_over_shared():
    # GLM drafter + a different refiner provider — mix providers per tier.
    phases = _build_init_phases(False, None, None, "zai", "glm-5.2",
                                "moonshot", None)
    assert phases["drafter"] == {"provider": "zai", "model": "glm-5.2"}
    assert phases["refiner"] == {"provider": "moonshot", "model": ""}


def test_single_tier_sends_no_phases_by_default():
    # Plain single-file with no pins → None, so the engine bakes its default.
    assert _build_init_phases(True, None, None, None, None, None, None) is None


def test_single_tier_with_pinned_provider():
    phases = _build_init_phases(True, "zai", None, None, None, None, None)
    assert phases == {"mode": "single",
                      "main": {"provider": "zai", "model": ""}}


def test_single_tier_rejects_per_tier_flags():
    with pytest.raises(click.UsageError):
        _build_init_phases(True, None, None, "zai", "glm-5.2", None, None)


# ─── payload wiring ──────────────────────────────────────────────────────────

def test_provision_action_includes_phases_in_body():
    captured = {}

    class _Resp:
        status_code = 202

        def raise_for_status(self):
            pass

        def json(self):
            return {"task_id": "t1"}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    phases = {"mode": "two_tier",
              "drafter": {"provider": "anthropic", "model": ""},
              "refiner": {"provider": "anthropic", "model": ""}}
    with patch.object(interests.requests, "post", _fake_post), \
            patch.object(interests, "log_api_response", lambda *_a, **_k: None):
        interests.provision_action("iid", repo_url="https://github.com/o/r",
                                   phases=phases, api_key="k")
    assert captured["json"]["phases"] == phases
    assert captured["json"]["auto_merge"] is True


def test_provision_action_omits_phases_when_none():
    class _Resp:
        status_code = 202

        def raise_for_status(self):
            pass

        def json(self):
            return {"task_id": "t1"}

    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _Resp()

    with patch.object(interests.requests, "post", _fake_post), \
            patch.object(interests, "log_api_response", lambda *_a, **_k: None):
        interests.provision_action("iid", api_key="k")
    assert "phases" not in captured["json"]
