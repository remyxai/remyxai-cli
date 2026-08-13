"""Tests for `outrider init`'s provider-key routing and forced re-provisioning.

Two field-reported gaps (issue #49) drive these:

1. `init` provisioned only REMYX_API_KEY, so `--drafter-provider zai` produced
   a repo that reported "✓ Outrider is set up" and then failed auth on its
   first run in ~260ms. The engine pushes exactly ONE provider key (the
   account's connected credential), so any other tier provider has to come
   from this machine — and a tier with no key anywhere must fail at plan time.
2. Provisioning short-circuits with "Already enabled" on an installed repo, so
   a wrong tier provider was permanent through the CLI. `--force` revokes the
   install first, which makes the provisioner re-drive and rewrite the
   workflow files.

Run with: pytest tests/cli/test_outrider_init_provider_keys.py -q
"""
from unittest.mock import patch

import click
import pytest

from remyxai.cli import outrider_actions
from remyxai.cli.outrider_actions import (
    _PROVIDER_SECRET_NAMES,
    _plan_provider_secrets,
    _require_gh_for_pushes,
    _tier_providers,
    _validate_provider_keys,
)

UID = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
FAKE_KEY = "x" * 40


def _phases(drafter=None, refiner=None):
    return {
        "mode": "two_tier",
        "drafter": {"model": "", **({"provider": drafter} if drafter else {})},
        "refiner": {"model": "", **({"provider": refiner} if refiner else {})},
    }


@pytest.fixture(autouse=True)
def _no_provider_env(monkeypatch):
    """Start every test from "no provider key in this shell"."""
    for name in _PROVIDER_SECRET_NAMES.values():
        monkeypatch.delenv(name, raising=False)


# ─── provider name → secret name mapping (field note 2) ─────────────────────

def test_moonshot_has_a_secret_name():
    assert _PROVIDER_SECRET_NAMES["moonshot"] == "MOONSHOT_API_KEY"


def test_secret_names_match_the_setup_local_backend_registry():
    """The two provider tables must not drift.

    `init` accepting `moonshot` while `set-provider-secret` rejected it is
    exactly what drift looks like from the outside.
    """
    from remyxai.cli.outrider_local import _BACKEND_REGISTRY

    assert {name: cfg["secret_env"] for name, cfg in _BACKEND_REGISTRY.items()} \
        == _PROVIDER_SECRET_NAMES


def test_provider_choices_cover_every_known_provider():
    assert set(outrider_actions.PROVIDER_CHOICES) == set(_PROVIDER_SECRET_NAMES)


# ─── tier provider extraction ───────────────────────────────────────────────

def test_tier_providers_dedupes_in_tier_order():
    assert _tier_providers(_phases("zai", "anthropic")) == ["zai", "anthropic"]
    assert _tier_providers(_phases("zai", "zai")) == ["zai"]
    assert _tier_providers({"mode": "single", "main": {"provider": "moonshot"}}) \
        == ["moonshot"]
    assert _tier_providers(None) == []


# ─── key routing ────────────────────────────────────────────────────────────
#
# The engine pushes a key for EVERY provider named across `phases` that the
# account has connected (remyxai/remyx#558). So "connected" means "covered",
# and the CLI's job is only the provider you haven't connected.

def test_single_connected_provider_needs_no_local_key():
    """The ordinary install: the engine pushes it, `gh` never involved."""
    plan = _plan_provider_secrets(_phases("anthropic", "anthropic"),
                                  connected=["anthropic"])
    assert plan.engine_providers == ("anthropic",)
    assert plan.preferred == "anthropic"
    assert plan.pushes == []
    assert plan.missing == []


def test_both_connected_tiers_are_covered_server_side():
    """The relaxation: two connected providers, no local keys, no refusal.

    Before the engine pushed per-tier keys this refused with "no API key for:
    moonshot" — correct then, wrong once the engine covers both.
    """
    plan = _plan_provider_secrets(_phases("zai", "moonshot"),
                                  connected=["zai", "moonshot"])
    assert plan.engine_providers == ("zai", "moonshot")
    assert plan.pushes == []
    assert plan.missing == []
    _validate_provider_keys(plan)          # no raise


def test_preferred_is_the_capable_tier():
    """`model_provider` decides the workflow's baked default, so the refiner
    (later tier) wins over the cheap drafter when both are covered."""
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["zai", "anthropic"])
    assert plan.preferred == "anthropic"


def test_unconnected_tier_without_a_local_key_is_missing():
    """The reported failure: drafter at z.ai, no z.ai credential anywhere."""
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["anthropic"])
    assert plan.engine_providers == ("anthropic",)
    assert plan.missing == ["zai"]


def test_unconnected_tier_is_pushed_from_the_environment(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["anthropic"])
    assert plan.engine_providers == ("anthropic",)
    assert plan.pushes == [("zai", "ZAI_API_KEY", FAKE_KEY)]
    assert plan.missing == []


def test_connected_provider_is_never_pushed_locally(monkeypatch):
    """A key in the shell doesn't drag `gh` into an install the engine covers."""
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    plan = _plan_provider_secrets(_phases("zai", "zai"), connected=["zai"])
    assert plan.engine_providers == ("zai",)
    assert plan.pushes == []


def test_a_connected_provider_is_never_missing(monkeypatch):
    for tiers in (("zai", "moonshot"), ("anthropic", "zai"), ("moonshot", None)):
        plan = _plan_provider_secrets(
            _phases(*tiers), connected=["anthropic", "zai", "moonshot"])
        assert plan.missing == [], tiers


def test_moonshot_tier_pushes_moonshot_api_key(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_KEY)
    plan = _plan_provider_secrets(_phases("anthropic", "moonshot"),
                                  connected=["anthropic"])
    assert plan.pushes == [("moonshot", "MOONSHOT_API_KEY", FAKE_KEY)]


def test_nothing_connected_and_nothing_local_is_missing_anthropic():
    """No tier names a provider and nothing is connected: the engine bakes
    anthropic into the workflow, so that's the key the repo will need."""
    plan = _plan_provider_secrets(_phases(), connected=[])
    assert plan.engine_providers == ()
    assert plan.preferred is None
    assert plan.missing == ["anthropic"]


def test_inline_anthropic_key_is_covered_by_the_engine():
    """--anthropic-key gets connected during preflight → the engine pushes it,
    so this stays a no-`gh` install."""
    plan = _plan_provider_secrets(_phases(), connected=[],
                                  anthropic_key=FAKE_KEY)
    assert plan.engine_providers == ("anthropic",)
    assert plan.pushes == []
    assert plan.missing == []


def test_unpinned_tiers_follow_the_connected_provider():
    plan = _plan_provider_secrets(_phases(), connected=["moonshot"])
    assert plan.engine_providers == ("moonshot",)
    assert plan.missing == []


# ─── plan-time validation ───────────────────────────────────────────────────

def test_missing_key_raises_with_the_env_var_to_set():
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["anthropic"])
    with pytest.raises(click.ClickException) as e:
        _validate_provider_keys(plan)
    assert "ZAI_API_KEY" in e.value.message
    assert "--skip-key-check" in e.value.message


def test_skip_key_check_downgrades_to_a_warning():
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["anthropic"])
    _validate_provider_keys(plan, skip_key_check=True)  # no raise


def test_gh_requirement_only_applies_when_a_push_is_needed():
    with patch("remyxai.cli.outrider_local._gh_available", return_value=False):
        _require_gh_for_pushes([])  # no raise


def test_missing_gh_fails_the_install_that_needs_it():
    pushes = [("zai", "ZAI_API_KEY", FAKE_KEY)]
    with patch("remyxai.cli.outrider_local._gh_available", return_value=False):
        with pytest.raises(click.ClickException, match="`gh` CLI"):
            _require_gh_for_pushes(pushes)


def test_unauthenticated_gh_fails_the_install_that_needs_it():
    pushes = [("zai", "ZAI_API_KEY", FAKE_KEY)]
    with patch("remyxai.cli.outrider_local._gh_available", return_value=True), \
         patch("remyxai.cli.outrider_local._gh_authenticated", return_value=False):
        with pytest.raises(click.ClickException, match="isn't authenticated"):
            _require_gh_for_pushes(pushes)


# ─── init end-to-end (mocked engine + gh) ───────────────────────────────────

def _init(monkeypatch, **kwargs):
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    defaults = dict(
        repo="owner/repo", interest_id=UID, auto_interest=False, mode="auto",
        anthropic_key=None, skip_confirm=True, dry_run=False, no_wait=False,
    )
    defaults.update(kwargs)
    return outrider_actions.handle_outrider_init(**defaults)


def _connected(*providers):
    """get_integration_status stub reporting exactly `providers` connected."""
    ids = {
        integration_id for integration_id, workflow_value
        in outrider_actions.MODEL_PROVIDERS if workflow_value in providers
    }
    return lambda provider, **kw: {"connected": provider in ids}


def test_init_pushes_the_other_tiers_secret_before_provisioning(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", FAKE_KEY)
    order = []
    completed = {"status": "completed", "result": {"merged": True}}
    with patch.object(outrider_actions, "get_interest", return_value={"id": UID}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status",
                      side_effect=_connected("anthropic")), \
         patch.object(outrider_actions, "_kick_off_recommendations"), \
         patch("remyxai.cli.outrider_local._gh_available", return_value=True), \
         patch("remyxai.cli.outrider_local._gh_authenticated", return_value=True), \
         patch("remyxai.cli.outrider_local._gh_set_secret",
               side_effect=lambda *a: order.append(("secret",) + a)) as sec, \
         patch.object(outrider_actions, "provision_action",
                      side_effect=lambda *a, **k: order.append("provision")
                      or {"task_id": "t1"}) as prov, \
         patch.object(outrider_actions, "poll_provision_action",
                      return_value=completed):
        _init(monkeypatch, drafter_provider="zai")

    sec.assert_called_once_with("owner/repo", "ZAI_API_KEY", FAKE_KEY)
    # The secret has to be live before provisioning fires the first run.
    assert order[0][0] == "secret" and order[1] == "provision"
    # And the engine is told to push the tier we could NOT cover locally.
    assert prov.call_args.kwargs["model_provider"] == "claude_code"


def test_init_refuses_to_provision_a_tier_with_no_key(monkeypatch):
    with patch.object(outrider_actions, "get_interest", return_value={"id": UID}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status",
                      side_effect=_connected("anthropic")), \
         patch.object(outrider_actions, "provision_action") as prov:
        with pytest.raises(click.ClickException, match="ZAI_API_KEY"):
            _init(monkeypatch, drafter_provider="zai")
    prov.assert_not_called()


def test_skip_key_check_provisions_anyway(monkeypatch):
    completed = {"status": "completed", "result": {"merged": True}}
    with patch.object(outrider_actions, "get_interest", return_value={"id": UID}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status",
                      side_effect=_connected("anthropic")), \
         patch.object(outrider_actions, "_kick_off_recommendations"), \
         patch.object(outrider_actions, "provision_action",
                      return_value={"task_id": "t1"}) as prov, \
         patch.object(outrider_actions, "poll_provision_action",
                      return_value=completed):
        _init(monkeypatch, drafter_provider="zai", skip_key_check=True)
    prov.assert_called_once()


def test_dry_run_surfaces_the_missing_key(monkeypatch):
    """The check runs on --dry-run too — that's the cheapest place to learn
    the install would fail auth."""
    with patch.object(outrider_actions, "get_integration_status",
                      side_effect=_connected("anthropic")), \
         patch.object(outrider_actions, "provision_action") as prov:
        with pytest.raises(click.ClickException, match="ZAI_API_KEY"):
            _init(monkeypatch, drafter_provider="zai", dry_run=True)
    prov.assert_not_called()


# ─── --force ────────────────────────────────────────────────────────────────
#
# Provisioning short-circuits with "Already enabled" on a fully-installed repo,
# which also skips workflow rewriting. `--force` is the engine flag that
# re-drives every step; the CLI used to fake it by revoking the installation
# first, which rotated the repo's REMYX_API_KEY as a side effect.

def _init_and_capture(monkeypatch, **kwargs):
    completed = {"status": "completed", "result": {"merged": True}}
    with patch.object(outrider_actions, "get_interest", return_value={"id": UID}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status",
                      side_effect=_connected("anthropic")), \
         patch.object(outrider_actions, "_kick_off_recommendations"), \
         patch.object(outrider_actions, "provision_action",
                      return_value={"task_id": "t1"}) as prov, \
         patch.object(outrider_actions, "poll_provision_action",
                      return_value=completed):
        _init(monkeypatch, **kwargs)
    return prov


def test_force_is_sent_to_the_engine(monkeypatch):
    prov = _init_and_capture(monkeypatch, force=True)
    assert prov.call_args.kwargs["force"] is True


def test_force_defaults_off(monkeypatch):
    prov = _init_and_capture(monkeypatch)
    assert prov.call_args.kwargs["force"] is False


def test_no_revoke_workaround_remains():
    """The revoke path rotated REMYX_API_KEY just to get a rewrite."""
    assert not hasattr(outrider_actions, "_revoke_installation_for")
    assert not hasattr(outrider_actions, "revoke_installation")


def test_preferred_provider_is_sent_as_model_provider(monkeypatch):
    prov = _init_and_capture(monkeypatch)
    assert prov.call_args.kwargs["model_provider"] == "claude_code"


def test_unconnected_tier_is_told_to_connect_it():
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["anthropic"])
    line = [l for l in outrider_actions._describe_key_plan(plan)
            if l.startswith("zai")][0]
    assert "not connected" in line
    with pytest.raises(click.ClickException) as e:
        _validate_provider_keys(plan)
    assert "connect the provider" in e.value.message
    assert "ZAI_API_KEY" in e.value.message


def test_plan_marks_the_workflow_default():
    plan = _plan_provider_secrets(_phases("zai", "anthropic"),
                                  connected=["zai", "anthropic"])
    lines = outrider_actions._describe_key_plan(plan)
    assert any("workflow default" in l and l.startswith("anthropic") for l in lines)
