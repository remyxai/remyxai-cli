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


# ─── --github-secrets-only: the key is sealed here, and Remyx never holds a copy ───────────
#
# REMYX-296. The promise is not "we delete it after" — it is "we cannot read
# it". Two things have to hold for that to be true from this side:
#
#   * `connect_credential` is never called, because that call IS the stored
#     copy the flag exists to avoid, and
#   * what leaves the machine is ciphertext, never the key.
#
# Everything else (`gh` no longer being a precondition, the plan line the
# customer reads before committing) follows from the routing.

class _FakeSealedBox:
    """Stand-in for libsodium: prefixes instead of encrypting, so a test can
    tell sealed output from plaintext without needing a real key pair."""

    def __init__(self, _public_key):
        pass

    def encrypt(self, raw):
        return b"SEALED:" + raw


def _seal_env(monkeypatch, sealable=("ANTHROPIC_API_KEY", "ZAI_API_KEY")):
    """Patch the public-key fetch + the sealed box; capture what was sent."""
    import sys
    import types

    nacl = types.ModuleType("nacl")
    public = types.ModuleType("nacl.public")
    encoding = types.ModuleType("nacl.encoding")
    public.SealedBox = _FakeSealedBox
    public.PublicKey = lambda data, encoder=None: object()
    encoding.Base64Encoder = object
    nacl.public, nacl.encoding = public, encoding
    monkeypatch.setitem(sys.modules, "nacl", nacl)
    monkeypatch.setitem(sys.modules, "nacl.public", public)
    monkeypatch.setitem(sys.modules, "nacl.encoding", encoding)

    monkeypatch.setattr(
        "remyxai.api.interests.get_actions_public_key",
        lambda *a, **k: {
            "repo": "owner/repo", "key_id": "12345",
            "key": "cHVibGljLWtleQ==",
            "sealable_secret_names": list(sealable),
        },
    )


def test_byok_routes_a_shell_key_to_the_sealed_lane_not_the_engine():
    """A connected provider does NOT capture a key we're sealing — otherwise
    the engine would push its stored copy over the customer's own."""
    import os

    os.environ["ZAI_API_KEY"] = FAKE_KEY
    try:
        plan = _plan_provider_secrets(
            _phases(drafter="zai", refiner="zai"), ["zai"], byok=True,
        )
    finally:
        del os.environ["ZAI_API_KEY"]

    assert [p for p, _, _ in plan.sealed] == ["zai"]
    assert plan.engine_providers == ()
    assert plan.pushes == []
    assert plan.missing == []
    assert plan.preferred == "zai"


def test_byok_never_falls_back_to_the_inline_anthropic_connect():
    """Without --github-secrets-only, "nothing connected + key in env" is deliberately routed
    through connect_credential. That branch is the stored copy, so under
    --github-secrets-only the same input must seal instead."""
    import os

    os.environ["ANTHROPIC_API_KEY"] = FAKE_KEY
    try:
        plain = _plan_provider_secrets(_phases(), [], byok=False)
        byok = _plan_provider_secrets(_phases(), [], byok=True)
    finally:
        del os.environ["ANTHROPIC_API_KEY"]

    assert plain.engine_providers == ("anthropic",)   # the connect branch
    assert byok.engine_providers == ()
    assert [p for p, _, _ in byok.sealed] == ["anthropic"]


def test_byok_leaves_an_unconnected_provider_without_a_local_key_missing():
    plan = _plan_provider_secrets(
        _phases(drafter="moonshot", refiner="moonshot"), [], byok=True,
    )
    assert plan.missing == ["moonshot"]
    assert plan.sealed == ()


def test_byok_keeps_a_connected_provider_it_has_no_local_key_for():
    """--github-secrets-only declines to create NEW Remyx-side credentials; it doesn't disown
    one the user already chose to connect."""
    plan = _plan_provider_secrets(
        _phases(drafter="zai", refiner="zai"), ["zai"], byok=True,
    )
    assert plan.engine_providers == ("zai",)
    assert plan.sealed == ()


def test_byok_does_not_require_gh():
    """`gh` is a precondition only for the push lane. Sealed keys ride the
    provision-action body, so an install with no `gh` on the machine works."""
    import os

    os.environ["ZAI_API_KEY"] = FAKE_KEY
    try:
        plan = _plan_provider_secrets(
            _phases(drafter="zai", refiner="zai"), [], byok=True,
        )
    finally:
        del os.environ["ZAI_API_KEY"]

    assert plan.pushes == []
    _require_gh_for_pushes(plan.pushes)  # must not raise; gh is never consulted


def test_byok_plan_line_states_where_the_key_ends_up():
    """--dry-run has to say this before the customer commits."""
    import os

    os.environ["ANTHROPIC_API_KEY"] = FAKE_KEY
    try:
        plan = _plan_provider_secrets(_phases(), [], byok=True)
    finally:
        del os.environ["ANTHROPIC_API_KEY"]

    line = "\n".join(outrider_actions._describe_key_plan(plan))
    # The three things the promise is made of: sealed here, GitHub-only,
    # unreadable by us. Wording can change; all three have to survive it.
    assert "sealed here" in line
    assert "GitHub secret" in line and "only" in line
    assert "Remyx can't read it" in line


def test_only_ciphertext_leaves_the_machine(monkeypatch):
    _seal_env(monkeypatch)
    payload = outrider_actions._seal_provider_secrets(
        UID, "https://github.com/owner/repo",
        [("anthropic", "ANTHROPIC_API_KEY", FAKE_KEY)], api_key="k",
    )
    assert len(payload) == 1
    entry = payload[0]
    assert entry["secret_name"] == "ANTHROPIC_API_KEY"
    assert entry["key_id"] == "12345"
    # The plaintext must not survive anywhere in what we're about to POST.
    assert FAKE_KEY not in str(entry)
    import base64
    assert base64.b64decode(entry["encrypted_value"]) == b"SEALED:" + FAKE_KEY.encode()


def test_sealing_refuses_a_secret_name_the_engine_would_reject(monkeypatch):
    """An older engine's allowlist won't have every name. Fail here with
    something actionable rather than mid-install on an opaque 400."""
    _seal_env(monkeypatch, sealable=("ANTHROPIC_API_KEY",))
    with pytest.raises(click.ClickException) as e:
        outrider_actions._seal_provider_secrets(
            UID, "https://github.com/owner/repo",
            [("zai", "ZAI_API_KEY", FAKE_KEY)], api_key="k",
        )
    assert "ZAI_API_KEY" in str(e.value)


def test_ensure_model_provider_never_connects_under_byok(monkeypatch):
    """The load-bearing assertion of this whole feature."""
    called = []
    monkeypatch.setattr(
        outrider_actions, "connect_credential",
        lambda *a, **k: called.append(a),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    outrider_actions._ensure_model_provider(
        None, "api-key", connected=[], byok=True,
    )
    assert called == [], "connect_credential must never run under --github-secrets-only"

    # ...and the same input DOES connect without the flag, so the test is
    # pinning the flag's behavior rather than a dead code path.
    outrider_actions._ensure_model_provider(None, "api-key", connected=[])
    assert len(called) == 1


def test_validate_hints_do_not_tell_a_byok_user_to_connect(monkeypatch):
    plan = _plan_provider_secrets(
        _phases(drafter="moonshot", refiner="moonshot"), [], byok=True,
    )
    with pytest.raises(click.ClickException) as e:
        _validate_provider_keys(plan, byok=True)
    msg = str(e.value)
    assert "seals it and relays it" in msg
    assert "drop --github-secrets-only" in msg
