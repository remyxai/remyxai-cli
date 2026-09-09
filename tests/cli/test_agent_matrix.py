"""The vendored agent/provider matrix, and validation against it.

Two things are being pinned here.

**The join.** A pair is valid when the agent's API family is one the provider
serves. Encoding that, rather than a list of blessed combinations, is what
makes a new provider usable by every agent that already speaks its family — so
the tests assert the *rule*, not a snapshot of today's rows.

**The strictness split.** This CLI ships independently of the action, so the
vendored matrix can be older than the action a user runs. Only a durable fact
may hard-fail; anything the CLI merely hasn't heard of has to pass through
with a warning, or every newly added provider would look broken until the user
upgraded.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from remyxai import agent_matrix as am

ROOT = Path(__file__).resolve().parent.parent.parent


def _levels(problems):
    return [p.level for p in problems]


def _text(problems):
    return " ".join(p.message for p in problems)


# ─── the vendored artifact ─────────────────────────────────────────────────


def test_the_vendored_matrix_has_the_shape_the_helpers_assume():
    from remyxai._agent_matrix import MATRIX

    for key in ("agents", "providers", "pairs"):
        assert key in MATRIX
    for name, info in MATRIX["agents"].items():
        for field in ("display_name", "api_family", "key_env", "capabilities"):
            assert field in info, f"agent {name} missing {field}"
    for pid, info in MATRIX["providers"].items():
        for field in ("display_name", "secret_env", "families", "verified",
                      "verification_caveat", "caller_supplied_endpoint"):
            assert field in info, f"provider {pid} missing {field}"


def test_the_vendored_matrix_is_not_hand_edited():
    """It is generated; the header has to say so, or someone will edit it."""
    text = (ROOT / "remyxai" / "_agent_matrix.py").read_text()
    assert "GENERATED FILE" in text
    assert "scripts/sync_agent_matrix.py" in text


def test_the_sync_script_round_trips_its_own_output():
    """`--check` must pass immediately after a write, or CI can never be green.

    Guards a real trap in the generator: the JSON is embedded in a raw string,
    where a trailing backslash is literal rather than a line continuation. An
    earlier revision emitted `r\"\"\"\\` and the module raised
    JSONDecodeError at import — while `--check` still passed, because it
    sliced past the marker and never imported anything.
    """
    source = ROOT / "remyxai" / "_agent_matrix.py"
    before = source.read_text()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_agent_matrix.py"),
         "--check", "--from", str(_artifact_for_check())],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert source.read_text() == before, "--check must not write"


def _artifact_for_check(tmp=[]):
    """Write the currently-vendored matrix out as a standalone artifact.

    Checking against the *action's* checkout would make this test depend on a
    sibling clone existing. Round-tripping the vendored data proves the
    generator is self-consistent, which is the part that can break.
    """
    from remyxai._agent_matrix import MATRIX

    if not tmp:
        import tempfile

        path = Path(tempfile.mkdtemp()) / "agent-matrix.json"
        path.write_text(json.dumps(MATRIX, indent=2, sort_keys=True) + "\n")
        tmp.append(path)
    return tmp[0]


# ─── the join ──────────────────────────────────────────────────────────────


def test_every_published_pair_validates():
    """The artifact and the checker cannot disagree about what works."""
    for agent, provider, _secret, _model, _verified in am.pair_rows():
        if provider.startswith("("):
            continue          # native-router placeholder, not a provider id
        problems = am.check_pair(agent, provider, model="x")
        assert am.first_error(problems) is None, (
            f"{agent} + {provider} is published but the checker rejects it: "
            f"{_text(problems)}"
        )


@pytest.mark.parametrize("agent,provider,expected_fix", [
    ("claude", "openai", "codex"),
    ("codex", "anthropic", "claude"),
    ("codex", "zai", "claude"),
])
def test_a_family_mismatch_errors_and_names_the_agent_that_works(
    agent, provider, expected_fix
):
    """The whole point of the join: the fix is derivable, not hand-written."""
    problems = am.check_pair(agent, provider)
    error = am.first_error(problems)
    assert error is not None
    assert f"--agent {expected_fix}" in error.message


def test_the_suggestion_prefers_a_direct_speaker_over_a_native_router():
    """R-CLI can reach almost anything, but only through its own account.

    Answering "OpenAI does not serve anthropic-messages" with "use backboard"
    tells the user to go sign up for a different service, when `codex` is a
    drop-in that uses the key they already have. Alphabetical ordering put
    `backboard` first until this was fixed.
    """
    error = am.first_error(am.check_pair("claude", "openai"))
    assert "--agent codex" in error.message
    assert "backboard" not in error.message


def test_a_native_router_reaches_a_provider_no_direct_agent_serves():
    """And when nothing direct exists, the router is the honest answer."""
    direct, routed = am.agents_serving("zai")
    assert "claude" in direct
    assert routed == ["backboard"]


# ─── strictness: errors are durable, unknowns pass through ─────────────────


def test_an_unknown_agent_warns_rather_than_failing():
    problems = am.check_pair("some-future-agent", "zai")
    assert _levels(problems) == [am.WARN]
    assert "does not know agent" in _text(problems)


def test_an_unknown_provider_warns_rather_than_failing():
    problems = am.check_pair("claude", "some-future-provider")
    assert _levels(problems) == [am.WARN]
    assert "does not know provider" in _text(problems)


def test_no_provider_is_always_valid():
    """The oldest supported shape: no provider input at all, a literal no-op."""
    assert am.check_pair("claude", "") == []
    assert am.check_pair("", "") == []


def test_an_unset_agent_means_the_default_agent():
    assert am.resolve_agent("") == "claude"
    assert am.resolve_agent("   ") == "claude"
    assert am.check_pair("", "zai") == am.check_pair("claude", "zai")


# ─── advisory warnings ─────────────────────────────────────────────────────


def test_a_verification_caveat_is_surfaced_with_the_published_wording():
    """The text comes from the artifact, so it cannot drift from the action."""
    from remyxai._agent_matrix import MATRIX

    caveat = MATRIX["providers"]["openrouter"]["verification_caveat"]
    assert caveat, "artifact carries no caveat to surface"
    problems = am.check_pair("claude", "openrouter", model="z-ai/glm-5.3")
    assert _levels(problems) == [am.WARN]
    assert caveat in _text(problems)


def test_the_no_default_model_warning_respects_a_named_model():
    """Otherwise the most ordinary configuration there is would warn.

    `--provider anthropic --model claude-opus-4-8` uses no default at all, so
    warning about the absent default is noise. check_pair has to be told what
    the caller named to know that.
    """
    assert am.check_pair("claude", "anthropic", model="claude-opus-4-8") == []
    warned = am.check_pair("claude", "anthropic", model="")
    assert _levels(warned) == [am.WARN]
    assert "no default model" in _text(warned)


def test_a_native_router_does_not_pretend_to_validate_provider_names():
    """R-CLI's catalogue is large, changes without notice, and uses different
    names — it has no `zai`. A local allowlist would go stale, so the run
    checks the live catalogue and this only says so."""
    problems = am.check_pair("backboard", "zai", model="glm-5.3")
    assert am.first_error(problems) is None
    assert "own catalogue" in _text(problems)


# ─── what the caller has to set ────────────────────────────────────────────


@pytest.mark.parametrize("agent,provider,expected", [
    ("claude", "anthropic", "ANTHROPIC_API_KEY"),
    ("claude", "zai", "ZAI_API_KEY"),
    ("codex", "moonshot", "MOONSHOT_API_KEY"),
    ("codex", "openrouter", "OPENROUTER_API_KEY"),
])
def test_secret_env_comes_from_the_provider(agent, provider, expected):
    assert am.secret_env(agent, provider) == expected


def test_a_native_routers_secret_is_its_own_regardless_of_provider():
    """Backboard's key is both the agent credential and the model-routing
    credential — one key reaches every provider it lists."""
    assert am.secret_env("backboard", "openai") == "BACKBOARD_API_KEY"
    assert am.secret_env("backboard", "google") == "BACKBOARD_API_KEY"


def test_a_custom_endpoint_falls_back_to_the_agents_own_credential():
    """`provider: custom` means the caller supplies endpoint and auth."""
    assert am.secret_env("claude", "custom") == "ANTHROPIC_API_KEY"
    assert am.secret_env("codex", "custom") == "CODEX_API_KEY"


def test_endpoint_and_default_model_are_read_per_family():
    """The same provider maps to a different endpoint per agent, because the
    agents speak different API families."""
    assert am.endpoint("claude", "moonshot") != am.endpoint("codex", "moonshot")
    assert am.default_model("claude", "zai") == "glm-5.3"


def test_the_vendored_matrix_is_current_with_a_sibling_action_checkout():
    """Catches the vendored copy rotting against the action it came from.

    `test_the_sync_script_round_trips_its_own_output` only proves the
    generator is self-consistent — it would stay green on a matrix a year
    out of date. This one compares against the real source when a checkout
    is available, and skips when it is not, so it helps during development
    without making the suite depend on a sibling clone.
    """
    from remyxai._agent_matrix import MATRIX

    candidates = [
        ROOT.parent / "outrider" / "docs" / "agent-matrix.json",
        Path.home() / "outrider" / "docs" / "agent-matrix.json",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        pytest.skip("no sibling remyxai/outrider checkout to compare against")

    published = json.loads(source.read_text())
    if published == MATRIX:
        return
    stale = sorted(set(published.get("providers", {})) - set(MATRIX["providers"]))
    raise AssertionError(
        f"remyxai/_agent_matrix.py is stale against {source}"
        + (f" (missing providers: {stale})" if stale else "")
        + ".\nRefresh it:\n"
        f"  python scripts/sync_agent_matrix.py --from {source}"
    )


# ─── no surface may claim Claude-only behavior ─────────────────────────────


def test_shared_command_help_does_not_claim_claude_only_behavior():
    """`trigger`, `setup-local` and `set-provider-secret` all work with any
    agent, so their help must not describe one.

    These read as though the flag does not apply when `agent` is codex or
    backboard, and they hid from an earlier sweep because the phrase sat on
    a continuation line of a multi-line `help=(...)` string rather than
    beside the `help=` itself.
    """
    from click.testing import CliRunner

    from remyxai.cli.commands import cli

    banned = (
        "Route Claude Code",
        "Claude Code picks",
        "so Claude Code uses",
        "route Claude Code at",
        "Claude Code subprocess",
    )
    for command in ("trigger", "setup-local", "set-provider-secret"):
        out = CliRunner().invoke(cli, ["outrider", command, "--help"]).output
        for phrase in banned:
            assert phrase not in out, (
                f"`outrider {command} --help` still says {phrase!r}"
            )


def test_the_only_claude_mentions_left_are_true_of_claude_specifically():
    """A blanket ban would be wrong — some statements about Claude Code are
    simply true, and deleting them loses real information.

    What survives, and why each is correct:

    * the engine's `claude_code` integration id, which cannot be renamed
      without a data migration on live installs;
    * the two-tier local install, which rewrites an Anthropic-Messages
      template in place and so is Claude-Code-only by construction;
    * "only Claude Code has a round cap", which is the reason the timeout is
      the sole spend bound on the other two agents.
    """
    import subprocess

    out = subprocess.run(
        ["grep", "-rn", "Claude", "--include=*.py", "remyxai/"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout
    for line in out.splitlines():
        if "_agent_matrix.py" in line:
            continue          # generated; carries the vendor's own names
        assert any(marker in line for marker in (
            "claude_code",              # engine integration id
            "Claude-Code-only",         # two-tier constraint
            "Only Claude Code has",     # the round-cap fact
            "Empty means Claude Code",  # the pinned default
            "engine lists this integration",
            "engine.remyx.ai/integrations",
            "Any provider counts equally",
            "was wrong in three ways",  # the cocoindex post-mortem comment
            'a Claude',                 # …continued
            "(Claude Code), not",       # the dropped-agent warning
        )), f"unclassified Claude mention: {line}"
