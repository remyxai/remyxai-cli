"""Tests for the no-App ("local") Outrider setup path.

`remyxai outrider setup-local` self-provisions with the user's own gh token —
no Remyx GitHub App. Covers workflow rendering (GITHUB_TOKEN vs PAT), the
gh-secret stdin invariant, the sha-on-update fix, rollback ordering, the
dry-run contract, and CLI wiring.
"""
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_local
from remyxai.cli.commands import cli


# ─── workflow rendering ─────────────────────────────────────────────────────

def test_render_uses_builtin_github_token():
    wf = outrider_local._render_local_workflow("uuid-123")
    assert "interest-id: uuid-123" in wf
    assert "github-token:" not in wf                      # uses the built-in GITHUB_TOKEN
    assert "workflow_dispatch:" in wf
    assert "rate-limit-days: '0'" in wf                   # don't suppress manual/scheduled runs


def test_render_declares_all_backend_secrets_on_uses_step():
    """The action's `env:` block on the uses step passes every registered
    backend's secret. The action's Configure step (outrider v1.7.29+)
    reads only the one matching `provider`; missing secrets evaluate to
    empty and fail clean at dispatch time. REMYX_API_KEY is included in
    the same block since it's needed for every run."""
    wf = outrider_local._render_local_workflow("uuid")
    assert "REMYX_API_KEY: ${{ secrets.REMYX_API_KEY }}" in wf
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in wf
    assert "ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}" in wf
    assert "MOONSHOT_API_KEY: ${{ secrets.MOONSHOT_API_KEY }}" in wf
    # The old fork-side Configure step is GONE — the action handles auth
    # wiring via its `provider` input now.
    assert "Configure provider auth" not in wf
    assert "ANTHROPIC_API_KEY_SECRET" not in wf


def test_render_declares_provider_workflow_input_with_all_backends():
    """Workflow exposes a `provider` workflow_dispatch input listing every
    backend registered in _BACKEND_REGISTRY. Dispatches can select any of
    them at run time; setup-local's --backend controls only the default."""
    wf = outrider_local._render_local_workflow("uuid")
    assert "      provider:" in wf
    assert "type: choice" in wf
    for name in outrider_local._BACKEND_REGISTRY:
        assert f"- {name}" in wf, f"missing provider option in workflow: {name}"
    # Default backend is anthropic when not overridden.
    assert "default: 'anthropic'" in wf


def test_render_forwards_provider_model_and_base_url_to_action():
    """`provider`, `model`, and `base-url` workflow_dispatch inputs are
    threaded into the action's `with:` block so per-dispatch overrides
    propagate all the way through — the action's `provider` picks default
    base-URL + auth wiring, `model` sets ANTHROPIC_MODEL, and an explicit
    `base-url` overrides the provider default (self-hosted / on-prem)."""
    wf = outrider_local._render_local_workflow("uuid")
    assert "provider: ${{ inputs.provider }}" in wf
    assert "model: ${{ inputs.model }}" in wf
    assert "model-base-url: ${{ inputs.base-url }}" in wf


def test_render_backend_moonshot_bakes_moonshots_timeout():
    """--backend moonshot sets the dispatch `provider` default to 'moonshot'
    and bakes moonshot's longer timeout (kimi thinking mode) into `with:`."""
    wf = outrider_local._render_local_workflow("uuid", backend="moonshot")
    assert "default: 'moonshot'" in wf
    assert "claude-timeout: '3600'" in wf


def test_render_backend_zai_uses_bumped_timeout():
    """--backend zai also gets the bumped 3600s — GLM's thinking mode adds
    per-turn latency similar to Kimi's."""
    wf = outrider_local._render_local_workflow("uuid", backend="zai")
    assert "default: 'zai'" in wf
    assert "claude-timeout: '3600'" in wf


def test_render_backend_anthropic_uses_default_timeout_900():
    """--backend anthropic keeps the historical 900s — Opus is fast enough
    per-turn that the default doesn't need bumping."""
    wf = outrider_local._render_local_workflow("uuid", backend="anthropic")
    assert "default: 'anthropic'" in wf
    assert "claude-timeout: '900'" in wf


def test_render_unknown_backend_raises():
    """A backend not in the registry raises ValueError; caller must catch
    or the CLI's UsageError handles it upstream."""
    with pytest.raises(ValueError, match="unknown backend"):
        outrider_local._render_local_workflow("uuid", backend="bedrock")


def test_render_stays_within_githubs_workflow_dispatch_input_ceiling():
    """workflow_dispatch accepts at most 10 inputs, and GitHub rejects the
    whole workflow past that — "maximum number of inputs for
    workflow_dispatch event is 10".

    This template shipped **11**, so every setup-local install wrote a
    workflow GitHub would not run. Nothing caught it: the tests asserted that
    particular inputs were present, never how many there were in total. This
    is the guard that would have.
    """
    yaml = pytest.importorskip("yaml")
    for backend in ("anthropic", "zai", "moonshot"):
        for agent in ("claude", "codex", "backboard"):
            wf = yaml.safe_load(
                outrider_local._render_local_workflow(
                    "uuid", backend=backend, agent=agent
                )
            )
            on = wf.get("on") or wf.get(True)
            inputs = (on["workflow_dispatch"] or {}).get("inputs") or {}
            assert len(inputs) <= 10, (
                f"{agent}/{backend} declares {len(inputs)} inputs "
                f"({sorted(inputs)}); GitHub's ceiling is 10 and it rejects "
                f"the workflow outright past it"
            )


def test_render_declares_the_inputs_the_action_canonically_declares():
    """Parity with the action's own outrider.yml, plus the agent axis.

    Two inputs were dropped to make room for `agent` inside the ceiling:
    `search-method`, which the canonical template never declared either, and
    `claude-timeout`, now baked into `with:` from the provider's default.
    """
    wf = outrider_local._render_local_workflow("uuid")
    assert "workflow_dispatch:" in wf
    assert "    inputs:" in wf
    for name in ("agent", "provider", "model", "base-url", "pin-arxiv",
                 "mode", "publish", "start-from-ref", "lead-content",
                 "staged-synthesis"):
        assert f"      {name}:" in wf, f"missing input declaration: {name}"


def test_render_does_not_declare_the_two_inputs_it_traded_away():
    """Pinned deliberately: re-adding either silently breaks the workflow by
    pushing it over the ceiling, and the failure looks like a YAML problem
    rather than a budget one."""
    yaml = pytest.importorskip("yaml")
    text = outrider_local._render_local_workflow("uuid")
    wf = yaml.safe_load(text)
    on = wf.get("on") or wf.get(True)
    declared = set(((on["workflow_dispatch"] or {}).get("inputs") or {}))
    # Parsed, not substring-matched: the baked `with:` line is indented
    # deeper than an input declaration, so a naive `"      claude-timeout:"
    # not in text` matches it and fails for the wrong reason.
    assert "search-method" not in declared
    assert "claude-timeout" not in declared
    # But the budget is still honored — just not overridable per dispatch.
    assert "claude-timeout: '900'" in text


def test_render_forwards_every_declared_input_to_the_action():
    """Every declared input reaches the action's `with:` block, or dispatching
    it does nothing and looks like the action ignoring the value."""
    yaml = pytest.importorskip("yaml")
    text = outrider_local._render_local_workflow("uuid")
    wf = yaml.safe_load(text)
    on = wf.get("on") or wf.get(True)
    declared = list(((on["workflow_dispatch"] or {}).get("inputs") or {}))
    # `base-url` is the one rename: the action's input is `model-base-url`.
    forwarded = {"base-url": "model-base-url"}
    for name in declared:
        target = forwarded.get(name, name)
        assert f"{target}: ${{{{ inputs.{name} }}}}" in text, (
            f"input {name} is declared but never forwarded"
        )


def test_render_declares_and_forwards_refiner_dispatch_inputs():
    """The runner must declare + forward every input the refiner
    (outrider-weekly-refine.yml) dispatches — mode, publish, start-from-ref,
    lead-content, staged-synthesis — or GitHub rejects the dispatch as an
    unknown workflow_dispatch key and no refinement PR opens."""
    wf = outrider_local._render_local_workflow("uuid")
    for name in ("mode", "publish", "start-from-ref", "lead-content", "staged-synthesis"):
        assert f"      {name}:" in wf, f"missing input declaration: {name}"
        assert f"{name}: ${{{{ inputs.{name} }}}}" in wf, f"missing forwarding for {name}"
    # Defaults match the action's own so scheduled/manual runs are unchanged.
    assert "default: 'recommend'" in wf   # mode
    assert "default: 'pr'" in wf          # publish


# ─── two-tier template rendering (fetched from remyxai/outrider@v1) ──────────

_FAKE_DRAFTER_TEMPLATE = (
    "name: Outrider daily\n"
    "on:\n  schedule:\n    - cron: '0 6 * * *'\n"
    "jobs:\n  scout:\n    env:\n"
    "      REMYX_API_KEY: ${{ secrets.REMYX_API_KEY }}\n"
    "      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}\n"
    "      ANTHROPIC_MODEL: claude-haiku-4-5\n"
    "    steps:\n"
    "      - uses: actions/checkout@v4\n"
    "      - uses: ./\n"
    "        with:\n"
    "          interest-id: '29ca03e7-454d-446c-9941-32c96c53d95d'\n"
    "          publish: branch\n"
)

# Mirrors the @v1 refiner's gap-analysis call + Opus dispatch anchors.
_FAKE_REFINER_TEMPLATE = (
    "name: Outrider weekly refine\n"
    "on:\n  workflow_dispatch: {}\n"
    "jobs:\n  refine:\n    steps:\n"
    "      - name: Generate gap analysis\n        id: gap\n        env:\n"
    "          BRANCH: ${{ steps.pick.outputs.picked }}\n"
    "          REPO: ${{ github.repository }}\n"
    "        run: |\n"
    "          python3 - <<'PYEOF'\n"
    "          req = urllib.request.Request(\n"
    '              "https://api.anthropic.com/v1/messages",\n'
    "              data=json.dumps({\n"
    '                  "model": "claude-sonnet-4-6",\n'
    "              }).encode(),\n"
    "              headers={\n"
    '                  "x-api-key": os.environ["ANTHROPIC_API_KEY"],\n'
    "              },\n"
    "          )\n          PYEOF\n"
    "      - name: Dispatch Opus refinement\n        run: |\n"
    "          gh workflow run outrider.yml \\\n"
    "            -f provider=anthropic \\\n"
    "            -f model=claude-opus-4-8 \\\n"
    "            -f publish=pr\n"
)


def test_render_drafter_rewrites_local_action_and_interest_id():
    """The drafter fetched from @v1 uses `uses: ./` (resolves only inside the
    outrider repo) and a self-test interest-id. Both must be rewritten so the
    installed drafter targets the published action + customer's interest."""
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_DRAFTER_TEMPLATE):
        rendered = outrider_local._render_drafter_workflow("cust-uuid")
    assert "uses: ./" not in rendered
    assert "uses: remyxai/outrider@v1" in rendered
    assert "interest-id: 'cust-uuid'" in rendered
    assert outrider_local._OUTRIDER_SELF_INTEREST_ID not in rendered


def test_render_drafter_guards_missing_local_action_ref():
    """If the template stops referencing the action via `uses: ./`, fail loud
    rather than silently shipping an un-rewritten drafter."""
    broken = _FAKE_DRAFTER_TEMPLATE.replace("uses: ./", "uses: remyxai/outrider@v1")
    with patch.object(outrider_local, "_fetch_outrider_template", return_value=broken):
        with pytest.raises(click.ClickException, match="no longer references the action"):
            outrider_local._render_drafter_workflow("cust-uuid")


def test_fetch_outrider_template_calls_gh_api_without_double_prefix(monkeypatch):
    """Regression: _fetch_outrider_template must pass the contents path directly
    to _gh_api_json (which already prepends `gh api`), not a leading 'api' arg
    that would produce `gh api api repos/...`."""
    import base64
    captured = {}

    def fake_gh_api_json(args):
        captured["args"] = args
        return {"content": base64.b64encode(b"hello").decode()}

    monkeypatch.setattr(outrider_local, "_gh_api_json", fake_gh_api_json)
    out = outrider_local._fetch_outrider_template(".github/workflows/outrider-daily.yml")
    assert out == "hello"
    assert captured["args"][0] != "api"
    assert captured["args"][0].startswith("repos/remyxai/outrider/contents/")
    assert "ref=v1" in captured["args"][0]


# ─── optional per-stage model overrides ─────────────────────────────────────

def test_provider_inference_and_uses_zai():
    assert outrider_local._provider_for_model("glm-5.2") == "zai"
    assert outrider_local._provider_for_model("glm-4.6") == "zai"
    assert outrider_local._provider_for_model("claude-opus-4-8") == "anthropic"
    assert outrider_local.uses_zai(None, "glm-5.2", None) is True
    assert outrider_local.uses_zai("claude-haiku-4-5", None, "claude-opus-4-8") is False


def test_drafter_model_default_leaves_template_untouched():
    """No --drafter-model → single-provider default preserved (Anthropic key,
    Haiku model, no z.ai base URL)."""
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_DRAFTER_TEMPLATE):
        wf = outrider_local._render_drafter_workflow("uuid")
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in wf
    assert "ANTHROPIC_MODEL: claude-haiku-4-5" in wf
    assert "model-base-url" not in wf


def test_drafter_model_glm_routes_at_zai():
    """--drafter-model glm-5.2 swaps to Bearer auth (ZAI_API_KEY), sets the GLM
    model + z.ai base URL, and drops ANTHROPIC_API_KEY (mutual exclusion)."""
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_DRAFTER_TEMPLATE):
        wf = outrider_local._render_drafter_workflow("uuid", model="glm-5.2")
    assert "ANTHROPIC_AUTH_TOKEN: ${{ secrets.ZAI_API_KEY }}" in wf
    assert "ANTHROPIC_API_KEY:" not in wf
    assert "ANTHROPIC_MODEL: glm-5.2" in wf
    assert "model-base-url: https://api.z.ai/api/anthropic" in wf


def test_drafter_model_anthropic_keeps_key_only_swaps_model():
    """A non-default Anthropic model swaps only the model — keeps x-api-key auth,
    no z.ai routing."""
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_DRAFTER_TEMPLATE):
        wf = outrider_local._render_drafter_workflow("uuid", model="claude-sonnet-4-6")
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in wf
    assert "ANTHROPIC_MODEL: claude-sonnet-4-6" in wf
    assert "model-base-url" not in wf


def test_refiner_gap_model_glm_and_refine_dispatch():
    """--refiner-model glm-5.2 routes the gap-analysis call at z.ai (endpoint +
    Bearer + ZAI env); --refine-model flips the dispatch model + provider."""
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_REFINER_TEMPLATE):
        wf = outrider_local._render_refiner_workflow(
            gap_model="glm-5.2", refine_model="glm-4.6",
        )
    # gap-analysis routed at z.ai
    assert "https://api.z.ai/api/anthropic/v1/messages" in wf
    assert '"model": "glm-5.2"' in wf
    assert "Bearer {os.environ['ZAI_API_KEY']}" in wf
    assert "ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}" in wf
    # dispatched refine run flipped to GLM/zai
    assert "-f model=glm-4.6" in wf
    assert "-f provider=zai" in wf


def test_refiner_default_leaves_template_untouched():
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_REFINER_TEMPLATE):
        wf = outrider_local._render_refiner_workflow()
    assert "https://api.anthropic.com/v1/messages" in wf
    assert '"model": "claude-sonnet-4-6"' in wf
    assert "-f model=claude-opus-4-8" in wf and "-f provider=anthropic" in wf


def test_refine_model_anthropic_keeps_provider_anthropic():
    with patch.object(outrider_local, "_fetch_outrider_template",
                      return_value=_FAKE_REFINER_TEMPLATE):
        wf = outrider_local._render_refiner_workflow(refine_model="claude-opus-4-8")
    assert "-f model=claude-opus-4-8" in wf and "-f provider=anthropic" in wf


def test_model_override_anchor_guard_fails_loud():
    """A template that lost the ANTHROPIC_MODEL anchor must raise, not silently
    skip the override."""
    broken = _FAKE_DRAFTER_TEMPLATE.replace("ANTHROPIC_MODEL: claude-haiku-4-5\n", "")
    with patch.object(outrider_local, "_fetch_outrider_template", return_value=broken):
        with pytest.raises(click.ClickException, match="no longer contains the expected anchor"):
            outrider_local._render_drafter_workflow("uuid", model="glm-5.2")


def test_model_flags_require_two_tier(monkeypatch):
    """--drafter-model without --two-tier is a usage error."""
    monkeypatch.setenv("REMYXAI_API_KEY", "rk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    with pytest.raises(click.UsageError, match="require --two-tier"):
        outrider_local.handle_outrider_setup_local(
            repo="o/r", interest_id="uuid", auto_interest=False, mode="review",
            anthropic_key=None, skip_confirm=True, dry_run=True,
            two_tier=False, drafter_model="glm-5.2",
        )


# ─── gh secret stdin invariant ───────────────────────────────────────────────

def test_gh_set_secret_value_via_stdin(monkeypatch):
    captured = {}

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return _Done()

    monkeypatch.setattr("subprocess.run", fake_run)
    outrider_local._gh_set_secret("o/r", "REMYX_API_KEY", "supersecret")
    assert "supersecret" not in captured["cmd"]
    assert captured["input"] == "supersecret"


def test_gh_set_secret_403_hint(monkeypatch):
    class _Fail:
        returncode = 1
        stderr = "HTTP 403: permission"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Fail())
    with pytest.raises(click.ClickException, match="admin scope"):
        outrider_local._gh_set_secret("o/r", "X", "v")


# ─── sha-on-update (regression: PUT over existing file 422'd) ────────────────

def test_gh_put_file_includes_sha_when_file_exists(monkeypatch):
    calls = []

    class _R:
        def __init__(self, rc=0, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "PUT" not in cmd:                       # the GET in _gh_get_file_sha
            import json
            return _R(out=json.dumps({"sha": "abc123"}))
        return _R(out="{}")

    monkeypatch.setattr("subprocess.run", fake_run)
    outrider_local._gh_put_file("o/r", "b", outrider_local.WORKFLOW_PATH, "x", "msg")
    put_cmd = next(c for c in calls if "PUT" in c)
    assert "sha=abc123" in put_cmd


# ─── handler: rollback ordering ──────────────────────────────────────────────

def _base_patches(monkeypatch):
    monkeypatch.setenv("REMYXAI_API_KEY", "rk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")


def test_rollback_deletes_branch_and_skips_secrets_when_put_fails(monkeypatch):
    _base_patches(monkeypatch)
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_local, "_resolve_interest_id", return_value=uid), \
         patch.object(outrider_local, "_gh_available", return_value=True), \
         patch.object(outrider_local, "_gh_authenticated", return_value=True), \
         patch.object(outrider_local, "_gh_default_branch", return_value="main"), \
         patch.object(outrider_local, "_gh_branch_exists", return_value=False), \
         patch.object(outrider_local, "_gh_get_branch_sha", return_value="sha"), \
         patch.object(outrider_local, "_gh_create_branch"), \
         patch.object(outrider_local, "_gh_put_file", side_effect=click.ClickException("422")), \
         patch.object(outrider_local, "_gh_open_pr") as open_pr, \
         patch.object(outrider_local, "_gh_delete_branch") as del_branch, \
         patch.object(outrider_local, "_gh_set_secret") as set_secret:
        with pytest.raises(click.ClickException):
            outrider_local.handle_outrider_setup_local(
                repo="o/r", interest_id=uid, auto_interest=False, mode="review",
                anthropic_key=None, skip_confirm=True, dry_run=False,
            )
    del_branch.assert_called_once()      # branch rolled back
    open_pr.assert_not_called()          # never got to the PR
    set_secret.assert_not_called()       # secrets never set (last step)


# ─── handler: happy path (review, default GITHUB_TOKEN auth) ─────────────────

def test_review_mode_enables_pr_creation_and_sets_secrets(monkeypatch):
    _base_patches(monkeypatch)
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_local, "_resolve_interest_id", return_value=uid), \
         patch.object(outrider_local, "_gh_available", return_value=True), \
         patch.object(outrider_local, "_gh_authenticated", return_value=True), \
         patch.object(outrider_local, "_gh_default_branch", return_value="main"), \
         patch.object(outrider_local, "_gh_branch_exists", return_value=False), \
         patch.object(outrider_local, "_gh_get_branch_sha", return_value="sha"), \
         patch.object(outrider_local, "_gh_create_branch"), \
         patch.object(outrider_local, "_gh_put_file"), \
         patch.object(outrider_local, "_gh_open_pr", return_value=("https://x/pull/1", 1)), \
         patch.object(outrider_local, "_gh_enable_pr_creation") as enable, \
         patch.object(outrider_local, "_gh_set_secret") as set_secret, \
         patch.object(outrider_local, "_gh_merge_pr") as merge:
        outrider_local.handle_outrider_setup_local(
            repo="o/r", interest_id=uid, auto_interest=False, mode="review",
            anthropic_key=None, skip_confirm=True, dry_run=False,
        )
    enable.assert_called_once()          # enables the repo's Actions-PR setting
    merge.assert_not_called()            # review mode doesn't merge
    names = {c.args[1] for c in set_secret.call_args_list}
    assert {"REMYX_API_KEY", "ANTHROPIC_API_KEY"} <= names
    # no GitHub token is ever stored as a secret
    assert not any("TOKEN" in n and n not in {"REMYX_API_KEY", "ANTHROPIC_API_KEY"}
                   for n in names)


# ─── dry-run + wiring ─────────────────────────────────────────────────────────

def test_dry_run_makes_no_gh_calls(monkeypatch):
    _base_patches(monkeypatch)
    with patch.object(outrider_local, "_gh_authenticated") as auth, \
         patch.object(outrider_local, "_resolve_interest_id") as ri, \
         patch.object(outrider_local, "_gh_create_branch") as cb, \
         patch.object(outrider_local, "_gh_set_secret") as ss:
        outrider_local.handle_outrider_setup_local(
            repo="o/r", interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False, mode="auto", anthropic_key=None,
            skip_confirm=True, dry_run=True,
        )
    for m in (auth, ri, cb, ss):
        m.assert_not_called()


@patch("remyxai.cli.commands.handle_outrider_setup_local")
def test_setup_local_wiring(mock_handler):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "setup-local", "--repo", "o/r",
        "--interest", "6a730cc4-010c-49ce-9c7f-6d9c59431739",
        "--mode", "review", "-y",
    ])
    assert result.exit_code == 0
    kwargs = mock_handler.call_args.kwargs
    assert kwargs["mode"] == "review"
    assert kwargs["skip_confirm"] is True


def test_setup_local_help_lists_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["outrider", "setup-local", "--help"])
    assert result.exit_code == 0
    for opt in ("--repo", "--interest", "--auto-interest", "--mode",
                "--anthropic-key", "--dry-run"):
        assert opt in result.output
    assert "--gh-pat" not in result.output   # dropped for v1


# ─── per-stage backend routing beyond z.ai ──────────────────────────────────
#
# `_provider_for_model` used to special-case GLM and call everything else
# Anthropic, so `--drafter-model kimi-k3` rendered the drafter against
# ANTHROPIC_API_KEY, never asked for MOONSHOT_API_KEY, and produced a repo that
# reported a clean install and then failed auth on its first run — the same
# class of bug as remyxai/remyx#550, on the no-App path.

class TestProviderInference:
    def test_infers_every_registered_backend(self):
        assert outrider_local.infer_provider("glm-5.2") == "zai"
        assert outrider_local.infer_provider("kimi-k3") == "moonshot"
        assert outrider_local.infer_provider("claude-opus-4-8") == "anthropic"

    def test_is_case_insensitive(self):
        assert outrider_local.infer_provider("Kimi-K3") == "moonshot"
        assert outrider_local.infer_provider("GLM-5.2") == "zai"

    def test_unknown_model_is_none_not_anthropic(self):
        """The caller warns instead of silently assuming a backend."""
        assert outrider_local.infer_provider("mystery-model-9") is None
        assert outrider_local.infer_provider("") is None
        assert outrider_local.infer_provider(None) is None

    def test_provider_for_model_falls_back_to_the_template_default(self):
        assert outrider_local._provider_for_model("mystery-model-9") == "anthropic"

    def test_kimi_no_longer_resolves_to_anthropic(self):
        assert outrider_local._provider_for_model("kimi-k3") == "moonshot"


class TestProvidersForStages:
    def test_unset_stages_count_as_the_template_default(self):
        assert outrider_local.providers_for_stages(None, None, None) == ["anthropic"]

    def test_kimi_drafter_pulls_in_moonshot(self):
        assert outrider_local.providers_for_stages("kimi-k3", None, None) == \
            ["anthropic", "moonshot"]

    def test_three_backends_at_once(self):
        assert outrider_local.providers_for_stages(
            "kimi-k3", "glm-5.2", "claude-opus-4-8") == \
            ["anthropic", "zai", "moonshot"]

    def test_all_stages_on_one_non_default_backend_drops_anthropic(self):
        assert outrider_local.providers_for_stages(
            "glm-5.2", "glm-5.2", "glm-4.6") == ["zai"]

    def test_unknown_models_are_reported(self):
        assert outrider_local._unknown_stage_models("kimi-k3", "nope-1", None) == \
            ["nope-1"]


class TestMoonshotStageRendering:
    def test_drafter_routes_at_moonshot(self):
        with patch.object(outrider_local, "_fetch_outrider_template",
                          return_value=_FAKE_DRAFTER_TEMPLATE):
            wf = outrider_local._render_drafter_workflow("uuid", model="kimi-k3")
        assert "ANTHROPIC_AUTH_TOKEN: ${{ secrets.MOONSHOT_API_KEY }}" in wf
        assert "ANTHROPIC_API_KEY:" not in wf          # mutually exclusive
        assert "ANTHROPIC_MODEL: kimi-k3" in wf
        assert "model-base-url: https://api.moonshot.ai/anthropic" in wf

    def test_refiner_gap_and_dispatch_route_at_moonshot(self):
        with patch.object(outrider_local, "_fetch_outrider_template",
                          return_value=_FAKE_REFINER_TEMPLATE):
            wf = outrider_local._render_refiner_workflow(
                gap_model="kimi-k3", refine_model="kimi-k3",
            )
        assert "https://api.moonshot.ai/anthropic/v1/messages" in wf
        assert '"model": "kimi-k3"' in wf
        assert "Bearer {os.environ['MOONSHOT_API_KEY']}" in wf
        assert "MOONSHOT_API_KEY: ${{ secrets.MOONSHOT_API_KEY }}" in wf
        assert "-f provider=moonshot" in wf
        assert "-f model=kimi-k3" in wf

    def test_zai_routing_is_unchanged(self):
        """The registry rewrite must not regress the path that already worked."""
        with patch.object(outrider_local, "_fetch_outrider_template",
                          return_value=_FAKE_DRAFTER_TEMPLATE):
            wf = outrider_local._render_drafter_workflow("uuid", model="glm-5.2")
        assert "ANTHROPIC_AUTH_TOKEN: ${{ secrets.ZAI_API_KEY }}" in wf
        assert "model-base-url: https://api.z.ai/api/anthropic" in wf


class TestTwoTierSecretCollection:
    """The install must push a secret for every backend its stages use."""

    def _run(self, monkeypatch, **kwargs):
        pushed = {}
        monkeypatch.setattr(outrider_local, "_resolve_interest_id",
                            lambda *a, **k: "6a730cc4-010c-49ce-9c7f-6d9c59431739")
        monkeypatch.setattr(outrider_local, "_gh_available", lambda: True)
        monkeypatch.setattr(outrider_local, "_gh_authenticated", lambda: True)
        monkeypatch.setattr(outrider_local, "_gh_set_secret",
                            lambda repo, name, value: pushed.__setitem__(name, value))
        monkeypatch.setattr(outrider_local, "_gh_default_branch", lambda r: "main")
        monkeypatch.setattr(outrider_local, "_gh_branch_exists", lambda r, b: False)
        monkeypatch.setattr(outrider_local, "_gh_get_branch_sha", lambda r, b: "sha")
        monkeypatch.setattr(outrider_local, "_gh_create_branch", lambda r, b, s: None)
        monkeypatch.setattr(outrider_local, "_gh_get_file_sha", lambda *a, **k: None)
        monkeypatch.setattr(outrider_local, "_gh_put_file", lambda *a, **k: None)
        monkeypatch.setattr(outrider_local, "_gh_open_pr",
                            lambda *a, **k: ("https://github.com/o/r/pull/1", 1))
        monkeypatch.setattr(outrider_local, "_gh_merge_pr", lambda r, n: True)
        monkeypatch.setattr(outrider_local, "_gh_enable_pr_creation", lambda r: None)
        monkeypatch.setattr(outrider_local, "_gh_dispatch", lambda r, b: True)
        monkeypatch.setattr(outrider_local, "_render_drafter_workflow",
                            lambda *a, **k: "drafter: yaml")
        monkeypatch.setattr(outrider_local, "_render_refiner_workflow",
                            lambda *a, **k: "refiner: yaml")
        monkeypatch.setattr(outrider_local, "_render_local_workflow",
                            lambda *a, **k: "main: yaml")
        opts = dict(
            repo="owner/name", interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False, mode="review", two_tier=True, no_cocoindex=True,
            anthropic_key=None, skip_confirm=True, dry_run=False,
        )
        opts.update(kwargs)
        outrider_local.handle_outrider_setup_local(**opts)
        return pushed

    def test_kimi_drafter_pushes_moonshot_api_key(self, monkeypatch):
        monkeypatch.setenv("REMYXAI_API_KEY", "rmx-test-key-000000000000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 24)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-" + "m" * 24)
        pushed = self._run(monkeypatch, drafter_model="kimi-k3")
        assert "MOONSHOT_API_KEY" in pushed, (
            "a Kimi drafter with no MOONSHOT_API_KEY is the reported failure"
        )
        assert "ANTHROPIC_API_KEY" in pushed      # refiner stages still default
        assert "REMYX_API_KEY" in pushed

    def test_all_three_backends_are_pushed(self, monkeypatch):
        monkeypatch.setenv("REMYXAI_API_KEY", "rmx-test-key-000000000000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 24)
        monkeypatch.setenv("ZAI_API_KEY", "zai-" + "z" * 24)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-" + "m" * 24)
        pushed = self._run(monkeypatch, drafter_model="kimi-k3",
                           refiner_model="glm-5.2",
                           refine_model="claude-opus-4-8")
        assert {"REMYX_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY",
                "MOONSHOT_API_KEY"} <= set(pushed)

    def test_single_backend_install_pushes_only_its_key(self, monkeypatch):
        monkeypatch.setenv("REMYXAI_API_KEY", "rmx-test-key-000000000000")
        monkeypatch.setenv("ZAI_API_KEY", "zai-" + "z" * 24)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        pushed = self._run(monkeypatch, drafter_model="glm-5.2",
                           refiner_model="glm-5.2", refine_model="glm-4.6")
        assert set(pushed) == {"REMYX_API_KEY", "ZAI_API_KEY"}

    def test_missing_key_for_a_used_backend_is_a_hard_error(self, monkeypatch):
        monkeypatch.setenv("REMYXAI_API_KEY", "rmx-test-key-000000000000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 24)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.setattr("click.prompt", lambda *a, **k: "")
        with pytest.raises(click.ClickException, match="MOONSHOT_API_KEY"):
            self._run(monkeypatch, drafter_model="kimi-k3")

    def test_dry_run_does_not_prompt_for_secrets(self, monkeypatch, capsys):
        monkeypatch.setenv("REMYXAI_API_KEY", "rmx-test-key-000000000000")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.setattr("click.prompt", lambda *a, **k: pytest.fail(
            "--dry-run must not prompt for secrets"))
        self._run(monkeypatch, drafter_model="kimi-k3", dry_run=True)
        out = capsys.readouterr().out
        assert "MOONSHOT_API_KEY (will prompt)" in out


# ─── the agent axis in the generated workflow ──────────────────────────────


def test_render_declares_the_agent_axis_with_every_known_agent():
    """`agent` is a choice input listing what the vendored matrix knows, so a
    new agent in the action reaches new installs without a CLI release."""
    from remyxai import agent_matrix

    wf = outrider_local._render_local_workflow("uuid")
    assert "      agent:" in wf
    for name in agent_matrix.known_agents():
        assert f"          - {name}" in wf, f"agent {name} not offered"
    assert "agent: ${{ inputs.agent }}" in wf


def test_render_defaults_the_agent_input_to_the_selected_agent():
    for agent in ("claude", "codex", "backboard"):
        wf = outrider_local._render_local_workflow("uuid", agent=agent)
        assert f"default: '{agent}'" in wf


def test_render_defaults_to_claude_when_no_agent_is_named():
    """Empty means Claude Code and always will — the pinned compatibility
    guarantee that keeps existing installs on the path they have today."""
    wf = outrider_local._render_local_workflow("uuid")
    assert "default: 'claude'" in wf


def test_render_unknown_agent_raises():
    with pytest.raises(ValueError, match="unknown agent"):
        outrider_local._render_local_workflow("uuid", agent="gemini-cli")


def test_render_references_every_credential_the_action_might_read():
    """Generated from the matrix, not hand-listed.

    The env block used to name three provider secrets literally, so a
    `codex` or `backboard` run on a setup-local install would have found no
    credential at all — the action would fail its preflight with the key it
    needed missing, on a workflow the CLI itself wrote.
    """
    from remyxai import agent_matrix

    wf = outrider_local._render_local_workflow("uuid")
    for agent in agent_matrix.known_agents():
        key = agent_matrix.agent_info(agent)["key_env"]
        assert f"{key}: ${{{{ secrets.{key} }}}}" in wf, f"missing {key}"
    for provider in agent_matrix.known_providers():
        secret = agent_matrix.provider_info(provider)["secret_env"]
        if secret:
            assert f"{secret}: ${{{{ secrets.{secret} }}}}" in wf, (
                f"missing {secret}"
            )


def test_every_backend_choice_actually_renders():
    """A choice list must promise exactly what the code behind it supports.

    Deriving `--backend` from the matrix (every provider serving
    anthropic-messages) put `openrouter` on the flag: click accepted it and
    the renderer then raised ValueError, because there is no `_STAGE_POLICY`
    row to render from. Caught before it shipped, pinned here.
    """
    assert outrider_local.TWO_TIER_BACKEND_CHOICES
    for backend in outrider_local.TWO_TIER_BACKEND_CHOICES:
        outrider_local._render_local_workflow("uuid", backend=backend)


def test_every_agent_choice_actually_renders():
    from remyxai.cli import outrider_actions

    assert outrider_actions.AGENT_CHOICES
    for agent in outrider_actions.AGENT_CHOICES:
        outrider_local._render_local_workflow("uuid", agent=agent)


def test_setup_local_agent_flag_reaches_the_rendered_workflow(monkeypatch):
    """`--agent codex` has to change the generated file, not just be accepted.

    Worth an end-to-end assertion rather than trusting the plumbing: the
    first cut of this wiring inserted `agent=agent` into the *bulk* call and
    left the single-repo one without it. It parsed — the insert landed inside
    a `dict(...)` where Python ignores indentation — and every test still
    passed, because nothing checked that the flag reached the rendered file.
    """
    from click.testing import CliRunner

    from remyxai.cli.commands import cli

    # setup-local prompts for the Remyx key before it renders anything.
    monkeypatch.setenv("REMYXAI_API_KEY", "test-key")
    monkeypatch.setenv("REMYX_API_KEY", "test-key")

    result = CliRunner().invoke(cli, [
        "outrider", "setup-local", "--repo", "owner/name",
        "--interest", "00000000-0000-0000-0000-000000000000",
        "--agent", "codex", "--dry-run", "--yes",
    ])
    assert result.exit_code == 0, result.output
    assert "default: 'codex'" in result.output
    assert "CODEX_API_KEY: ${{ secrets.CODEX_API_KEY }}" in result.output


# ─── gateway model ids ─────────────────────────────────────────────────────


def test_a_gateway_model_id_infers_no_direct_provider():
    """`z-ai/glm-5.3` is served by OpenRouter, not by z.ai.

    The prefix heuristic reads leading characters, so every namespaced id
    matched nothing and fell through to the template default — meaning
    `z-ai/glm-5.3` was treated as an Anthropic model and would have been
    rendered against ANTHROPIC_API_KEY with no base URL. Exactly the
    dead-on-arrival failure `infer_provider`'s docstring warns about, in a
    shape the prefix table could not see.
    """
    assert outrider_local.is_gateway_model("z-ai/glm-5.3")
    assert outrider_local.infer_provider("z-ai/glm-5.3") is None
    # A bare id still resolves — the heuristic is unchanged for those.
    assert outrider_local.infer_provider("glm-5.3") == "zai"


def test_a_gateway_id_is_a_gateway_id_even_when_the_namespace_looks_direct():
    """`anthropic/claude-3-haiku` is an OpenRouter id, not an Anthropic one.

    Resolving it to `anthropic` would happen to pick the right *secret* and
    the wrong *endpoint*, which is the worst kind of near-miss.
    """
    assert outrider_local.infer_provider("anthropic/claude-3-haiku") is None


def test_a_two_tier_stage_rejects_a_gateway_model_id(monkeypatch):
    """Certainly wrong, so it fails rather than warns — the install it would
    produce cannot authenticate on its first run."""
    import click
    import pytest as _pytest
    from click.testing import CliRunner

    from remyxai.cli.commands import cli

    monkeypatch.setenv("REMYXAI_API_KEY", "test-key")
    monkeypatch.setenv("REMYX_API_KEY", "test-key")
    result = CliRunner().invoke(cli, [
        "outrider", "setup-local", "--repo", "owner/name",
        "--interest", "00000000-0000-0000-0000-000000000000",
        "--two-tier", "--drafter-model", "z-ai/glm-5.3",
        "--dry-run", "--yes",
    ])
    assert result.exit_code != 0
    assert "cannot use a gateway model id" in result.output
    assert "z-ai/glm-5.3" in result.output


def test_an_unrecognized_bare_model_still_only_warns(monkeypatch):
    """It might be fine — a new Anthropic model name, say — so proceeding
    with a warning is the right call for these."""
    from click.testing import CliRunner

    from remyxai.cli.commands import cli

    monkeypatch.setenv("REMYXAI_API_KEY", "test-key")
    monkeypatch.setenv("REMYX_API_KEY", "test-key")
    result = CliRunner().invoke(cli, [
        "outrider", "setup-local", "--repo", "owner/name",
        "--interest", "00000000-0000-0000-0000-000000000000",
        "--two-tier", "--drafter-model", "some-new-model-9",
        "--dry-run", "--yes",
    ])
    assert result.exit_code == 0, result.output
    assert "can't tell which backend" in result.output


# ─── cocoindex is the action's job, not the template's ─────────────────────


def test_the_template_does_not_install_cocoindex_itself():
    """It used to, and that was wrong three ways once a run could use an
    agent other than Claude Code:

    * it symlinked into `~/.claude/skills` unconditionally, so a Codex or
      R-CLI run cloned a skill into a directory that agent never reads;
    * the ENVIRONMENTS.md it wrote told *every* agent that `ccc` was "a
      Claude Code skill", a route two of the three agents do not have;
    * the action installs cocoindex itself when `enable-cocoindex` is true
      (its default), so each install did the ~1GB install twice and wrote
      two different ENVIRONMENTS.md files.
    """
    wf = outrider_local._render_local_workflow("uuid")
    assert "pipx install" not in wf
    assert "~/.claude/skills" not in wf
    assert "ENVIRONMENTS.md" not in wf


def test_no_cocoindex_forwards_the_action_input():
    """`--no-cocoindex` now turns the action's own install off rather than
    omitting steps the template no longer has."""
    on = outrider_local._render_local_workflow("uuid", no_cocoindex=False)
    off = outrider_local._render_local_workflow("uuid", no_cocoindex=True)
    assert "enable-cocoindex: 'true'" in on
    assert "enable-cocoindex: 'false'" in off


def test_the_generated_workflow_says_nothing_claude_specific():
    """A workflow that can dispatch three agents must not describe one.

    The `provider` input called itself "which model provider to route
    Claude Code at", which reads as though the input does not apply when
    `agent` is codex or backboard. It applies to whichever agent is
    selected.
    """
    for agent in ("claude", "codex", "backboard"):
        wf = outrider_local._render_local_workflow("uuid", agent=agent)
        assert "route Claude Code at" not in wf
        assert "Claude Code skill" not in wf
