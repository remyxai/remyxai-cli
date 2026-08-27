"""Tests for the `remyxai search` display handlers.

Covers the stats key mismatch found while verifying REMYX-286: the handler read
papers_* keys while the endpoint returns assets_*, so every count printed 0.
"""
import remyxai.cli.search_actions as sa


# The shape /api/v1.0/search/stats returns. Values are synthetic and chosen so
# the derived percentages are exact.
_STATS = {
    "total_assets": 1000,
    "assets_with_docker": 250,
    "assets_without_docker": 750,
    "recent_additions": 40,
    "categories": {"cs.CV": 600, "cs.LG": 400},
    "classified_papers": 500,
    "citation_coverage_pct": 50.0,
    "backfilled_papers": 0,
}


def _run_stats(monkeypatch, capsys, stats):
    monkeypatch.setattr(sa, "get_stats", lambda: stats)
    sa.handle_stats()
    return capsys.readouterr().out


def test_stats_reports_the_real_totals(monkeypatch, capsys):
    out = _run_stats(monkeypatch, capsys, _STATS)
    assert "Total Assets: 1000" in out
    assert "With Docker: 250" in out
    assert "Without Docker: 750" in out
    assert "Recent Additions (7 days): 40" in out


def test_stats_percentages_are_computed_off_the_total(monkeypatch, capsys):
    out = _run_stats(monkeypatch, capsys, _STATS)
    assert "25.0%" in out    # 250 / 1000
    assert "75.0%" in out    # 750 / 1000


def test_stats_does_not_read_the_papers_prefixed_keys(monkeypatch, capsys):
    """Guard against the regression: papers_* must not be resurrected."""
    stats = dict(_STATS)
    stats.update(papers_with_docker=1, total_papers=2, papers_without_docker=3)
    out = _run_stats(monkeypatch, capsys, stats)
    assert "Total Assets: 1000" in out
    # Exact-line matches: "With Docker: 1" is a prefix of "With Docker: 1000".
    lines = [ln.strip() for ln in out.splitlines()]
    assert "Total Assets: 2" not in lines
    assert not any(ln.startswith("With Docker: 1 ") for ln in lines)
    assert not any(ln.startswith("Without Docker: 3 ") for ln in lines)


def test_stats_survives_an_empty_catalog(monkeypatch, capsys):
    out = _run_stats(monkeypatch, capsys, {})
    assert "Total Assets: 0" in out
    assert "With Docker: 0" in out


def test_stats_lists_top_categories_by_count(monkeypatch, capsys):
    out = _run_stats(monkeypatch, capsys, _STATS)
    assert out.index("cs.CV: 600") < out.index("cs.LG: 400")


# ─── REMYX-287: --format json on search query / search list ──────────

import json

import pytest

from remyxai.api.search import Asset


def _asset(arxiv_id="2408.16245v5", **kw):
    fields = dict(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="an abstract",
        authors=["A. Author"],
        categories=["cs.LG", "cs.CV"],
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at="2026-05-26T00:00:00+00:00",
        github_url="https://github.com/org/repo",
        has_docker=True,
        docker_image=f"remyxai/{arxiv_id}:latest",
    )
    fields.update(kw)
    return Asset(**fields)


def _json_out(capsys):
    """Parse stdout as JSON — fails loudly if any human chrome leaked in."""
    out = capsys.readouterr().out
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout was not valid JSON ({e}):\n{out!r}")


# ── search list ──


def test_list_json_emits_a_parseable_envelope(monkeypatch, capsys):
    monkeypatch.setattr(sa, "list_assets", lambda **kw: {
        "assets": [_asset("2605.27296v1"), _asset("2605.27128v1")],
        "total": 137, "limit": 2, "offset": 0,
    })
    sa.handle_list(limit=2, output_format="json")
    body = _json_out(capsys)
    assert body["total"] == 137
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert [a["arxiv_id"] for a in body["assets"]] == ["2605.27296v1", "2605.27128v1"]


def test_list_json_carries_the_fields_the_ticket_asks_for(monkeypatch, capsys):
    monkeypatch.setattr(sa, "list_assets", lambda **kw: {
        "assets": [_asset()], "total": 1, "limit": 20, "offset": 0,
    })
    sa.handle_list(output_format="json")
    asset = _json_out(capsys)["assets"][0]
    for key in ("arxiv_id", "title", "github_url", "has_docker", "categories"):
        assert key in asset, f"missing {key}"
    assert asset["has_docker"] is True
    assert asset["categories"] == ["cs.LG", "cs.CV"]
    assert asset["docker_image"] == "remyxai/2408.16245v5:latest"


def test_list_json_prints_no_human_chrome(monkeypatch, capsys):
    """The header used to print before the request, corrupting stdout."""
    monkeypatch.setattr(sa, "list_assets", lambda **kw: {
        "assets": [_asset()], "total": 1, "limit": 20, "offset": 0,
    })
    sa.handle_list(output_format="json")
    out = capsys.readouterr().out
    assert "Recently Added" not in out
    assert "=" * 10 not in out
    assert "💡" not in out


def test_list_json_empty_result_is_still_valid_json(monkeypatch, capsys):
    monkeypatch.setattr(sa, "list_assets", lambda **kw: {
        "assets": [], "total": 0, "limit": 20, "offset": 0,
    })
    sa.handle_list(has_docker=True, output_format="json")
    body = _json_out(capsys)
    assert body["assets"] == []
    assert body["total"] == 0


def test_list_text_mode_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(sa, "list_assets", lambda **kw: {
        "assets": [_asset()], "total": 1, "limit": 20, "offset": 0,
    })
    sa.handle_list(output_format="text")
    out = capsys.readouterr().out
    assert "📚 Recently Added Research Assets" in out
    assert "🐳" in out


def test_list_json_error_goes_to_stderr_and_exits_nonzero(monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sa, "list_assets", boom)
    with pytest.raises(SystemExit) as exc:
        sa.handle_list(output_format="json")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "stdout must stay clean for a jq consumer"
    assert "engine exploded" in captured.err


def test_list_text_error_stays_on_stdout_and_does_not_exit(monkeypatch, capsys):
    """Text-mode behaviour is deliberately left alone."""
    def boom(**kw):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sa, "list_assets", boom)
    sa.handle_list(output_format="text")
    assert "❌ List failed" in capsys.readouterr().out


# ── search query ──


def test_query_json_emits_a_parseable_envelope(monkeypatch, capsys):
    monkeypatch.setattr(sa, "search_assets", lambda **kw: {
        "assets": [_asset("2601.04052v2")],
        "total": 1, "query": "vision language model", "strategy": "hybrid",
    })
    sa.handle_search("vision language model", output_format="json")
    body = _json_out(capsys)
    assert body["query"] == "vision language model"
    assert body["strategy"] == "hybrid"
    assert body["total"] == 1
    assert body["assets"][0]["arxiv_id"] == "2601.04052v2"


def test_query_json_prints_no_human_chrome(monkeypatch, capsys):
    monkeypatch.setattr(sa, "search_assets", lambda **kw: {
        "assets": [_asset()], "total": 1, "query": "q", "strategy": "hybrid",
    })
    sa.handle_search("q", output_format="json")
    out = capsys.readouterr().out
    assert "🔍 Searching for" not in out
    assert "Filters:" not in out


def test_query_json_empty_result_is_still_valid_json(monkeypatch, capsys):
    monkeypatch.setattr(sa, "search_assets", lambda **kw: {
        "assets": [], "total": 0, "query": "q", "strategy": "hybrid",
    })
    sa.handle_search("q", has_docker=True, output_format="json")
    assert _json_out(capsys)["assets"] == []


def test_query_text_mode_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(sa, "search_assets", lambda **kw: {
        "assets": [_asset()], "total": 1, "query": "q", "strategy": "hybrid",
    })
    sa.handle_search("q", output_format="text")
    out = capsys.readouterr().out
    assert "🔍 Searching for: 'q'" in out
    assert "Filters:" in out


def test_query_json_error_goes_to_stderr_and_exits_nonzero(monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sa, "search_assets", boom)
    with pytest.raises(SystemExit) as exc:
        sa.handle_search("q", output_format="json")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "engine exploded" in captured.err


# ── CLI wiring: the option must reach the handler ────────────────────

from click.testing import CliRunner

from remyxai.cli.commands import cli


@pytest.mark.parametrize("argv,handler", [
    (["search", "list", "--format", "json"], "handle_list"),
    (["search", "list", "-f", "json"], "handle_list"),
    (["search", "query", "q", "--format", "json"], "handle_search"),
    (["search", "query", "q", "-f", "json"], "handle_search"),
])
def test_format_option_is_passed_through(monkeypatch, argv, handler):
    seen = {}
    import remyxai.cli.commands as cmds
    monkeypatch.setattr(cmds, handler, lambda *a, **kw: seen.update(kw))
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert seen.get("output_format") == "json"


@pytest.mark.parametrize("argv,handler", [
    (["search", "list"], "handle_list"),
    (["search", "query", "q"], "handle_search"),
])
def test_format_defaults_to_text(monkeypatch, argv, handler):
    seen = {}
    import remyxai.cli.commands as cmds
    monkeypatch.setattr(cmds, handler, lambda *a, **kw: seen.update(kw))
    assert CliRunner().invoke(cli, argv).exit_code == 0
    assert seen.get("output_format") == "text"


@pytest.mark.parametrize("argv", [
    ["search", "list", "--format", "yaml"],
    ["search", "query", "q", "--format", "csv"],
])
def test_unsupported_format_is_rejected(argv):
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_format_json_is_advertised_in_help():
    for argv in (["search", "list", "--help"], ["search", "query", "--help"]):
        out = CliRunner().invoke(cli, argv).output
        assert "--format" in out and "json" in out


# ─── search stats / search info --format json ────────────────────────


def test_stats_json_emits_the_payload_verbatim(monkeypatch, capsys):
    monkeypatch.setattr(sa, "get_stats", lambda: _STATS)
    sa.handle_stats(output_format="json")
    assert _json_out(capsys) == _STATS


def test_stats_json_prints_no_human_chrome(monkeypatch, capsys):
    monkeypatch.setattr(sa, "get_stats", lambda: _STATS)
    sa.handle_stats(output_format="json")
    out = capsys.readouterr().out
    assert "📊" not in out
    assert "Total Assets:" not in out
    assert "💡" not in out


def test_stats_json_error_goes_to_stderr_and_exits_nonzero(monkeypatch, capsys):
    def boom():
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sa, "get_stats", boom)
    with pytest.raises(SystemExit) as exc:
        sa.handle_stats(output_format="json")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "engine exploded" in captured.err


def test_info_json_emits_the_asset(monkeypatch, capsys):
    monkeypatch.setattr(sa, "get_asset", lambda aid: _asset(aid))
    sa.handle_info("2408.16245v5", output_format="json")
    body = _json_out(capsys)
    assert body["arxiv_id"] == "2408.16245v5"
    assert body["docker_image"] == "remyxai/2408.16245v5:latest"


def test_info_json_missing_asset_does_not_print_a_tip_to_stdout(monkeypatch, capsys):
    """This shipped broken: the not-found path wrote human text to stdout and
    exited 0, so a script could not tell a miss from a hit."""
    monkeypatch.setattr(sa, "get_asset", lambda aid: None)
    with pytest.raises(SystemExit) as exc:
        sa.handle_info("9999.99999v1", output_format="json")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not found" in captured.err
    assert "💡" not in captured.err


def test_info_text_missing_asset_keeps_the_tip(monkeypatch, capsys):
    monkeypatch.setattr(sa, "get_asset", lambda aid: None)
    sa.handle_info("9999.99999v1", output_format="text")
    out = capsys.readouterr().out
    assert "not found" in out
    assert "💡 Tip" in out


def test_info_json_error_goes_to_stderr_and_exits_nonzero(monkeypatch, capsys):
    def boom(aid):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(sa, "get_asset", boom)
    with pytest.raises(SystemExit) as exc:
        sa.handle_info("2408.16245v5", output_format="json")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "engine exploded" in captured.err


@pytest.mark.parametrize("argv,handler", [
    (["search", "stats", "--format", "json"], "handle_stats"),
    (["search", "stats", "-f", "json"], "handle_stats"),
    (["search", "info", "2408.16245v5", "--format", "json"], "handle_info"),
    (["search", "info", "2408.16245v5", "-f", "json"], "handle_info"),
])
def test_format_option_reaches_stats_and_info(monkeypatch, argv, handler):
    seen = {}
    import remyxai.cli.commands as cmds
    monkeypatch.setattr(cmds, handler, lambda *a, **kw: seen.update(kw))
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert seen.get("output_format") == "json"


def test_every_search_subcommand_supports_format_json():
    """The whole group should be scriptable, not three quarters of it."""
    runner = CliRunner()
    for sub in ("query", "list", "info", "stats"):
        out = runner.invoke(cli, ["search", sub, "--help"]).output
        assert "--format" in out, f"search {sub} lacks --format"
        assert "json" in out, f"search {sub} lacks a json choice"
