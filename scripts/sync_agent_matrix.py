#!/usr/bin/env python3
"""Refresh ``remyxai/_agent_matrix.py`` from the action's published matrix.

The agent/provider compatibility matrix is generated in ``remyxai/outrider``
(``scripts/gen_agent_matrix.py`` → ``docs/agent-matrix.json``) and consumed by
three repos: the action itself, this CLI, and the engine. Before this script
existed, the CLI kept its own hand-written ``_BACKEND_REGISTRY`` — which drifted,
because it was the copy a human edited.

**This file is the only thing that should ever write** ``remyxai/_agent_matrix.py``.

Usage::

    # from a local checkout of the action (preferred while developing)
    python scripts/sync_agent_matrix.py --from ../outrider/docs/agent-matrix.json

    # from the published artifact
    python scripts/sync_agent_matrix.py

    # CI: fail if the vendored copy differs from the source
    python scripts/sync_agent_matrix.py --check --from ../outrider/docs/agent-matrix.json

Why a generated *Python module* rather than a vendored ``.json``: the CLI ships
as a wheel with no ``package_data``, so a data file would have to be wired into
the build and would break the CLI at import time on any wheel built without it.
A ``.py`` ships automatically and cannot go missing. The JSON text is embedded
verbatim so the vendored copy stays byte-comparable against the source.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "remyxai" / "_agent_matrix.py"

PUBLISHED_URL = (
    "https://raw.githubusercontent.com/remyxai/outrider/main/"
    "docs/agent-matrix.json"
)

HEADER = '''\
"""Vendored copy of the action's agent/provider compatibility matrix.

GENERATED FILE — do not hand-edit. Refresh with::

    python scripts/sync_agent_matrix.py --from <path-to>/docs/agent-matrix.json

Source of truth: remyxai/outrider :: docs/agent-matrix.json, itself generated
from ``src/agents/providers.py``. Read this through
:mod:`remyxai.agent_matrix`, which adds the query helpers — nothing should
import ``MATRIX`` directly.

Vendored from: {origin}
"""
import json

# The source artifact's text, embedded verbatim so this file stays
# byte-comparable against it (see scripts/sync_agent_matrix.py --check).
_RAW = r"""
{raw}
"""

MATRIX = json.loads(_RAW)
'''


def _fetch(origin: str) -> str:
    if origin.startswith(("http://", "https://")):
        with urllib.request.urlopen(origin, timeout=30) as fh:  # noqa: S310
            return fh.read().decode("utf-8")
    return Path(origin).read_text(encoding="utf-8")


def render(raw: str, origin: str) -> str:
    """Build the module text for an artifact's raw JSON."""
    # Validate before embedding: a truncated download or an HTML error page
    # would otherwise be vendored and only fail at CLI import time.
    matrix = json.loads(raw)
    for key in ("agents", "providers", "pairs"):
        if key not in matrix:
            raise SystemExit(
                f"refusing to vendor: artifact has no {key!r} key "
                f"(got {sorted(matrix)}) — is {origin} the right source?"
            )
    if '"""' in raw:
        # Would terminate the embedding literal early.
        raise SystemExit("refusing to vendor: artifact contains a triple quote")
    return HEADER.format(origin=origin, raw=raw.rstrip("\n"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from", dest="origin", default=PUBLISHED_URL,
        help="Path or URL to agent-matrix.json (default: the published copy).",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if the vendored copy is stale; write nothing.",
    )
    args = ap.parse_args()

    try:
        raw = _fetch(args.origin)
    except OSError as exc:
        print(f"could not read {args.origin}: {exc}", file=sys.stderr)
        return 2

    rendered = render(raw, args.origin)
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""

    if args.check:
        # Compare the parsed matrix, not the file text: the header records
        # which origin it came from, so a local path and the published URL
        # render different text from identical data. The *data* is what has
        # to match.
        if _embedded(current) == json.loads(raw):
            print(f"{TARGET.relative_to(ROOT)} is up to date")
            return 0
        print(
            f"{TARGET.relative_to(ROOT)} is STALE against {args.origin}.\n"
            f"Refresh it:\n"
            f"  python scripts/sync_agent_matrix.py --from {args.origin}",
            file=sys.stderr,
        )
        return 1

    TARGET.write_text(rendered, encoding="utf-8")
    matrix = json.loads(raw)
    print(
        f"wrote {TARGET.relative_to(ROOT)} "
        f"({len(matrix['agents'])} agents, {len(matrix['providers'])} providers, "
        f"{len(matrix['pairs'])} pairs) from {args.origin}"
    )
    return 0


def _embedded(module_text: str) -> object:
    """The matrix currently vendored, or None when there isn't one."""
    if not module_text:
        return None
    marker = '_RAW = r"""\n'
    try:
        start = module_text.index(marker) + len(marker)
        end = module_text.index('\n"""', start)
    except ValueError:
        return None
    try:
        return json.loads(module_text[start:end])
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
