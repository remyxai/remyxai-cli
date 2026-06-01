"""Shared pytest setup.

Sets `REMYXAI_API_KEY` before any test module imports `remyxai.api`,
so the module-level `HEADERS` dict (built at import time in
`remyxai/api/__init__.py`) has a non-empty Bearer token. Per-test
autouse fixtures elsewhere can't fix this on their own because they
run after collection-time imports.
"""
import os

os.environ.setdefault("REMYXAI_API_KEY", "test-key-collection-time")
