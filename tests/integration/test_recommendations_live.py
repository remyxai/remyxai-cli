"""
Live integration tests for the recommendations and interests API.
Requires REMYXAI_API_KEY to be set. Skipped automatically otherwise.

Run:
    REMYXAI_API_KEY=your_key pytest tests/integration/test_recommendations_live.py -v

The tests are designed to be non-destructive:
  - The interest created in test_create_interest is cleaned up in teardown.
  - Refresh tasks are triggered but we only poll briefly; we don't require
    them to complete within the test timeout (cold start can take 40-120s).
    Use --live-wait to block until completion if you want full end-to-end.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

# ─── skip entire module if no API key ────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not os.environ.get("REMYXAI_API_KEY"),
    reason="REMYXAI_API_KEY not set — skipping live integration tests",
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def created_interest():
    """
    Create a test Research Interest before the module runs,
    delete it after — regardless of test outcomes.
    """
    from remyxai.api.interests import create_interest, delete_interest

    interest = create_interest(
        name="[test] CLI Integration Test",
        context="Retrieval-augmented generation, hybrid search, dense retrieval",
        daily_count=2,
        is_active=False,   # inactive so it doesn't affect the real daily digest
    )
    yield interest

    # Teardown — best-effort
    try:
        delete_interest(interest["id"])
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Research Interests — CRUD
# ═════════════════════════════════════════════════════════════════════════════

class TestInterestsLive:

    def test_list_interests_returns_list(self):
        from remyxai.api.interests import list_interests
        interests = list_interests()
        assert isinstance(interests, list)
        # Every interest has the required envelope fields
        for i in interests:
            assert "id" in i
            assert "name" in i
            assert "daily_count" in i
            assert "is_active" in i
            assert _is_valid_uuid(i["id"])

    def test_create_interest_returns_full_object(self, created_interest):
        i = created_interest
        assert _is_valid_uuid(i["id"])
        assert i["name"] == "[test] CLI Integration Test"
        assert i["daily_count"] == 2
        assert i["is_active"] is False

    def test_get_interest_by_id(self, created_interest):
        from remyxai.api.interests import get_interest
        fetched = get_interest(created_interest["id"])
        assert fetched["id"] == created_interest["id"]
        assert fetched["name"] == created_interest["name"]

    def test_update_interest_daily_count(self, created_interest):
        from remyxai.api.interests import update_interest
        result = update_interest(created_interest["id"], daily_count=3)
        assert result["daily_count"] == 3
        # Other fields unchanged
        assert result["name"] == created_interest["name"]

    def test_update_interest_name_clears_pool(self, created_interest):
        from remyxai.api.interests import update_interest
        result = update_interest(
            created_interest["id"],
            name="[test] CLI Integration Test (renamed)",
        )
        assert "renamed" in result["name"]
        # pool_invalidated may be 0 if pool was empty — just check it's present or absent gracefully
        assert isinstance(result.get("pool_invalidated", 0), int)

    def test_toggle_interest(self, created_interest):
        from remyxai.api.interests import toggle_interest, get_interest
        original_state = get_interest(created_interest["id"])["is_active"]
        toggled = toggle_interest(created_interest["id"])
        assert toggled["is_active"] is not original_state
        # Toggle back
        toggle_interest(created_interest["id"])

    def test_created_interest_appears_in_list(self, created_interest):
        from remyxai.api.interests import list_interests
        interests = list_interests()
        ids = [i["id"] for i in interests]
        assert created_interest["id"] in ids


# ═════════════════════════════════════════════════════════════════════════════
# Recommendations — digest and list
# ═════════════════════════════════════════════════════════════════════════════

class TestRecommendationsLive:

    def test_digest_returns_expected_shape(self):
        from remyxai.api.recommendations import get_recommendations_digest
        data = get_recommendations_digest(period="week", limit=5)

        assert "date" in data
        assert "period" in data
        assert "interests" in data
        assert "total_papers" in data
        assert isinstance(data["interests"], list)
        assert isinstance(data["total_papers"], int)

    def test_digest_interests_have_recommendation_envelope(self):
        from remyxai.api.recommendations import get_recommendations_digest
        data = get_recommendations_digest(period="week", limit=5)

        for interest in data["interests"]:
            assert "id" in interest
            assert "name" in interest
            assert "count" in interest
            assert "recommendations" in interest
            assert isinstance(interest["recommendations"], list)

            for rec in interest["recommendations"]:
                # Every recommendation must have the source-agnostic envelope fields
                assert "recommendation_id" in rec, f"missing recommendation_id in {rec}"
                assert "source_type" in rec, f"missing source_type in {rec}"
                assert "resource_id" in rec, f"missing resource_id in {rec}"
                assert "title" in rec, f"missing title in {rec}"
                assert "url" in rec, f"missing url in {rec}"
                assert "relevance_score" in rec, f"missing relevance_score in {rec}"
                assert "resource" in rec, f"missing resource in {rec}"
                assert isinstance(rec["resource"], dict)
                assert 0.0 <= rec["relevance_score"] <= 1.0

    def test_list_recommended_flat(self):
        from remyxai.api.recommendations import list_recommended
        data = list_recommended(period="week", limit=10)

        assert "count" in data
        assert isinstance(data["count"], int)
        # Engine returns "papers" key on this endpoint
        recs = data.get("papers", data.get("recommendations", []))
        assert isinstance(recs, list)
        assert len(recs) == data["count"]

    def test_list_recommended_filter_by_interest(self):
        from remyxai.api.interests import list_interests
        from remyxai.api.recommendations import list_recommended

        interests = list_interests()
        if not interests:
            pytest.skip("No interests configured — skipping filter test")

        active = [i for i in interests if i.get("is_active")]
        if not active:
            pytest.skip("No active interests — skipping filter test")

        interest_id = active[0]["id"]
        data = list_recommended(interest_id=interest_id, period="week", limit=5)
        recs = data.get("papers", data.get("recommendations", []))

        # All returned recs should belong to the requested interest
        for rec in recs:
            assert rec.get("interest_id") == interest_id, (
                f"Expected interest_id={interest_id}, got {rec.get('interest_id')}"
            )

    def test_arxiv_paper_resource_has_expected_fields(self):
        from remyxai.api.recommendations import get_recommendations_digest
        data = get_recommendations_digest(period="week", limit=5)

        arxiv_recs = [
            rec
            for interest in data["interests"]
            for rec in interest["recommendations"]
            if rec.get("source_type") == "arxiv_paper"
        ]

        if not arxiv_recs:
            pytest.skip("No arxiv_paper recommendations in last week — skipping field check")

        for rec in arxiv_recs:
            r = rec["resource"]
            assert "arxiv_id" in r
            assert "authors" in r
            assert isinstance(r["authors"], list)
            assert "has_docker" in r
            assert isinstance(r["has_docker"], bool)


# ═════════════════════════════════════════════════════════════════════════════
# Refresh — trigger and poll
# ═════════════════════════════════════════════════════════════════════════════

class TestRefreshLive:

    def test_trigger_refresh_returns_tasks(self, created_interest):
        """
        Trigger a refresh for the test interest (inactive, so it won't
        pollute the real daily digest). Verify we get a task back.
        """
        from remyxai.api.interests import toggle_interest
        from remyxai.api.recommendations import trigger_recommendations_refresh

        # Temporarily activate so the refresh runs
        toggle_interest(created_interest["id"])
        try:
            result = trigger_recommendations_refresh(
                interest_id=created_interest["id"],
                num_results=2,
            )
        finally:
            toggle_interest(created_interest["id"])  # deactivate again

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        task = result["tasks"][0]
        assert _is_valid_uuid(task["task_id"])
        assert task["interest_id"] == created_interest["id"]
        assert task["status"] == "pending"

    def test_poll_task_returns_valid_status(self, created_interest):
        """
        Trigger a refresh, then poll it a few times.
        We don't require it to complete (cold start can take 40-120s)
        but the status must be a recognised value throughout.
        """
        from remyxai.api.interests import toggle_interest
        from remyxai.api.recommendations import (
            trigger_recommendations_refresh,
            poll_refresh_task,
        )

        toggle_interest(created_interest["id"])
        try:
            result = trigger_recommendations_refresh(
                interest_id=created_interest["id"],
                num_results=2,
            )
        finally:
            toggle_interest(created_interest["id"])

        task_id = result["tasks"][0]["task_id"]
        valid_statuses = {"pending", "running", "completed", "failed"}

        # Poll up to 3 times, 5s apart — enough to confirm the task is progressing
        for _ in range(3):
            status = poll_refresh_task(task_id)
            assert status["task_id"] == task_id
            assert status["status"] in valid_statuses
            assert isinstance(status.get("progress", 0), int)
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(5)

    def test_trigger_all_interests_refresh(self):
        """
        Trigger a refresh for ALL active interests (no interest_id).
        Verify we get one task per active interest.
        """
        from remyxai.api.interests import list_interests
        from remyxai.api.recommendations import trigger_recommendations_refresh

        active_interests = [i for i in list_interests() if i.get("is_active")]
        if not active_interests:
            pytest.skip("No active interests — skipping all-interests refresh test")

        result = trigger_recommendations_refresh()
        assert len(result["tasks"]) == len(active_interests)
        for task in result["tasks"]:
            assert _is_valid_uuid(task["task_id"])
            assert task["status"] == "pending"
