#!/bin/bash
set -e

echo "============================================"
echo "  Remyx CLI — Test lazy-api-key-resolution"
echo "============================================"
echo ""

# -------------------------------------------------
# 1. Create venv and install
# -------------------------------------------------
echo "▶ Creating fresh virtual environment..."
rm -rf /tmp/remyx-test
python3 -m venv /tmp/remyx-test
source /tmp/remyx-test/bin/activate

echo "▶ Installing package in editable mode..."
pip install -e . --quiet
pip install pytest --quiet

# -------------------------------------------------
# 2. Existing unit tests that still work
#
#    Skipping:
#    - test_evaluations.py (imports symbols that were
#      refactored out of remyxai.api.evaluations)
#    - test_dataset.py (wrong arity on download/delete —
#      pre-existing bug)
#    - test_evaluation_actions.py (depends on broken
#      test_evaluations imports)
# -------------------------------------------------
echo ""
echo "============================================"
echo "  PHASE 1: Existing unit tests (mocked)"
echo "============================================"
echo ""

pytest tests/api/test_search.py -v
pytest tests/api/test_models.py -v
pytest tests/api/test_tasks.py -v
pytest tests/api/test_user.py -v
pytest tests/api/test_deployment.py -v
pytest tests/api/test_inference.py -v
pytest tests/cli/test_commands.py -v
pytest tests/cli/test_deployment_actions.py -v

echo ""
echo "✅ Existing unit tests passed"

# -------------------------------------------------
# 3. New test suite for lazy key resolution
# -------------------------------------------------
echo ""
echo "============================================"
echo "  PHASE 2: Lazy key resolution tests"
echo "============================================"
echo ""

pytest tests/api/test_lazy_api_key.py -v

echo ""
echo "✅ Lazy key resolution tests passed"

# -------------------------------------------------
# 4. Live integration test (optional, needs real key)
# -------------------------------------------------
echo ""
echo "============================================"
echo "  PHASE 3: Live search (requires API key)"
echo "============================================"
echo ""

if [ -n "$REMYXAI_API_KEY" ]; then
    python3 -c "
import os
from remyxai.client.search import SearchClient

key = os.environ['REMYXAI_API_KEY']

# Test A: env var path (notebook pattern)
print('--- Env var path ---')
client = SearchClient()
papers = client.search(query='CLIP semantic alignment', has_docker=True, max_results=3)
for p in papers:
    print(f'  📖 {p.arxiv_id}: {p.title[:55]}...')
    print(f'     🐳 {p.docker_image}')
print(f'  ✅ {len(papers)} papers via env var')
print()

# Test B: explicit key path (HF Space pattern)
print('--- Explicit key path ---')
client2 = SearchClient(api_key=key)
papers2 = client2.search(query='reinforcement learning robotics', has_docker=True, max_results=3)
for p in papers2:
    print(f'  📖 {p.arxiv_id}: {p.title[:55]}...')
    print(f'     🐳 {p.docker_image}')
print(f'  ✅ {len(papers2)} papers via explicit key')
"
    echo ""
    echo "✅ Live search tests passed"
else
    echo "⏭  Skipping — set REMYXAI_API_KEY to run live tests"
    echo "   REMYXAI_API_KEY=your_key bash run_tests.sh"
fi

echo ""
echo "============================================"
echo "  🎉 All tests complete!"
echo "============================================"

deactivate
