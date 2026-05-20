import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Drop the cached ``Settings`` between tests.

    Many tests flip env vars via ``monkeypatch`` or direct assignment; if the
    cache survives, the next test sees stale config. Clearing both before and
    after the test isolates each one.
    """
    from core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
