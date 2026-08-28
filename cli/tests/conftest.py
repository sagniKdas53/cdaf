"""Keep the suite off the developer's real pricing cache.

`calculate_cost` prefers a synced price over the static table, so a populated
~/.cache/cdaf/pricing.json would otherwise change the cost numbers these tests
assert on.
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_pricing_cache():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CDAF_PRICING_CACHE"] = os.path.join(tmp, "pricing.json")
        yield
        os.environ.pop("CDAF_PRICING_CACHE", None)
