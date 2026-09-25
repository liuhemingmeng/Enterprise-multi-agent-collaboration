"""Pytest bootstrap for the P2 suite.

Runs before any test module is imported, so the settings below are in place
by the time ``p2_agent.settings`` reads the environment.
"""

import os

# --- keep the suite offline and deterministic ------------------------------
# Without these the workflow would try to call the real P1 retrieval API and a
# real LLM whenever a developer has a populated .env, making results depend on
# network state instead of on the code under test.
os.environ.setdefault("RAG_API_KEY", "")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("LLM_BASE_URL", "")
os.environ.setdefault("P1_RAG_BASE_URL", "")

# --- rate limiting is a production concern ---------------------------------
# Every TestClient request arrives from the same peer IP, so an enabled limiter
# would reject legitimate assertions partway through the suite.  The limiter
# itself is covered by tests/test_rate_limit.py, which turns it on explicitly.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
