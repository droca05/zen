"""
db.py — Supabase client singleton for the entire app.

Import with:
    from db import get_sb

Returns the Supabase client when SUPABASE_URL + SUPABASE_KEY are set,
or None when they are absent — so every caller can fall back to the
local JSON files and the app keeps working without a DB connection.
"""
from __future__ import annotations
import os

# Load .env if python-dotenv is installed (it is in requirements.txt).
# This must run before os.environ.get() calls.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

_client = None


def get_sb():
    """Return the Supabase client, or None if not configured."""
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if url and key:
        try:
            from supabase import create_client
            # Strip any path suffix the user may have copied from the dashboard
            url = url.rstrip("/").removesuffix("/rest/v1")
            _client = create_client(url, key)
        except Exception as e:
            print(f"[db] Supabase init failed: {e}. Falling back to JSON files.")
    return _client
