"""Pytest configuration and fixtures."""
import sys
from unittest.mock import MagicMock

# Mock psycopg and related packages before any storage imports
sys.modules['psycopg'] = MagicMock()
sys.modules['psycopg.rows'] = MagicMock()
sys.modules['pgvector'] = MagicMock()
sys.modules['pgvector.psycopg'] = MagicMock()
sys.modules['psycopg_pool'] = MagicMock()
