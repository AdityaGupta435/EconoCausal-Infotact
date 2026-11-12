"""Pytest root configuration and test isolation fixtures for EconoCausal."""

import mongomock
import pytest
from backend.database import set_db_override, init_db_indexes

# Initialize immediate module-level mock DB so any TestClient(app) at import time uses mock DB
_default_mock_client = mongomock.MongoClient()
_default_mock_db = _default_mock_client["econocausal_isolated_test"]
set_db_override(_default_mock_db)
init_db_indexes(_default_mock_db)


@pytest.fixture(autouse=True)
def isolated_test_db():
    """Isolate each test with a clean in-memory mongomock database."""
    client = mongomock.MongoClient()
    mock_db = client["econocausal_test"]
    set_db_override(mock_db)
    init_db_indexes(mock_db)
    yield mock_db
    set_db_override(_default_mock_db)
