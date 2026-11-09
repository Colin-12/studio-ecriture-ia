"""Memory system components for Phase 1."""

from .database import get_engine, get_session, init_db

# High-level querying is available from memory.continuity_checker. Importing it
# here would make even database/model imports depend on retrieval's bootstrap.
__all__ = ["get_engine", "get_session", "init_db"]
