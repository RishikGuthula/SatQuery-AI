"""
Tests for Conversational Session Context across queries.
"""

import pytest
from core.models import SessionContext, ConversationTurn
from agent.controller import process_query


class TestSessionContext:
    """Test multi-turn session memory."""

    def test_session_context_turn_recording(self):
        ctx = SessionContext()
        ctx.add_turn(
            query="Find water bodies",
            answer="Water coverage: 10%",
            tool_used="water_detection",
            metadata={"water_percent": 10.0},
        )

        assert len(ctx.history) == 1
        assert ctx.last_query == "Find water bodies"
        assert ctx.last_tool_used == "water_detection"

    def test_session_history_limit(self):
        ctx = SessionContext()
        for i in range(15):
            ctx.add_turn(query=f"Query {i}", answer=f"Answer {i}", tool_used="tool")

        # History should be bounded at max 10
        assert len(ctx.history) == 10
        assert ctx.history[-1].query == "Query 14"

    def test_controller_preserves_and_updates_session(self, rgb_bytes):
        ctx = SessionContext()
        res1 = process_query("Find water", rgb_bytes, session_context=ctx)
        assert len(res1.session_context.history) == 1

        res2 = process_query("Detect vegetation", rgb_bytes, session_context=res1.session_context)
        assert len(res2.session_context.history) == 2
        assert res2.session_context.history[0].query == "Find water"
        assert res2.session_context.history[1].query == "Detect vegetation"
