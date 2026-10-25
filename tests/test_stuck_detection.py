"""Audit item #5: stuck detection."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

from agents.agent import Agent


def make(use_openai: bool = True, **kwargs) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=True, fold_tail_tokens=0)
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    options.update(kwargs)
    return Agent(**options)


class CurrentStateTests(unittest.TestCase):
    """Pre-#5 behaviour, pinned before the change (see #5 实施记录 for how each one changed)."""

    def test_fold_resets_counters(self):
        agent = make()
        for _ in range(3):
            agent._record_tool_outcome("read_file", False)
        self.assertEqual(agent._tool_error_streak, 3)
        agent._record_fold_event()
        self.assertEqual(agent._tool_error_streak, 0)
        self.assertEqual(agent._same_tool_repeat_count, 0)

    def test_compact_context_counted_twice(self):
        agent = make()
        for i in range(3):
            agent._openai_messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        agent._generate_folded_session_memory = AsyncMock(return_value=None)  # skipped → recorded inside the tool
        result = asyncio.run(agent._execute_compact_context_tool({}))
        agent._record_tool_outcome("compact_context", not agent._looks_like_tool_failure("compact_context", result, ""))
        self.assertEqual(agent._same_tool_repeat_count, 2)  # one call, counted twice

    def test_error_word_in_successful_read_counts_as_failure(self):
        agent = make()
        self.assertTrue(agent._looks_like_tool_failure("read_file", "   1 | raise ValueError('error')", ""))


if __name__ == "__main__":
    unittest.main()
