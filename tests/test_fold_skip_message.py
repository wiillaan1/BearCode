"""Audit item #44: when the whole history sits in the kept tail, both compaction entry points say so."""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import AsyncMock

from agents.agent import FOLD_SKIPPED_ALL_IN_TAIL, Agent
from agents.tool_outcome import is_tool_failure


def short_agent(use_openai: bool) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=True, model="gpt-4o")  # default tail budget
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    agent = Agent(**options)
    msgs = agent._openai_messages if use_openai else agent._anthropic_messages
    for i in range(3):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}" if use_openai else [{"type": "text", "text": f"answer {i}"}]})
    agent._generate_folded_session_memory = AsyncMock()
    return agent


class FoldSkipMessageTests(unittest.TestCase):
    def test_tool_and_manual_compact_share_the_real_reason(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = short_agent(use_openai)
                result = asyncio.run(agent._execute_compact_context_tool({"reason": "stuck"}))
                self.assertEqual(result, FOLD_SKIPPED_ALL_IN_TAIL)
                self.assertNotIn("insufficient history", result)
                self.assertFalse(is_tool_failure("compact_context", result))  # #5: prefix rule replaced _looks_like_tool_failure
                out = io.StringIO()
                with redirect_stdout(out):
                    asyncio.run(agent.compact())
                self.assertIn(" ".join(FOLD_SKIPPED_ALL_IN_TAIL.split()[:6]), " ".join(out.getvalue().split()))
                self.assertEqual(agent._fold_count, 0)
                agent._generate_folded_session_memory.assert_not_awaited()

    def test_failed_generation_keeps_the_old_message(self):
        agent = short_agent(True)
        agent.fold_tail_tokens = 0  # head not empty: a real fold is attempted and fails
        agent._generate_folded_session_memory = AsyncMock(return_value=None)
        result = asyncio.run(agent._execute_compact_context_tool({}))
        self.assertIn("insufficient history or memory generation failed validation", result)
        self.assertEqual(agent._fold_count, 0)


if __name__ == "__main__":
    unittest.main()
