"""Round 1: cheap caps — #6 wall clock (soft), #45 safety-net defaults, #18 run_shell timeout cap."""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import openai

from agents.agent import Agent
from agents.tools import _run_shell


def make(use_openai: bool = True, **kwargs) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=True, fold_tail_tokens=0)
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    options.update(kwargs)
    return Agent(**options)


def overflow():
    resp = httpx.Response(400, request=httpx.Request("POST", "https://example.test/v1"))
    return openai.BadRequestError("too long", response=resp, body={"code": "context_length_exceeded"})


MEMORY = {"episode_memory": {"task_description": "t", "key_events": [], "current_progress": "p"},
          "working_memory": {"immediate_goal": "g", "current_challenges": "", "next_actions": []},
          "tool_memory": {"tools_used": [], "derived_rules": []}}


class CurrentStateTests(unittest.IsolatedAsyncioTestCase):
    """Pre-round-1 behaviour, pinned before the change (see 第 1 轮实施记录 for how each one changed)."""

    async def test_shell_timeout_has_no_cap(self):
        self.assertEqual(await _run_shell({"command": "echo hi", "timeout": 10 ** 9}), "hi\n")

    async def test_timeout_message_uses_requested_value(self):
        self.assertEqual(await _run_shell({"command": "sleep 2", "timeout": 100}), "Command timed out after 100ms")

    def test_no_wall_clock_parameter(self):
        with self.assertRaises(TypeError):
            make(max_wall_seconds=5)

    async def test_budget_exhausted_during_overflow_recovery_raises(self):
        agent = make()
        for i in range(3):
            agent._openai_messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        calls = 0

        async def stream():
            nonlocal calls
            calls += 1
            agent._require_model_budget()  # what _call_openai_stream's _do does first
            if calls == 1:
                raise overflow()
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}

        async def fold(*args, **kwargs):
            agent.max_cost_usd = 0  # budget runs out while the compaction side query is in flight
            return MEMORY
        agent._call_openai_stream = stream
        agent._generate_folded_session_memory = fold
        with patch("agents.agent.save_folded_session_memory"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Model request blocked: Cost limit reached"):
                await agent._chat_openai("go")


if __name__ == "__main__":
    unittest.main()
