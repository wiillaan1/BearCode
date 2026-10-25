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


class FormerStateTests(unittest.IsolatedAsyncioTestCase):
    """Pinned before round 1 (commit cb4ad22) as CurrentStateTests; how each one changed:
    1. shell timeout had no cap                     → capped at 600000ms (note appended at the end)
    2. timeout message used the requested value     → unchanged within the cap (kept as is below)
    3. no max_wall_seconds parameter                → parameter exists
    4. budget exhausted during #2 recovery raised   → graceful exit (BudgetExceededError caught in the loop)
    """

    async def test_shell_timeout_is_capped(self):
        result = await _run_shell({"command": "echo hi", "timeout": 10 ** 9})
        self.assertTrue(result.startswith("hi\n"))
        self.assertIn("requested 1000000000ms exceeds the 600000ms limit", result)

    async def test_timeout_message_uses_requested_value(self):
        self.assertEqual(await _run_shell({"command": "sleep 2", "timeout": 100}), "Command timed out after 100ms")

    def test_wall_clock_parameter_exists(self):
        self.assertEqual(make(max_wall_seconds=5).max_wall_seconds, 5)

    async def test_budget_exhausted_during_overflow_recovery_exits_gracefully(self):
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
            await agent._chat_openai("go")
        self.assertEqual(agent.last_finish_reason, "budget_exceeded")


# ---------------------------------------------------------------- new behaviour (round 1)

import json
import os
import time
from types import SimpleNamespace

from agents.agent import SAFETY_NET_MAX_TURNS, SAFETY_NET_MAX_WALL_SECONDS, BudgetExceededError
from agents.tool_outcome import is_tool_failure


class LoopBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        for target in (patch("agents.agent.save_folded_session_memory"), patch("agents.agent.save_dropped_messages"),
                       patch("agents.verify_on_stop._settings_files", return_value=[])):
            target.start()
            self.addCleanup(target.stop)

    def agent(self, use_openai, **kwargs):
        agent = make(use_openai, **kwargs)
        agent._auto_save = lambda strict=False: None
        return agent

    async def drive(self, agent, steps, request="go", via_chat=True, on_call=None):
        """steps: list of tool lists [(name, inp, result)] | str final text. on_call(n) runs before model call n."""
        steps, queue, seen = list(steps), [], []
        ids = iter(range(10_000))

        async def run_checked(name, inp):
            return queue.pop(0)

        async def stream():
            seen.append(json.loads(json.dumps(agent._openai_messages if agent.use_openai else agent._anthropic_messages)))
            agent._require_model_budget()  # what the real _call_*_stream does first
            if on_call:  # after the in-call check: simulates time passing while the model is generating
                on_call(len(seen))
            step = steps.pop(0) if steps else "done"
            if isinstance(step, Exception):
                raise step
            if isinstance(step, str):
                agent._emit_text(step)
                if agent.use_openai:
                    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": step}}]}
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=step)],
                                       usage=SimpleNamespace(input_tokens=1, output_tokens=1))
            calls = [(f"c{next(ids)}", n, i) for n, i, r in step]
            queue.extend(r for _, _, r in step)
            if agent.use_openai:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": c, "type": "function", "function": {"name": n, "arguments": json.dumps(i)}} for c, n, i in calls]}}]}
            return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   content=[SimpleNamespace(type="tool_use", id=c, name=n, input=i) for c, n, i in calls])

        agent._run_checked_tool = run_checked
        setattr(agent, "_call_openai_stream" if agent.use_openai else "_call_anthropic_stream", stream)
        if via_chat:
            agent._turn_output_buffer = None
            await agent.chat(request)
        else:
            await (agent._chat_openai if agent.use_openai else agent._chat_anthropic)(request)
        return seen

    @staticmethod
    def msgs(agent):
        return agent._openai_messages if agent.use_openai else agent._anthropic_messages

    def pairs_ok(self, agent):
        msgs = self.msgs(agent)
        if agent.use_openai:
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    got = {x.get("tool_call_id") for x in msgs[i + 1:i + 1 + len(m["tool_calls"])]}
                    self.assertEqual(got, {c["id"] for c in m["tool_calls"]})
        else:
            self.assertEqual(agent._normalize_anthropic_messages(msgs), msgs)

    async def both(self, check):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                await check(use_openai)


READ = ("read_file", {"file_path": "a.txt"}, "1\tx")


class WallClockTests(LoopBase):
    async def test_expiry_exits_gracefully_at_next_checkpoint(self):
        async def check(use_openai):
            agent = self.agent(use_openai, max_wall_seconds=60)

            def expire(n):
                if n == 2:
                    agent._turn_deadline = time.monotonic() - 1
            seen = await self.drive(agent, [[READ], [READ], "never"], on_call=expire)
            # the clock runs out while call 2 is generating: its tool round is skipped by the pre-tool check,
            # and no third request happens
            self.assertEqual(len(seen), 2)
            self.assertEqual(agent.last_finish_reason, "budget_exceeded")
            self.assertIn("Wall-clock limit reached", self.out.getvalue())
            self.assertIn("Tool execution skipped: Wall-clock limit reached", json.dumps(self.msgs(agent)))
            self.pairs_ok(agent)
        await self.both(check)

    async def test_loop_top_check(self):
        async def check(use_openai):
            agent = self.agent(use_openai, max_wall_seconds=60)
            orig = agent._prepare_model_context

            async def prepare():
                await orig()
                agent._turn_deadline = time.monotonic() - 1
            agent._prepare_model_context = prepare
            seen = await self.drive(agent, ["never"])
            self.assertEqual(seen, [])
            self.assertEqual(agent.last_finish_reason, "budget_exceeded")
        await self.both(check)

    async def test_clock_is_per_chat_and_cleared_after(self):
        async def check(use_openai):
            agent = self.agent(use_openai, max_wall_seconds=60)
            deadlines = []
            await self.drive(agent, ["one"], on_call=lambda n: deadlines.append(agent._turn_deadline))
            self.assertIsNotNone(deadlines[0])
            self.assertIsNone(agent._turn_deadline)  # stopped when chat() returned
            self.assertFalse(agent._check_budget()["exceeded"])  # background side queries after the turn are not blocked
            await self.drive(agent, ["two"], on_call=lambda n: deadlines.append(agent._turn_deadline))
            self.assertGreater(deadlines[1], deadlines[0])  # restarted
            self.assertEqual(agent.last_finish_reason, "end_turn" if not use_openai else "stop")
        await self.both(check)

    def test_validation(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                make(max_wall_seconds=bad)
        self.assertIsNone(make()._turn_deadline)

    def test_message_order_keeps_cost_first(self):
        agent = make(max_cost_usd=0, max_wall_seconds=60)
        agent._turn_deadline = time.monotonic() - 1
        self.assertIn("Cost limit reached", agent._check_budget()["reason"])


class ChildCapsTests(unittest.TestCase):
    def child_of(self, parent):
        return parent._create_child_agent(system_prompt="t", tools=[])

    def test_unlimited_parent_gives_safety_net(self):
        child = self.child_of(make())
        self.assertEqual((child.max_turns, child.max_wall_seconds), (SAFETY_NET_MAX_TURNS, SAFETY_NET_MAX_WALL_SECONDS))

    def test_child_takes_min_of_parent_remaining_and_default(self):
        parent = make(max_turns=500)
        parent.current_turns = 10
        parent._turn_deadline = time.monotonic() + 30
        child = self.child_of(parent)
        self.assertEqual(child.max_turns, SAFETY_NET_MAX_TURNS)  # min(490, 100)
        self.assertLessEqual(child.max_wall_seconds, 30)  # never more than the parent has left
        parent2 = make(max_turns=5)
        parent2.current_turns = 2
        parent2._turn_deadline = time.monotonic() + 10_000
        child2 = self.child_of(parent2)
        self.assertEqual(child2.max_turns, 3)
        self.assertEqual(child2.max_wall_seconds, SAFETY_NET_MAX_WALL_SECONDS)

    def test_expired_parent_stops_child_through_the_chain(self):
        parent = make()
        parent._turn_deadline = time.monotonic() + 60
        child = self.child_of(parent)
        parent._register_child(child)
        parent._turn_deadline = time.monotonic() - 1
        self.assertIn("Wall-clock", child._check_budget()["reason"])


class BudgetErrorTests(LoopBase):
    async def test_budget_error_is_a_runtime_error_with_same_message(self):
        agent = make(max_cost_usd=0)
        with self.assertRaisesRegex(RuntimeError, "Model request blocked: Cost limit reached"):
            agent._require_model_budget()
        with self.assertRaises(BudgetExceededError):
            agent._require_model_budget()

    async def test_overflow_recovery_budget_exit_is_graceful_and_next_turn_works(self):
        async def check(use_openai):
            agent = self.agent(use_openai, max_wall_seconds=60)
            for i in range(3):
                self.msgs(agent).extend([{"role": "user", "content": f"q{i}"},
                                         {"role": "assistant", "content": f"a{i}" if use_openai else [{"type": "text", "text": f"a{i}"}]}])

            async def fold(*args, **kwargs):
                agent._turn_deadline = time.monotonic() - 1  # the clock runs out during the compaction side query
                return MEMORY
            agent._generate_folded_session_memory = fold
            err = overflow() if use_openai else __import__("anthropic").BadRequestError(
                "prompt is too long", response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                body={"type": "error", "error": {"type": "invalid_request_error"}})
            seen = await self.drive(agent, [err, "never"])
            self.assertEqual(len(seen), 2)  # the retry itself is blocked before any request
            self.assertEqual(agent.last_finish_reason, "budget_exceeded")
            self.assertIn("Budget exceeded: Wall-clock limit reached", self.out.getvalue())
            msgs = self.msgs(agent)
            self.assertEqual(msgs[-1]["role"], "user")  # ends with the current request: valid
            self.pairs_ok(agent)
            seen2 = await self.drive(agent, ["fine"], request="next")
            self.assertEqual(len(seen2), 1)
            self.assertEqual(self.msgs(agent)[-1]["role"], "assistant")
        await self.both(check)

    async def test_guard_continuation_respects_wall_clock(self):
        async def check(use_openai):
            agent = self.agent(use_openai, max_wall_seconds=60)
            write = ("edit_file", {"file_path": "a.py", "old_string": "x", "new_string": "y"}, "Successfully edited a.py\n\ndiff")

            def expire(n):
                if n == 2:
                    agent._turn_deadline = time.monotonic() - 1
            seen = await self.drive(agent, [[write], "done", "never"], on_call=expire)
            self.assertEqual(len(seen), 2)  # guard fired after call 2, the continuation is stopped at the loop top
            self.assertEqual(agent.last_finish_reason, "budget_exceeded")
        await self.both(check)


class ShellCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_requested_timeout_above_cap_is_clamped_with_note(self):
        with patch.dict(os.environ, {"BEAR_MAX_SHELL_TIMEOUT_MS": "300"}):
            result = await _run_shell({"command": "sleep 2", "timeout": 60000})
        self.assertTrue(result.startswith("Command timed out after 300ms"))
        self.assertIn("requested 60000ms exceeds the 300ms limit", result)
        self.assertTrue(is_tool_failure("run_shell", result))
        with patch.dict(os.environ, {"BEAR_MAX_SHELL_TIMEOUT_MS": "300"}):
            ok = await _run_shell({"command": "echo hi", "timeout": 60000})
            failed = await _run_shell({"command": "exit 3", "timeout": 60000})
        self.assertTrue(ok.startswith("hi\n"))
        self.assertIn("ran with 300ms", ok)
        self.assertTrue(failed.startswith("Command failed (exit code 3)"))

    async def test_within_cap_unchanged_and_default_cap(self):
        self.assertEqual(await _run_shell({"command": "echo hi", "timeout": 5000}), "hi\n")
        self.assertEqual(await _run_shell({"command": "sleep 2", "timeout": 100}), "Command timed out after 100ms")
        self.assertEqual(await _run_shell({"command": "echo hi", "timeout": 0}), "Error: timeout must be positive")
        from agents.tools import _max_shell_timeout_ms
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BEAR_MAX_SHELL_TIMEOUT_MS", None)
            self.assertEqual(_max_shell_timeout_ms(), 600_000)
        with patch.dict(os.environ, {"BEAR_MAX_SHELL_TIMEOUT_MS": "nonsense"}):
            self.assertEqual(_max_shell_timeout_ms(), 600_000)

    def test_schema_mentions_cap(self):
        from agents.tool_schemas import tool_definitions
        shell = next(t for t in tool_definitions if t["name"] == "run_shell")
        self.assertIn("max: 600000", shell["input_schema"]["properties"]["timeout"]["description"])


class EntryPointDefaultsTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(max_turns=None, max_time=None, max_cost=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_one_shot_gets_safety_net_and_repl_does_not(self):
        from agents.main import resolve_run_caps
        self.assertEqual(resolve_run_caps(self.args(), one_shot=True),
                         {"max_turns": SAFETY_NET_MAX_TURNS, "max_wall_seconds": SAFETY_NET_MAX_WALL_SECONDS, "max_cost_usd": None})
        self.assertEqual(resolve_run_caps(self.args(), one_shot=False),
                         {"max_turns": None, "max_wall_seconds": None, "max_cost_usd": None})
        self.assertEqual(resolve_run_caps(self.args(max_turns=7, max_time=30.0, max_cost=0.5), one_shot=True),
                         {"max_turns": 7, "max_wall_seconds": 30.0, "max_cost_usd": 0.5})

    def test_help_says_safety_net_not_budget(self):
        import inspect
        from agents import main
        src = inspect.getsource(main)
        self.assertIn("safety-net default, not a budget; override with --max-turns / --max-time", src)
        self.assertIn('"--max-time"', src)

if __name__ == "__main__":
    unittest.main()
