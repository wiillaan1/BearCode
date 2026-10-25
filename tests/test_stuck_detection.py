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


class FormerStateTests(unittest.TestCase):
    """Pinned before #5 (commit b0366ed) as CurrentStateTests; each assertion changed as the plan intended:
    1. a fold reset the counters            → a fold no longer resets the stuck streak
    2. compact_context was counted twice     → counted once (only by the loop)
    3. "error" in a successful read = failed → prefix rule: not a failure
    """

    def test_fold_no_longer_resets_streak(self):
        agent = make()
        agent._stuck_streak, agent._stuck_signature = 3, "sig"
        agent._record_fold_event()
        self.assertEqual((agent._stuck_streak, agent._stuck_signature), (3, "sig"))

    def test_compact_context_no_longer_records_inside_the_tool(self):
        agent = make()
        for i in range(3):
            agent._openai_messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        agent._generate_folded_session_memory = AsyncMock(return_value=None)
        asyncio.run(agent._execute_compact_context_tool({}))
        self.assertEqual(agent._stuck_streak, 0)  # the tool itself records nothing; the loop records once

    def test_error_word_in_successful_read_is_not_failure(self):
        from agents.tool_outcome import is_tool_failure
        self.assertFalse(is_tool_failure("read_file", "   1 | raise ValueError('error')"))




# ---------------------------------------------------------------- new behaviour (#5)

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from agents.tool_outcome import STUCK_WARNING_MARKER, call_signature, is_tool_failure, result_digest

MISSING = ("read_file", {"file_path": "missing.txt"}, "Error reading file: [Errno 2] No such file or directory: 'missing.txt'")


class OutcomePureTests(unittest.TestCase):
    def test_signature(self):
        self.assertEqual(call_signature("edit_file", {"a": 1, "b": 2}), call_signature("edit_file", {"b": 2, "a": 1}))
        self.assertNotEqual(call_signature("edit_file", {"a": 1}), call_signature("write_file", {"a": 1}))
        self.assertNotEqual(call_signature("run_shell", {"command": "echo a  b"}), call_signature("run_shell", {"command": "echo a b"}))
        raw = call_signature("read_file", None, raw_arguments='{"file_path": ')
        self.assertEqual(raw, call_signature("read_file", None, raw_arguments='{"file_path": '))
        self.assertNotEqual(raw, call_signature("read_file", None, raw_arguments='{"file_pa'))
        self.assertEqual(call_signature("read_file", {}), call_signature("read_file", {}))  # Anthropic parse fallback

    def test_failure_prefixes(self):
        failures = [
            ("read_file", "Tool not available in this run: x"), ("read_file", "Error reading file: nope"),
            ("write_file", "Error: invalid arguments for write_file: x"), ("run_shell", "Action denied: no"),
            ("run_shell", "User denied this action."), ("run_shell", "Command failed (exit code 1)\nStdout: x"),
            ("run_shell", "Command timed out after 30000ms"), ("edit_file", "Error: old_string not found in a"),
            ("edit_file", "Warning: a.py was modified externally since your last read."),
            ("compact_context", "Compaction skipped: insufficient history"),
            ("compact_context", "Context folding limit reached (3). Continue without compacting."),
            ("compact_context", "Context folding is disabled for this run."),
            ("agent", "Error executing tool: boom"),
        ]
        for name, text in failures:
            self.assertTrue(is_tool_failure(name, text), (name, text))
        successes = [
            ("read_file", "   1 | raise ValueError('error')"), ("edit_file", "Successfully edited a.py\n- error\n+ ok"),
            ("run_shell", "3 passed; 0 errors"), ("list_files", "No files found matching the pattern."),
            ("compact_context", "No compaction needed: the whole conversation already fits"),
            ("read_file", "Warning: this file looks generated"),  # Warning: only means failure for write/edit
        ]
        for name, text in successes:
            self.assertFalse(is_tool_failure(name, text), (name, text))

    def test_digest(self):
        self.assertEqual(result_digest("abc"), result_digest("abc"))
        self.assertNotEqual(result_digest("abc"), result_digest("abd"))


class LoopBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        for target in (patch("agents.agent.save_folded_session_memory"), patch("agents.agent.save_dropped_messages"),
                       patch("agents.verify_on_stop._settings_files", return_value=[]),
                       patch("pathlib.Path.home", return_value=Path(self.home.name))):
            target.start()
            self.addCleanup(target.stop)

    def agent(self, use_openai: bool, **kwargs) -> Agent:
        agent = make(use_openai, **kwargs)
        agent._auto_save = lambda strict=False: None
        return agent

    async def drive(self, agent: Agent, steps: list, request: str = "do it"):
        """steps: list of tool tuples (one response) | str final text | async callable run before the next call."""
        steps = list(steps)
        queue: list[str] = []
        seen = []
        ids = iter(range(10_000))

        async def run_checked(name, inp):
            return queue.pop(0)

        async def stream():
            while steps and callable(steps[0]):
                await steps.pop(0)()
            seen.append(json.loads(json.dumps(agent._openai_messages if agent.use_openai else agent._anthropic_messages)))
            step = steps.pop(0) if steps else "done"
            if isinstance(step, str):
                agent._emit_text(step)
                if agent.use_openai:
                    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": step}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=step)],
                                       usage=SimpleNamespace(input_tokens=10, output_tokens=1))
            calls = [(f"c{next(ids)}", n, i) for n, i, r in step]
            queue.extend(r for _, _, r in step)
            if agent.use_openai:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": c, "type": "function", "function": {"name": n, "arguments": json.dumps(i)}} for c, n, i in calls]}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=10, output_tokens=1),
                                   content=[SimpleNamespace(type="tool_use", id=c, name=n, input=i) for c, n, i in calls])

        agent._run_checked_tool = run_checked
        if agent.use_openai:
            agent._call_openai_stream = stream
            await agent._chat_openai(request)
        else:
            agent._call_anthropic_stream = stream
            await agent._chat_anthropic(request)
        return seen

    @staticmethod
    def results(agent: Agent) -> list[str]:
        msgs = agent._openai_messages if agent.use_openai else agent._anthropic_messages
        if agent.use_openai:
            return [m["content"] for m in msgs if m.get("role") == "tool"]
        return [b["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "tool_result"]

    @staticmethod
    def pairs_ok(test, agent):
        msgs = agent._openai_messages if agent.use_openai else agent._anthropic_messages
        if agent.use_openai:
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    got = {x.get("tool_call_id") for x in msgs[i + 1:i + 1 + len(m["tool_calls"])]}
                    test.assertEqual(got, {c["id"] for c in m["tool_calls"]})
        else:
            test.assertEqual(agent._normalize_anthropic_messages(msgs), msgs)

    async def both(self, check):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                await check(use_openai)


class StuckLoopTests(LoopBase):
    async def test_three_same_failures_warn_in_tool_result(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[MISSING], [MISSING], [MISSING], "I give up."])
            results = self.results(agent)
            self.assertNotIn(STUCK_WARNING_MARKER, results[1])
            self.assertTrue(results[2].startswith(MISSING[2]))
            self.assertIn("\n\n" + STUCK_WARNING_MARKER, results[2])
            self.assertIn("stuck", self.out.getvalue().lower())
        await self.both(check)

    async def test_fold_does_not_reset(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            msgs = agent._openai_messages if use_openai else agent._anthropic_messages
            for i in range(3):
                msgs += [{"role": "user", "content": f"q{i}"},
                         {"role": "assistant", "content": f"a{i}" if use_openai else [{"type": "text", "text": f"a{i}"}]}]
            agent._generate_folded_session_memory = AsyncMock(return_value={
                "episode_memory": {"task_description": "t", "key_events": [], "current_progress": "p"},
                "working_memory": {"immediate_goal": "g", "current_challenges": "", "next_actions": []},
                "tool_memory": {"tools_used": [], "derived_rules": []}})
            fold = lambda: agent._compact_conversation(trigger="auto")
            await self.drive(agent, [[MISSING], [MISSING], fold, [MISSING], "stop"])
            self.assertEqual(agent._fold_count, 1)
            self.assertIn(STUCK_WARNING_MARKER, self.results(agent)[-1])
        await self.both(check)

    async def test_same_call_success_with_changing_result_is_progress(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            poll = [[("run_shell", {"command": "cat status"}, f"step {i}")] for i in range(5)]
            await self.drive(agent, poll + ["ok"])
            self.assertFalse(any(STUCK_WARNING_MARKER in r for r in self.results(agent)))
            agent = self.agent(use_openai)
            same = [[("run_shell", {"command": "cat status"}, "pending")] for _ in range(3)]
            await self.drive(agent, same + ["ok"])
            self.assertIn(STUCK_WARNING_MARKER, self.results(agent)[2])
        await self.both(check)

    async def test_different_call_resets(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            other = ("list_files", {"pattern": "*"}, "a.py")
            await self.drive(agent, [[MISSING], [MISSING], [other], [MISSING], [MISSING], "stop"])
            self.assertFalse(any(STUCK_WARNING_MARKER in r for r in self.results(agent)))
        await self.both(check)

    async def test_six_unattended_stops_with_history_message(self):
        async def check(use_openai):
            agent = self.agent(use_openai)  # no confirm_fn → unattended
            seen = await self.drive(agent, [[MISSING]] * 8)
            self.assertEqual(len(seen), 6)  # no 7th request
            self.assertEqual(agent.last_finish_reason, "stuck")
            msgs = agent._openai_messages if use_openai else agent._anthropic_messages
            last = msgs[-1]
            self.assertEqual(last["role"], "assistant")
            text = last["content"] if use_openai else last["content"][0]["text"]
            self.assertIn("read_file", text)
            self.assertIn("6", text)
            self.pairs_ok(self, agent)
            # next user turn: valid history, counter reset by the new request
            seen2 = await self.drive(agent, [[MISSING], "ok"], request="try something else")
            self.assertEqual(len(seen2), 2)
            self.assertNotIn(STUCK_WARNING_MARKER, self.results(agent)[-1])
            self.assertEqual(agent._stuck_streak, 1)
            self.pairs_ok(self, agent)
        await self.both(check)

    async def test_dont_ask_mode_is_unattended(self):
        agent = self.agent(True, permission_mode="dontAsk", confirm_fn=AsyncMock(return_value=True))
        await self.drive(agent, [[MISSING]] * 8)
        agent.confirm_fn.assert_not_awaited()
        self.assertEqual(agent.last_finish_reason, "stuck")

    async def test_attended_asks_user_yes_resets_then_no_stops(self):
        async def check(use_openai):
            confirm = AsyncMock(side_effect=[True, False])
            agent = self.agent(use_openai, confirm_fn=confirm)
            seen = await self.drive(agent, [[MISSING]] * 14)
            self.assertEqual(confirm.await_count, 2)
            self.assertEqual(len(seen), 12)
            self.assertEqual(agent.last_finish_reason, "stuck")
            results = self.results(agent)
            self.assertNotIn(STUCK_WARNING_MARKER, results[6])  # streak restarted after "yes"
            self.assertIn(STUCK_WARNING_MARKER, results[8])
            self.pairs_ok(self, agent)
        await self.both(check)

    async def test_sub_agent_stops_and_reports_without_asking(self):
        async def check(use_openai):
            confirm = AsyncMock(return_value=True)
            agent = self.agent(use_openai, is_sub_agent=True, confirm_fn=confirm)
            agent._output_buffer = []
            await self.drive(agent, [[MISSING]] * 8)
            confirm.assert_not_awaited()
            self.assertEqual(agent.last_finish_reason, "stuck")
            self.assertIn("read_file", "".join(agent._output_buffer))
        await self.both(check)

    async def test_stop_waits_for_the_whole_round(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            other = ("list_files", {"pattern": "*"}, "a.py")
            steps = [[MISSING]] * 5 + [[MISSING, other]]  # 6th failure shares a round with another call
            await self.drive(agent, steps + ["never"])
            self.assertEqual(agent.last_finish_reason, "stuck")
            self.pairs_ok(self, agent)
            self.assertIn("a.py", self.results(agent)[-1])
        await self.both(check)

    async def test_compact_context_counted_once(self):
        # compact_context goes through the real executor via the checkpoint layer
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = self.agent(use_openai)
                msgs = agent._openai_messages if use_openai else agent._anthropic_messages
                for i in range(3):
                    msgs += [{"role": "user", "content": f"q{i}"},
                             {"role": "assistant", "content": f"a{i}" if use_openai else [{"type": "text", "text": f"a{i}"}]}]
                agent._generate_folded_session_memory = AsyncMock(return_value=None)
                agent._begin_tool_round([("k1", "compact_context")])
                res = await agent._run_tracked_tool("k1", "compact_context", {"reason": "x"})
                agent._track_progress("k1", "compact_context", {"reason": "x"})
                self.assertTrue(res.startswith("Compaction skipped:"))
                self.assertEqual(agent._stuck_streak, 1)  # one call, one count

    async def test_large_failure_uses_raw_result(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            big = "Command failed (exit code 1)\nStdout: " + "boom\n" * 10_000  # > 30KB → persisted with a uuid path
            call = ("run_shell", {"command": "make"}, big)
            await self.drive(agent, [[call], [call], [call], "stop"])
            results = self.results(agent)
            self.assertTrue(results[0].startswith("[Result too large"))
            self.assertIn(STUCK_WARNING_MARKER, results[2])
        await self.both(check)


class StuckStateTests(unittest.TestCase):
    def test_clear_history_resets(self):
        agent = make()
        agent._stuck_streak = 4
        agent._stuck_signature = "x"
        agent._clear_history_keep_system()
        self.assertEqual(agent._stuck_streak, 0)
        self.assertIsNone(agent._stuck_signature)

    def test_guidance_has_no_counters_and_keeps_long_context_hint(self):
        agent = Agent(api_key="fake", api_base="https://example.test/v1", enable_mcp=False, enable_memory_prefetch=False,
                      enable_online_evolution=False, enable_folding=True)
        text = agent._build_fold_guidance_section()
        self.assertIn("Current context utilization", text)
        self.assertNotIn("error streak", text)
        self.assertNotIn("repeat count", text)
        self.assertNotIn("retried without progress", text)
        self.assertIn("compact_context", text)

    def test_thresholds_validated_and_inherited(self):
        self.assertEqual((make().stuck_warn_after, make().stuck_stop_after), (3, 6))
        for bad in ((0, 6), (4, 3)):
            with self.assertRaises(ValueError):
                make(stuck_warn_after=bad[0], stuck_stop_after=bad[1])
        import inspect
        self.assertIn("stuck_warn_after=self.stuck_warn_after", inspect.getsource(Agent))
        self.assertIn("stuck_stop_after=self.stuck_stop_after", inspect.getsource(Agent))

    def test_old_heuristic_removed(self):
        self.assertFalse(hasattr(Agent, "_looks_like_tool_failure"))
        self.assertFalse(hasattr(Agent, "_record_tool_outcome"))


class EvalStuckStatusTests(unittest.TestCase):
    def test_stuck_run_is_not_scored_as_completed(self):
        from agents.eval_folding import _run_case

        class FakeAgent:
            def __init__(self, **kwargs):
                self.session_id = "s"
                self._aborted = False
                self._fold_count = 0
                self._openai_messages = [{"role": "system", "content": "t"}]
                self._anthropic_messages = []
                self._folded_session_memories = []
                self.total_input_tokens = self.total_output_tokens = 0
                self.side_input_tokens = self.side_output_tokens = 0
                self.current_turns = 0
                self.last_finish_reason = None
                self._mcp_manager = type("Mcp", (), {"disconnect_all": AsyncMock()})()

            async def run_once(self, prompt):
                self.last_finish_reason = "stuck"
                return {"text": "Stopped: read_file repeated 6 times without progress."}

            async def drain_background_skill_tasks(self):
                return None

            async def aclose(self):
                return None

            def abort(self):
                self._aborted = True

            def _get_current_cost_usd(self):
                return 0.0

        case = {"task_id": "t/1", "prompts": ["only"], "answers": ["X"]}
        with tempfile.TemporaryDirectory() as tmp, patch("agents.eval_folding.Agent", FakeAgent):
            row = asyncio.run(_run_case(case, "fold_off", output_dir=Path(tmp), model="m", api_base="http://x/v1",
                                        api_key="k", use_openai=True, fold_threshold=0.7, max_turns=5, max_cost=1.0,
                                        timeout=10, score_mode="normalized", fold_after_turn=None))
        self.assertEqual(row["status"], "stuck")
        self.assertFalse(row["correct"])


if __name__ == "__main__":
    unittest.main()
