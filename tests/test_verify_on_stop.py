"""Audit item #3: verify-on-stop guard."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.agent import OVERFLOW_MARKER_TEXT, Agent
from agents.verify_on_stop import (
    CONTEXT_OVERFLOW_MARKER,
    VERIFY_COMMAND_KEYWORDS,
    VERIFY_ON_STOP_MARKER,
    is_harness_injected_message,
    is_successful_write,
    is_verification,
    load_verify_command_keywords,
    verify_on_stop_prompt,
)

KW = VERIFY_COMMAND_KEYWORDS
EDIT_OK = "Successfully edited src/app.py\n\n- x\n+ raise ValueError('error')"


class PureFunctionTests(unittest.TestCase):
    def test_successful_write_uses_prefix_not_error_heuristic(self):
        self.assertTrue(is_successful_write("edit_file", EDIT_OK))  # diff mentions "error": still a write
        self.assertTrue(is_successful_write("write_file", "Successfully wrote a.py (3 lines)"))
        self.assertFalse(is_successful_write("edit_file", "Error: old_string not found in a.py"))
        self.assertFalse(is_successful_write("write_file", "Action denied: Blocked in plan mode: write_file"))
        self.assertFalse(is_successful_write("write_file", "User denied this action."))
        self.assertFalse(is_successful_write("run_shell", "Successfully done"))
        self.assertFalse(is_successful_write("mcp__fs__write", "Successfully wrote"))

    def test_verification_matches_keywords_case_insensitively_and_ignores_exit_code(self):
        self.assertTrue(is_verification("run_shell", {"command": "python -m PyTest -q"}, "1 passed", KW))
        self.assertTrue(is_verification("run_shell", {"command": "cargo test"}, "Command failed (exit code 101)", KW))
        self.assertTrue(is_verification("run_shell", {"command": "go test ./..."}, "ok", KW))
        self.assertFalse(is_verification("run_shell", {"command": "ls -la"}, "x", KW))
        self.assertFalse(is_verification("read_file", {"command": "pytest"}, "x", KW))
        self.assertFalse(is_verification("run_shell", None, "x", KW))

    def test_denied_and_timed_out_runs_are_not_verification(self):
        for result in ("Action denied: Shell commands blocked in plan mode", "User denied this action.",
                       "Command timed out after 30000ms"):
            self.assertFalse(is_verification("run_shell", {"command": "pytest"}, result, KW), result)

    def test_table_covers_this_repos_own_checks(self):
        # Data point only: the table is per-ecosystem, not derived from this repo.
        for command in ("python -m unittest discover -s tests -q", "git diff --check", "node --check web/app.js",
                        "node --test web/tests/*.test.mjs", "pytest -q"):
            self.assertTrue(is_verification("run_shell", {"command": command}, "ok", KW), command)

    def test_settings_extra_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            user, project = Path(tmp) / "user.json", Path(tmp) / "project.json"
            user.write_text(json.dumps({"verifyOnStop": {"extraCommands": ["bazel test", "", "  ", 3]}}))
            project.write_text(json.dumps({"verifyOnStop": {"extraCommands": ["ls"]}}))
            with patch("agents.verify_on_stop._settings_files", return_value=[(user, True), (project, False)]):
                keywords = load_verify_command_keywords()
            self.assertEqual(keywords[len(KW):], ("bazel test",))  # empty entries dropped, untrusted project ignored
            with patch("agents.verify_on_stop._settings_files", return_value=[(user, True), (project, True)]):
                self.assertEqual(load_verify_command_keywords()[len(KW):], ("bazel test", "ls"))
            with patch("agents.verify_on_stop._settings_files", return_value=[(Path(tmp) / "missing.json", True)]):
                self.assertEqual(load_verify_command_keywords(), KW)

    def test_harness_injected_messages(self):
        self.assertTrue(OVERFLOW_MARKER_TEXT.startswith(CONTEXT_OVERFLOW_MARKER))
        self.assertTrue(is_harness_injected_message({"role": "user", "content": verify_on_stop_prompt(["a.py"])}))
        self.assertTrue(is_harness_injected_message({"role": "user", "content": OVERFLOW_MARKER_TEXT.format(n=3)}))
        self.assertFalse(is_harness_injected_message({"role": "user", "content": "fix the bug"}))
        self.assertFalse(is_harness_injected_message({"role": "assistant", "content": VERIFY_ON_STOP_MARKER}))

    def test_prompt_names_no_command_and_caps_file_list(self):
        prompt = verify_on_stop_prompt([f"f{i}.py" for i in range(7)])
        self.assertTrue(prompt.startswith(VERIFY_ON_STOP_MARKER))
        self.assertIn("f4.py, +2 more", prompt)
        self.assertNotIn("f5.py", prompt)
        for keyword in ("pytest", "npm test", "go test", "cargo"):
            self.assertNotIn(keyword, prompt)


# ---- loop tests: scripted model turns, scripted tool results ----

def W(path="src/app.py", result=None):
    return ("edit_file", {"file_path": path, "old_string": "a", "new_string": "b"}, result or f"Successfully edited {path}\n\ndiff")


def T(command="pytest -q", result="3 passed"):
    return ("run_shell", {"command": command}, result)


def R(path="src/app.py"):
    return ("read_file", {"file_path": path}, "1\tcontent")


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target in (patch("agents.verify_on_stop._settings_files", return_value=[]),
                       patch("agents.agent.save_dropped_messages")):
            target.start()
            self.addCleanup(target.stop)
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def agent(self, use_openai: bool, **kwargs) -> Agent:
        options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                       enable_online_evolution=False, enable_folding=False)
        options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
        options.update(kwargs)
        agent = Agent(**options)
        agent._auto_save = lambda strict=False: None
        return agent

    async def drive(self, agent: Agent, steps: list, *, stop_reason: str | None = None, sub_agent_buffer: bool = False):
        """steps: each item is a list of tool tuples (one model response) or a str (final text)."""
        steps = list(steps)
        results: dict[str, str] = {}
        seen: list[list[dict]] = []
        counter = iter(range(10_000))

        async def run_checked(name, inp):
            return results[json.dumps([name, inp], sort_keys=True)]

        async def stream():
            seen.append(json.loads(json.dumps(agent._openai_messages if agent.use_openai else agent._anthropic_messages)))
            step = steps.pop(0) if steps else "done"
            if isinstance(step, str):
                agent._emit_text(step)
                if agent.use_openai:
                    return {"choices": [{"finish_reason": stop_reason or "stop", "message": {"role": "assistant", "content": step}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                return SimpleNamespace(stop_reason=stop_reason or "end_turn", content=[SimpleNamespace(type="text", text=step)],
                                       usage=SimpleNamespace(input_tokens=10, output_tokens=1))
            calls = []
            for name, inp, result in step:
                results[json.dumps([name, inp], sort_keys=True)] = result
                calls.append((f"c{next(counter)}", name, inp))
            if agent.use_openai:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(inp)}} for cid, name, inp in calls]}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=10, output_tokens=1),
                                   content=[SimpleNamespace(type="tool_use", id=cid, name=name, input=inp) for cid, name, inp in calls])

        agent._run_checked_tool = run_checked
        if agent.use_openai:
            agent._call_openai_stream = stream
            await agent._chat_openai("please fix the bug")
        else:
            agent._call_anthropic_stream = stream
            await agent._chat_anthropic("please fix the bug")
        return seen

    def guards(self, agent: Agent) -> list[dict]:
        msgs = agent._openai_messages if agent.use_openai else agent._anthropic_messages
        return [m for m in msgs if isinstance(m.get("content"), str) and m["content"].startswith(VERIFY_ON_STOP_MARKER)]

    async def both(self, check):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                await check(use_openai)

    async def test_write_without_verification_triggers_exactly_once(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            seen = await self.drive(agent, [[R(), W()], "All done.", [W("src/other.py")], "Done again."])
            guards = self.guards(agent)
            self.assertEqual(len(guards), 1)  # second stop after another write is not checked again
            self.assertIn("src/app.py", guards[0]["content"])
            self.assertEqual(len(seen), 4)
            self.assertIn(VERIFY_ON_STOP_MARKER, json.dumps(seen[2][-1]))  # the request after the guard carries it
            self.assertIn("verify-on-stop", self.out.getvalue())
        await self.both(check)

    async def test_write_then_verify_does_not_trigger(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            seen = await self.drive(agent, [[W()], [T(result="Command failed (exit code 1)\nStdout: 1 failed")], "Done."])
            self.assertEqual(self.guards(agent), [])  # a failing test run still counts as verification
            self.assertEqual(len(seen), 3)
        await self.both(check)

    async def test_verify_then_write_triggers(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[T()], [W()], "Done."])
            self.assertEqual(len(self.guards(agent)), 1)
        await self.both(check)

    async def test_read_only_turn_does_not_trigger(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[R()], [T("ls")], "Done."])
            self.assertEqual(self.guards(agent), [])
        await self.both(check)

    async def test_failed_write_is_not_a_write_and_denied_verification_is_not_verification(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[W(result="Error: old_string not found in src/app.py")], "Done."])
            self.assertEqual(self.guards(agent), [])
            agent = self.agent(use_openai)
            await self.drive(agent, [[W()], [T(result="User denied this action.")], "Done."])
            self.assertEqual(len(self.guards(agent)), 1)
        await self.both(check)

    async def test_plan_mode_disables_guard_including_writes_made_before_exit(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            agent.permission_mode = "plan"
            await self.drive(agent, [[W(".bear/plans/plan.md")], "Plan written."])
            self.assertEqual(self.guards(agent), [])
            agent = self.agent(use_openai)
            agent.permission_mode = "plan"
            steps = [[W(".bear/plans/plan.md")], "Plan ready."]
            orig = agent._record_verify_ledger

            def record(name, inp, result):  # leave plan mode right after the plan-file write, as exit_plan_mode would
                orig(name, inp, result)
                agent.permission_mode = "default"
            agent._record_verify_ledger = record
            await self.drive(agent, steps)
            self.assertEqual(agent.permission_mode, "default")
            self.assertEqual(self.guards(agent), [])
        await self.both(check)

    async def test_ledger_resets_each_turn(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[W()], [T()], "Done."])
            await self.drive(agent, ["Just answering a question."])
            self.assertEqual(self.guards(agent), [])
            await self.drive(agent, [[W()], "Done."])
            self.assertEqual(len(self.guards(agent)), 1)
        await self.both(check)

    async def test_truncated_final_text_does_not_trigger(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            await self.drive(agent, [[W()], "partial ans"], stop_reason="length" if use_openai else "max_tokens")
            self.assertEqual(self.guards(agent), [])
        await self.both(check)

    async def test_budget_exhausted_after_guard_exits_normally(self):
        async def check(use_openai):
            # Guard continuations do not count toward max_turns, so cost is what stops them: each call costs ~$11
            # here, so the cap is crossed by the call that triggers the guard and the next request is refused.
            agent = self.agent(use_openai, max_cost_usd=15)
            agent.input_price_per_million = agent.output_price_per_million = 1_000_000
            seen = await self.drive(agent, [[W()], "Done.", "never reached"])
            self.assertEqual(len(self.guards(agent)), 1)
            self.assertEqual(len(seen), 2)  # the post-guard request is blocked by the turn budget, no loop
            self.assertEqual(agent.last_finish_reason, "budget_exceeded")
        await self.both(check)

    async def test_sub_agent_guard_goes_to_output_not_terminal(self):
        async def check(use_openai):
            agent = self.agent(use_openai, is_sub_agent=True)
            agent._output_buffer = []
            await self.drive(agent, [[W()], "Child done.", "Ran nothing; no tests here."])
            self.assertEqual(len(self.guards(agent)), 1)
            self.assertNotIn("verify-on-stop", self.out.getvalue())
            self.assertIn("no tests here", "".join(agent._output_buffer))
        await self.both(check)


class DropInteractionTests(unittest.TestCase):
    """#2 fallback must keep the real request when harness-injected user messages follow it."""

    def test_guard_message_is_not_taken_as_current_request(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = Agent(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                              enable_online_evolution=False, enable_folding=False,
                              **({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"}))
                msgs = agent._openai_messages if use_openai else agent._anthropic_messages
                for t in range(3):
                    msgs.append({"role": "user", "content": f"old {t} " + "x" * 1000})
                    msgs.append({"role": "assistant", "content": f"ok {t}" if use_openai else [{"type": "text", "text": f"ok {t}"}]})
                real = {"role": "user", "content": "REAL REQUEST " + "y" * 10000}  # big: dropped if mistaken for history
                msgs.append(real)
                msgs.append({"role": "assistant", "content": "done" if use_openai else [{"type": "text", "text": "done"}]})
                msgs.append({"role": "user", "content": verify_on_stop_prompt(["a.py"])})
                with patch("agents.agent.save_dropped_messages"):
                    self.assertTrue(agent._drop_earliest_messages(1))
                self.assertIn(real, msgs)
                self.assertTrue(is_harness_injected_message(msgs[-1]))


if __name__ == "__main__":
    unittest.main()
