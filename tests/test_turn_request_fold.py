"""Audit item #41: a fold keeps the request the current turn is working on, right after the memory block."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai

from agents.agent import Agent
from agents.verify_on_stop import VERIFY_ON_STOP_MARKER

REQUEST = "Please refactor module X and keep the public API."


def memory():
    return {"episode_memory": {"task_description": "t", "key_events": [], "current_progress": "p"},
            "working_memory": {"immediate_goal": "g", "current_challenges": "", "next_actions": []},
            "tool_memory": {"tools_used": [], "derived_rules": []}}


def _resp(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1"))


def overflow(agent):
    if agent.use_openai:
        return openai.BadRequestError("too long", response=_resp(400), body={"code": "context_length_exceeded"})
    return anthropic.BadRequestError("prompt is too long", response=_resp(400),
                                     body={"type": "error", "error": {"type": "invalid_request_error"}})


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        for target in (patch("agents.agent.save_folded_session_memory"), patch("agents.agent.save_dropped_messages"),
                       patch("agents.verify_on_stop._settings_files", return_value=[])):
            target.start()
            self.addCleanup(target.stop)

    def agent(self, use_openai: bool, **kwargs) -> Agent:
        options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                       enable_online_evolution=False, enable_folding=True, model="gpt-4o")
        options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
        options.update(kwargs)
        # #4: these histories are tiny and would sit entirely in the default verbatim tail (no fold); tail-on is covered in test_fold_tail.py
        options.setdefault("fold_tail_tokens", 0)
        agent = Agent(**options)
        agent._auto_save = lambda strict=False: None
        self.transcripts = []
        agent.fold_observer = lambda event: self.transcripts.append(event["transcript"])
        agent._generate_folded_session_memory = AsyncMock(side_effect=lambda *a, **k: memory())
        msgs = self.msgs(agent)
        for i in range(2):  # _compact_* needs at least 4 messages
            msgs.append({"role": "user", "content": f"earlier {i}"})
            msgs.append({"role": "assistant", "content": f"ok {i}" if use_openai else [{"type": "text", "text": f"ok {i}"}]})
        return agent

    @staticmethod
    def msgs(agent):
        return agent._openai_messages if agent.use_openai else agent._anthropic_messages

    @staticmethod
    def after_memory(agent):
        msgs = Base.msgs(agent)
        i = next(k for k, m in enumerate(msgs) if isinstance(m.get("content"), str) and m["content"].startswith("<session-folded-memory>"))
        return msgs[i + 1:]

    async def drive(self, agent: Agent, steps: list, request: str = REQUEST, via_chat: bool = False):
        """steps: ("tool", name, inp, result) | ("text", t) | ("raise", exc) | ("hit", t) | async callable."""
        steps = list(steps)
        seen = []
        ids = iter(range(10_000))

        async def run_checked(name, inp):
            if name == "compact_context":
                return await agent._execute_compact_context_tool(inp)
            return next(s[3] for s in self._tool_results if s[1] == name)

        self._tool_results = [s for s in steps if isinstance(s, tuple) and s[0] == "tool"]

        async def stream():
            while steps and callable(steps[0]):
                await steps.pop(0)()
            seen.append(list(self.msgs(agent)))
            step = steps.pop(0) if steps else ("text", "done")
            if step[0] == "raise":
                raise step[1]
            if step[0] in ("text", "hit"):
                stop = ("model_context_window_exceeded" if step[0] == "hit" else ("stop" if agent.use_openai else "end_turn"))
                if agent.use_openai:
                    return {"choices": [{"finish_reason": stop, "message": {"role": "assistant", "content": step[1]}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=step[1])],
                                       usage=SimpleNamespace(input_tokens=10, output_tokens=1))
            cid = f"c{next(ids)}"
            if agent.use_openai:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": step[1], "arguments": json.dumps(step[2])}}]}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=10, output_tokens=1),
                                   content=[SimpleNamespace(type="tool_use", id=cid, name=step[1], input=step[2])])

        agent._run_checked_tool = run_checked
        setattr(agent, "_call_openai_stream" if agent.use_openai else "_call_anthropic_stream", stream)
        if via_chat:
            await agent.chat(request)
        else:
            await (agent._chat_openai if agent.use_openai else agent._chat_anthropic)(request)
        return seen

    async def both(self, check):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                await check(use_openai)


class TurnRequestFoldTests(Base):
    # 1, 2
    async def test_fold_inside_turn_keeps_request_object_and_transcript_still_has_it(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            fold = lambda: agent._compact_conversation(trigger="auto")
            seen = await self.drive(agent, [("tool", "read_file", {"file_path": "a"}, "1\tx"), fold, ("text", "done")])
            retry = seen[1]
            self.assertEqual(len(retry), (2 if use_openai else 1) + 1)
            self.assertIs(retry[-1], agent._turn_request)
            self.assertEqual(retry[-1], {"role": "user", "content": REQUEST})
            self.assertIn(REQUEST, self.transcripts[0])
        await self.both(check)

    # 3
    async def test_two_folds_in_one_turn_keep_one_copy(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            fold = lambda: agent._compact_conversation(trigger="auto")
            seen = await self.drive(agent, [("tool", "read_file", {"file_path": "a"}, "1\tx"), fold,
                                            ("tool", "list_files", {"pattern": "*"}, "a"), fold, ("text", "done")])
            self.assertEqual(agent._fold_count, 2)
            self.assertEqual(sum(1 for m in seen[2] if m.get("content") == REQUEST), 1)
            self.assertEqual(seen[2][-1], {"role": "user", "content": REQUEST})
        await self.both(check)

    # 4
    async def test_between_turns_nothing_is_put_back(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            agent._turn_output_buffer = None
            await self.drive(agent, [("text", "answered")], via_chat=True)
            self.assertIsNone(agent._turn_request)
            for trigger in ("manual", "eval_scheduled"):
                agent2 = self.agent(use_openai)
                await self.drive(agent2, [("text", "answered")], via_chat=True)
                self.assertTrue(await agent2._compact_conversation(trigger=trigger))
                self.assertEqual(self.after_memory(agent2), [])
        await self.both(check)

    # 5
    async def test_compact_context_tool_keeps_request_and_never_adopts_tool_result(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            fold = lambda: agent._compact_conversation(trigger="auto")
            seen = await self.drive(agent, [("tool", "compact_context", {"reason": "long"}, None),
                                            ("tool", "read_file", {"file_path": "a"}, "1\tx"), fold, ("text", "done")])
            after_tool = seen[1]
            self.assertEqual(self.after_memory_of(after_tool)[0], {"role": "user", "content": REQUEST})
            self.assertIn("Context compacted", json.dumps(self.after_memory_of(after_tool)[1:]))
            self.assertEqual(self.after_memory_of(seen[2]), [{"role": "user", "content": REQUEST}])
        await self.both(check)

    @staticmethod
    def after_memory_of(msgs):
        i = next(k for k, m in enumerate(msgs) if isinstance(m.get("content"), str) and m["content"].startswith("<session-folded-memory>"))
        return msgs[i + 1:]

    # 6
    async def test_reactive_overflow_retry_carries_request(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            seen = await self.drive(agent, [("raise", overflow(agent)), ("text", "done")])
            self.assertEqual(self.after_memory_of(seen[1]), [{"role": "user", "content": REQUEST}])
            self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow")
        await self.both(check)

    # 7
    async def test_overflow_stop_retry_carries_request(self):
        agent = self.agent(False)
        seen = await self.drive(agent, [("hit", "half"), ("text", "done")])
        self.assertEqual(self.after_memory_of(seen[1]), [{"role": "user", "content": REQUEST}])
        self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow_stop")

    # 8
    async def test_fold_after_guard_keeps_real_request_not_guard(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            fold = lambda: agent._compact_conversation(trigger="auto")
            seen = await self.drive(agent, [("tool", "edit_file", {"file_path": "a.py", "old_string": "x", "new_string": "y"},
                                             "Successfully edited a.py\n\ndiff"), ("text", "done"), fold, ("text", "ok")])
            kept = self.after_memory_of(seen[2])
            self.assertEqual(kept, [{"role": "user", "content": REQUEST}])
            self.assertNotIn(VERIFY_ON_STOP_MARKER, json.dumps(kept))
        await self.both(check)

    # 10
    async def test_drop_after_fold_still_treats_request_as_current(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            msgs = self.msgs(agent)
            request = {"role": "user", "content": REQUEST + " " + "y" * 6000}
            msgs.append(request)
            agent._turn_request = request
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            msgs = self.msgs(agent)
            for r in range(4):
                if use_openai:
                    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"t{r}", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": f"t{r}", "content": "z" * 3000}]
                else:
                    msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{r}", "name": "read_file", "input": {}}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{r}", "content": "z" * 3000}]}]
            self.assertTrue(agent._drop_earliest_messages(1))
            self.assertTrue(any(m is request for m in self.msgs(agent)))
        await self.both(check)


class TurnRequestSizeTests(Base):
    # 9: the criterion is the same estimator that triggers threshold folding, so a fold loop cannot start.
    async def test_small_window_with_big_system_prompt_does_not_put_back(self):
        async def check(use_openai):
            agent = self.agent(use_openai, model="unknown-small-model", custom_system_prompt="S" * 30_000)
            self.assertEqual(agent.context_window, 32000)
            request = {"role": "user", "content": "R" * int(agent.effective_window * 0.20 * 3)}  # ~20% of the window
            self.msgs(agent).append(request)
            agent._turn_request = request
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            self.assertEqual(self.after_memory(agent), [])
            self.assertIn("too large to keep verbatim", self.out.getvalue())  # Info box wraps lines
            agent._generate_folded_session_memory.reset_mock()
            await agent._check_and_compact()
            agent._generate_folded_session_memory.assert_not_awaited()  # no fold loop
        await self.both(check)

    async def test_same_request_on_big_window_is_put_back(self):
        async def check(use_openai):
            agent = self.agent(use_openai, model="gpt-4o", custom_system_prompt="S" * 30_000)
            request = {"role": "user", "content": "R" * int(26_000 * 0.20 * 3)}
            self.msgs(agent).append(request)
            agent._turn_request = request
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            self.assertEqual(len(self.after_memory(agent)), 1)
            self.assertIs(self.after_memory(agent)[0], request)
        await self.both(check)

    async def test_precheck_skips_obviously_oversized_request(self):
        agent = self.agent(True)
        request = {"role": "user", "content": "R" * int(agent.effective_window * 0.30 * 3)}
        agent._openai_messages.append(request)
        agent._turn_request = request
        self.assertTrue(await agent._compact_conversation(trigger="auto"))
        self.assertEqual(self.after_memory(agent), [])


class TurnRequestIdentityTests(Base):
    """The request dict must survive every in-turn history rewrite; otherwise #41 silently stops working."""

    async def test_identity_survives_history_rewrites(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            msgs = self.msgs(agent)
            request = {"role": "user", "content": REQUEST}
            msgs.append(request)
            agent._turn_request = request
            for r in range(6):
                if use_openai:
                    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"t{r}", "type": "function",
                             "function": {"name": "read_file", "arguments": json.dumps({"file_path": "a"})}}]},
                             {"role": "tool", "tool_call_id": f"t{r}", "content": "z" * 40_000}]
                else:
                    msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{r}", "name": "read_file", "input": {"file_path": "a"}}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{r}", "content": "z" * 40_000}]}]
            present = lambda: any(m is request for m in self.msgs(agent))
            # microcompact (idle) alone first: it skips results that snip already replaced
            agent.last_input_token_count = 0
            agent.last_api_call_time = time.time() - 3600
            agent._run_compression_pipeline()
            self.assertIn("[Old result cleared]", json.dumps(self.msgs(agent)))
            self.assertTrue(present())
            # then budget (>=0.5) and snip (>=0.6)
            agent.last_input_token_count = agent.effective_window
            agent.last_api_call_time = time.time()
            agent._run_compression_pipeline()
            self.assertIn("budgeted", json.dumps(self.msgs(agent)))
            self.assertIn("[Content snipped", json.dumps(self.msgs(agent)))
            self.assertTrue(present())
            # interrupted tool call repair
            if use_openai:
                self.msgs(agent).append({"role": "assistant", "content": None, "tool_calls": [{"id": "dangling", "type": "function",
                                         "function": {"name": "read_file", "arguments": "{}"}}]})
            else:
                self.msgs(agent).append({"role": "assistant", "content": [{"type": "tool_use", "id": "dangling", "name": "read_file", "input": {}}]})
            agent._repair_interrupted_tool_calls()
            self.assertTrue(present())
            # #2 fallback
            self.assertTrue(agent._drop_earliest_messages(1))
            self.assertTrue(present())
            # context-reset branch of compact_context: fold, then the loop appends the tool result as a user message
            self.assertTrue(await agent._compact_conversation(trigger="tool"))
            agent._append_context_reset_message("Context compacted into structured session memory.")
            self.assertTrue(present())
            self.assertIs(agent._turn_request, request)
        await self.both(check)

    async def test_lost_request_is_reported_not_silent(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            agent._turn_request = {"role": "user", "content": REQUEST}  # never appended: an equal dict is not enough
            self.msgs(agent).append({"role": "user", "content": REQUEST})
            with self.assertLogs(level="WARNING") as logs:
                self.assertTrue(await agent._compact_conversation(trigger="auto"))
            self.assertIn("turn request lost before fold", "\n".join(logs.output))
            self.assertEqual(self.after_memory(agent), [])
        await self.both(check)

    async def test_plan_clear_and_execute_adopts_new_request(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            agent._turn_request = self.msgs(agent)[-2]
            agent._clear_history_keep_system()
            self.assertIsNone(agent._turn_request)  # legitimate absence: history was reset on purpose
            agent._append_context_reset_message("User approved the plan. Context was cleared.\n\n## Approved Plan:\nstep 1")
            self.assertIs(agent._turn_request, self.msgs(agent)[-1])
            for i in range(3):
                self.msgs(agent).append({"role": "assistant", "content": f"working {i}"} if use_openai else
                                        {"role": "assistant", "content": [{"type": "text", "text": f"working {i}"}]})
                self.msgs(agent).append({"role": "user", "content": f"note {i}"})
            agent._turn_request = next(m for m in self.msgs(agent) if "Approved Plan" in str(m.get("content")))
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            self.assertIn("Approved Plan", self.after_memory(agent)[0]["content"])
        await self.both(check)


if __name__ == "__main__":
    unittest.main()
