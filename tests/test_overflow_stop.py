"""Audit item #38: Anthropic stop_reason="model_context_window_exceeded" (a normal response, not an exception)."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx

from agents.agent import Agent
from agents.verify_on_stop import VERIFY_ON_STOP_MARKER

HIT = "model_context_window_exceeded"


def memory():
    return {"episode_memory": {"task_description": "finish task", "key_events": [], "current_progress": "ready"},
            "working_memory": {"immediate_goal": "continue", "current_challenges": "", "next_actions": []},
            "tool_memory": {"tools_used": [], "derived_rules": []}}


def text(t: str, stop: str = "end_turn"):
    return ("text", t, stop)


def tools(*calls, stop: str = "tool_use"):
    return ("tools", list(calls), stop)


class OverflowStopLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        for target in (patch("agents.agent.save_folded_session_memory"), patch("agents.agent.save_dropped_messages"),
                       patch("agents.verify_on_stop._settings_files", return_value=[])):
            target.start()
            self.addCleanup(target.stop)

    def agent(self, use_openai: bool = False, *, history: bool = True, fold=memory, **kwargs) -> Agent:
        options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                       enable_online_evolution=False, enable_folding=True)
        options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
        options.update(kwargs)
        agent = Agent(**options)
        agent._auto_save = lambda strict=False: None
        agent._turn_output_buffer = []
        agent._generate_folded_session_memory = AsyncMock(side_effect=lambda *a, **k: fold() if fold else None)
        if history:  # _compact_* needs at least 4 messages
            msgs = agent._openai_messages if use_openai else agent._anthropic_messages
            for i in range(2):
                msgs.append({"role": "user", "content": f"earlier {i}"})
                msgs.append({"role": "assistant", "content": f"ok {i}" if use_openai else [{"type": "text", "text": f"ok {i}"}]})
        return agent

    async def drive(self, agent: Agent, steps: list, request: str = "do the task"):
        steps = list(steps)
        seen: list[list[dict]] = []
        counter = iter(range(10_000))
        executed: list[str] = []

        async def run_checked(name, inp):
            executed.append(name)
            return f"Successfully edited {inp.get('file_path')}\n\ndiff" if name == "edit_file" else "ok"

        async def stream():
            seen.append(json.loads(json.dumps(agent._openai_messages if agent.use_openai else agent._anthropic_messages)))
            kind, payload, stop = steps.pop(0) if steps else ("text", "done", "end_turn" if not agent.use_openai else "stop")
            if kind == "text":
                agent._emit_text(payload)
            if agent.use_openai:
                message = {"role": "assistant", "content": payload if kind == "text" else None}
                if kind == "tools":
                    message["tool_calls"] = [{"id": f"c{next(counter)}", "type": "function",
                                              "function": {"name": n, "arguments": json.dumps(i)}} for n, i in payload]
                return {"choices": [{"finish_reason": stop, "message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            content = ([SimpleNamespace(type="text", text=payload)] if kind == "text" else
                       [SimpleNamespace(type="tool_use", id=f"c{next(counter)}", name=n, input=i) for n, i in payload])
            return SimpleNamespace(stop_reason=stop, content=content, usage=SimpleNamespace(input_tokens=10, output_tokens=1))

        agent._run_checked_tool = run_checked
        if agent.use_openai:
            agent._call_openai_stream = stream
            await agent._chat_openai(request)
        else:
            agent._call_anthropic_stream = stream
            await agent._chat_anthropic(request)
        return seen, executed

    @staticmethod
    def texts(msgs: list[dict]) -> str:
        return json.dumps(msgs)

    # 1
    async def test_hit_without_tools_compacts_and_drops_truncated_message(self):
        agent = self.agent()
        seen, _ = await self.drive(agent, [text("PARTIAL ANSWER", HIT), text("FULL ANSWER")])
        self.assertEqual(len(seen), 2)
        agent._generate_folded_session_memory.assert_awaited_once()
        self.assertEqual(agent._fold_count, 1)
        self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow_stop")
        retry_request = seen[1]
        # folded-memory block + this turn's request (kept by #41); the truncated reply never entered history
        self.assertEqual(len(retry_request), 2)
        self.assertTrue(retry_request[0]["content"].startswith("<session-folded-memory>"))
        self.assertEqual(retry_request[1], {"role": "user", "content": "do the task"})
        self.assertNotIn("PARTIAL ANSWER", self.texts(agent._anthropic_messages))
        self.assertEqual(agent._turn_output_buffer, ["FULL ANSWER"])  # buffer cut back before the retry
        self.assertEqual(agent._anthropic_messages[-1]["content"], [{"type": "text", "text": "FULL ANSWER"}])
        self.assertEqual(agent.last_finish_reason, "end_turn")
        self.assertIn("partial answer above was discarded", self.out.getvalue())

    # 2
    async def test_hit_with_tools_compacts_instead_of_skipping(self):
        agent = self.agent()
        seen, executed = await self.drive(agent, [tools(("read_file", {"file_path": "a.py"}), stop=HIT), text("done")])
        self.assertEqual(executed, [])
        self.assertEqual(len(seen), 2)
        self.assertNotIn("Tool execution skipped", self.texts(agent._anthropic_messages))
        self.assertNotIn("tool_use", self.texts(agent._anthropic_messages))
        self.assertEqual(agent._fold_count, 1)

    # 3
    async def test_second_hit_falls_back_to_original_behaviour(self):
        agent = self.agent()
        seen, _ = await self.drive(agent, [text("P1", HIT), text("P2", HIT)])
        self.assertEqual(len(seen), 2)
        self.assertEqual(agent._generate_folded_session_memory.await_count, 1)
        self.assertEqual(agent._anthropic_messages[-1]["content"], [{"type": "text", "text": "P2"}])
        self.assertEqual(agent.last_finish_reason, HIT)

        agent = self.agent()
        seen, executed = await self.drive(agent, [tools(("read_file", {"file_path": "a"}), stop=HIT),
                                                  tools(("read_file", {"file_path": "b"}), stop=HIT)])
        self.assertEqual(executed, [])
        self.assertIn("Tool execution skipped: incomplete model response.", self.texts(agent._anthropic_messages[-1]))
        self.assertEqual(agent._generate_folded_session_memory.await_count, 1)

    # 4
    async def test_folding_disabled_keeps_original_behaviour(self):
        agent = self.agent(enable_folding=False)
        seen, _ = await self.drive(agent, [text("PARTIAL", HIT)])
        self.assertEqual(len(seen), 1)
        agent._generate_folded_session_memory.assert_not_awaited()
        self.assertEqual(agent._anthropic_messages[-1]["content"], [{"type": "text", "text": "PARTIAL"}])
        self.assertEqual(agent._turn_output_buffer, ["PARTIAL"])

    # 5
    async def test_failed_compaction_uses_up_the_turn(self):
        agent = self.agent(fold=None)
        seen, _ = await self.drive(agent, [tools(("read_file", {"file_path": "a"}), stop=HIT)])
        self.assertEqual(len(seen), 1)
        self.assertIn("incomplete model response", self.texts(agent._anthropic_messages[-1]))
        self.assertEqual(agent._generate_folded_session_memory.await_count, 1)
        agent = self.agent(fold=None)
        await self.drive(agent, [tools(("read_file", {"file_path": "a"}), stop="tool_use"), text("P", HIT)])
        self.assertEqual(agent._generate_folded_session_memory.await_count, 1)

    # 6
    async def test_fold_limit_is_bypassed(self):
        agent = self.agent(max_fold_limit=1)
        agent._fold_count = 1
        seen, _ = await self.drive(agent, [text("P", HIT), text("ok")])
        self.assertEqual(len(seen), 2)
        self.assertEqual(agent._fold_count, 2)
        self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow_stop")

    # 7
    async def test_verify_ledger_survives_compaction(self):
        agent = self.agent()
        seen, _ = await self.drive(agent, [tools(("edit_file", {"file_path": "a.py", "old_string": "x", "new_string": "y"})),
                                           text("P", HIT), text("All done."), text("No tests here.")])
        self.assertEqual(agent._fold_count, 1)
        guards = [m for m in agent._anthropic_messages if isinstance(m.get("content"), str)
                  and m["content"].startswith(VERIFY_ON_STOP_MARKER)]
        self.assertEqual(len(guards), 1)
        self.assertEqual(len(seen), 4)

    # 8
    async def test_other_stop_reasons_unchanged(self):
        agent = self.agent()
        seen, _ = await self.drive(agent, [text("cut", "max_tokens")])
        self.assertEqual(len(seen), 1)
        self.assertEqual(agent._anthropic_messages[-1]["content"], [{"type": "text", "text": "cut"}])
        agent = self.agent()
        await self.drive(agent, [tools(("read_file", {"file_path": "a"}), stop="max_tokens")])
        self.assertIn("incomplete model response", self.texts(agent._anthropic_messages[-1]))
        agent = self.agent()
        await self.drive(agent, [text("fine")])
        for a in (agent,):
            a._generate_folded_session_memory.assert_not_awaited()

    # 9
    async def test_flag_resets_each_turn(self):
        agent = self.agent()
        await self.drive(agent, [text("P", HIT), text("ok")])
        msgs = agent._anthropic_messages
        msgs.extend([{"role": "user", "content": "more"}, {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                     {"role": "user", "content": "more"}, {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}])
        seen, _ = await self.drive(agent, [text("P", HIT), text("ok again")], request="next turn")
        self.assertEqual(len(seen), 2)
        self.assertEqual(agent._fold_count, 2)

    # 10
    async def test_openai_length_is_untouched(self):
        agent = self.agent(use_openai=True)
        seen, _ = await self.drive(agent, [text("cut", "length")])
        self.assertEqual(len(seen), 1)
        agent._generate_folded_session_memory.assert_not_awaited()
        self.assertEqual(agent._openai_messages[-1]["content"], "cut")

    # 12
    async def test_sub_agent_discard_is_silent(self):
        agent = self.agent(is_sub_agent=True)
        agent._output_buffer = []
        await self.drive(agent, [text("PARTIAL", HIT), text("FULL")])
        self.assertEqual(agent._output_buffer, ["FULL"])
        self.assertNotIn("discarded", self.out.getvalue())
        self.assertNotIn("model_context_window_exceeded", self.out.getvalue())


class OverflowStopRealSdkTests(unittest.IsolatedAsyncioTestCase):
    # 11: the real SDK stream parser must hand the unknown stop_reason through unchanged.
    async def test_real_sdk_stream_hit_then_retry(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            first = len(bodies) == 1
            events = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "fake",
                       "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 0}}},
                      {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                      {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta",
                                                                            "text": "HALF WRITTEN" if first else "complete answer"}},
                      {"type": "content_block_stop", "index": 0},
                      {"type": "message_delta", "delta": {"stop_reason": HIT if first else "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 2}},
                      {"type": "message_stop"}]
            body = "".join("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e)) for e in events)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        agent = Agent(api_key="fake", api_protocol="anthropic", custom_system_prompt="test", enable_mcp=False,
                      enable_memory_prefetch=False, enable_online_evolution=False, enable_folding=True)
        agent._auto_save = lambda strict=False: None
        await agent._anthropic_client.close()
        agent._anthropic_client = anthropic.AsyncAnthropic(
            api_key="fake", base_url="http://mock.test", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agent._generate_folded_session_memory = AsyncMock(return_value=memory())
        for i in range(2):
            agent._anthropic_messages += [{"role": "user", "content": f"earlier {i}"},
                                          {"role": "assistant", "content": [{"type": "text", "text": f"ok {i}"}]}]
        agent._turn_output_buffer = []
        with patch("agents.agent.save_folded_session_memory"), redirect_stdout(io.StringIO()):
            await agent._chat_anthropic("go")
        self.assertEqual(len(bodies), 2)
        self.assertNotIn("HALF WRITTEN", json.dumps(bodies[1]["messages"]))
        self.assertTrue(bodies[1]["messages"][0]["content"].startswith("<session-folded-memory>"))
        self.assertEqual(agent.last_finish_reason, "end_turn")
        # The stream path emits a leading "\n" before the first text delta; only the retry's text may remain.
        self.assertEqual("".join(agent._turn_output_buffer).strip(), "complete answer")
        self.assertNotIn("HALF WRITTEN", "".join(agent._turn_output_buffer))
        self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow_stop")


if __name__ == "__main__":
    unittest.main()
