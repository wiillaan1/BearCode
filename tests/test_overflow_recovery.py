"""Audit item #2: reactive compaction when a model call fails with a context-overflow 400."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai

from agents.agent import Agent, _is_context_overflow
from agents.session_memory import format_folded_memory


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1"))


def openai_error(code: str = "context_length_exceeded", message: str | None = None) -> openai.BadRequestError:
    message = message or ("This model's maximum context length is 128000 tokens. However, your messages resulted "
                          "in 140008 tokens. Please reduce the length of the messages.")
    body = {"message": message, "type": "invalid_request_error", "param": "messages", "code": code}
    return openai.BadRequestError(message, response=_response(400), body=body)


def anthropic_error(message: str = "prompt is too long: 210000 tokens > 200000 maximum") -> anthropic.BadRequestError:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return anthropic.BadRequestError(message, response=_response(400), body=body)


def memory():
    return {"episode_memory": {"task_description": "finish task", "key_events": [], "current_progress": "ready"},
            "working_memory": {"immediate_goal": "continue", "current_challenges": "", "next_actions": []},
            "tool_memory": {"tools_used": [], "derived_rules": []}}


def make_agent(use_openai: bool, **kwargs) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=True)
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    options.update(kwargs)
    # #4: these histories are tiny and would sit entirely in the default verbatim tail (no fold); tail-on is covered in test_fold_tail.py
    options.setdefault("fold_tail_tokens", 0)
    return Agent(**options)


def overflow_error(agent: Agent) -> Exception:
    return openai_error() if agent.use_openai else anthropic_error()


# ---- history builders: earlier turns with tool rounds, then a current turn with several tool rounds ----

def _openai_round(tag: str, size: int = 2000) -> list[dict]:
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call-{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call-{tag}", "content": f"{tag} " + "x" * size}]


def _anthropic_round(tag: str, size: int = 2000) -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"tu-{tag}", "name": "read_file", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu-{tag}", "content": f"{tag} " + "x" * size}]}]


def build_history(agent: Agent, *, past_turns: int = 3, current_rounds: int = 3) -> list[dict]:
    rnd = _openai_round if agent.use_openai else _anthropic_round
    messages = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    for t in range(past_turns):
        messages.append({"role": "user", "content": f"turn {t} request"})
        messages.extend(rnd(f"t{t}r0"))
        messages.append({"role": "assistant", "content": f"turn {t} done" if agent.use_openai else [{"type": "text", "text": f"turn {t} done"}]})
    messages.append({"role": "user", "content": "CURRENT REQUEST"})
    for r in range(current_rounds):
        messages.extend(rnd(f"cur{r}"))
    return messages


def assert_pairs_intact(test: unittest.TestCase, agent: Agent) -> None:
    if agent.use_openai:
        msgs = agent._openai_messages
        for i, msg in enumerate(msgs):
            if msg.get("role") == "tool":
                j = i - 1
                while msgs[j].get("role") == "tool":
                    j -= 1
                ids = {c["id"] for c in msgs[j].get("tool_calls") or []}
                test.assertIn(msg["tool_call_id"], ids)
    else:
        msgs = agent._anthropic_messages
        test.assertEqual(agent._normalize_anthropic_messages(msgs), msgs)
        test.assertEqual(msgs[0]["role"], "user")
        for i, msg in enumerate(msgs):
            if agent._anthropic_tool_result_ids(msg):
                test.assertEqual(agent._anthropic_tool_result_ids(msg), agent._anthropic_tool_use_ids(msgs[i - 1]))


class OverflowDetectionTests(unittest.TestCase):
    def test_openai_context_length_exceeded(self):
        self.assertTrue(_is_context_overflow(openai_error()))

    def test_openai_other_400_is_not_overflow(self):
        self.assertFalse(_is_context_overflow(openai_error(code="invalid_value", message="bad parameter")))

    def test_anthropic_prompt_is_too_long(self):
        # Keyword "prompt is too long" is from the Anthropic docs; the surrounding text of this message is illustrative.
        self.assertTrue(_is_context_overflow(anthropic_error()))
        self.assertTrue(_is_context_overflow(anthropic_error("Prompt is too long")))

    def test_anthropic_context_keyword_branch_unverified(self):
        # 未实测，待真实报文核实：this branch exists for messages without "prompt is too long" (e.g. older models'
        # input + max_tokens overflow). The strings below are made up to exercise the keyword rule, not real responses.
        self.assertTrue(_is_context_overflow(anthropic_error("input length and max_tokens exceed context limit")))
        self.assertTrue(_is_context_overflow(anthropic_error("context is too long")))
        self.assertFalse(_is_context_overflow(anthropic_error("context window settings are invalid")))

    def test_anthropic_other_invalid_request_is_not_overflow(self):
        self.assertFalse(_is_context_overflow(anthropic_error("Your workspace has reached its spend limit")))
        self.assertFalse(_is_context_overflow(anthropic_error("max_tokens: Field required")))

    def test_non_400_and_plain_errors_are_not_overflow(self):
        rate = openai.RateLimitError("rate limited", response=_response(429), body={"code": "context_length_exceeded"})
        self.assertFalse(_is_context_overflow(rate))
        self.assertFalse(_is_context_overflow(ValueError("prompt is too long")))


class CompactionFlagTests(unittest.TestCase):
    def test_ignore_fold_limit_bypasses_limit_but_still_counts(self):
        agent = make_agent(True, max_fold_limit=1)
        agent._fold_count = 1
        build_history(agent)
        agent._generate_folded_session_memory = AsyncMock(return_value=memory())
        with patch("agents.agent.save_folded_session_memory"):
            self.assertFalse(asyncio.run(agent._compact_conversation(trigger="auto")))
            self.assertTrue(asyncio.run(agent._compact_conversation(trigger="overflow", ignore_fold_limit=True)))
        self.assertEqual(agent._fold_count, 2)
        self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow")

    def test_ignore_fold_limit_still_respects_enable_folding(self):
        agent = make_agent(True, enable_folding=False)
        build_history(agent)
        agent._generate_folded_session_memory = AsyncMock(return_value=memory())
        self.assertFalse(asyncio.run(agent._compact_conversation(trigger="overflow", ignore_fold_limit=True)))
        agent._generate_folded_session_memory.assert_not_awaited()


class DropEarliestMessagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch("agents.session.get_project_session_dir", return_value=Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def dropped_rows(self, agent: Agent) -> list[dict]:
        path = Path(self.tmp.name) / f"{agent.session_id}.dropped.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def check_protocol(self, use_openai: bool):
        agent = make_agent(use_openai)
        messages = build_history(agent)
        original = [json.loads(json.dumps(m)) for m in messages]
        prefix = 1 if use_openai else 0
        self.assertTrue(agent._drop_earliest_messages(1))
        msgs = agent._openai_messages if use_openai else agent._anthropic_messages
        assert_pairs_intact(self, agent)
        if use_openai:
            self.assertEqual(msgs[0]["role"], "system")
        self.assertIn({"role": "user", "content": "CURRENT REQUEST"}, msgs)
        self.assertEqual(msgs[-2:], original[-2:])  # last tool round of the current turn is kept
        markers = [m for m in msgs if isinstance(m.get("content"), str) and m["content"].startswith("[Context overflow:")]
        self.assertEqual(len(markers), 1)
        size = lambda ms: sum(len(json.dumps(m, ensure_ascii=False).encode()) for m in ms)
        self.assertLessEqual(size([m for m in msgs[prefix:] if m not in markers]), size(original[prefix:]) * 0.5)

        rows = self.dropped_rows(agent)
        dropped = [r["message"] for r in rows]
        self.assertIn(f"{len(dropped)} earlier messages", markers[0]["content"])
        self.assertTrue(all(r["fallback_round"] == 1 and r["time"] for r in rows))
        # restorable: dropped rows + kept messages (without the marker) give back the original history in order
        kept = [m for m in msgs if m not in markers]
        canon = lambda ms: sorted(json.dumps(m, sort_keys=True) for m in ms)
        self.assertEqual(canon(kept + dropped), canon(original))
        self.assertEqual(dropped, [m for m in original if m not in kept])
        return agent

    def test_openai_drop_keeps_pairs_and_persists(self):
        self.check_protocol(True)

    def test_anthropic_drop_keeps_pairs_and_persists(self):
        self.check_protocol(False)

    def test_second_round_drops_inside_current_turn_and_accumulates_marker(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai)
                build_history(agent, past_turns=1, current_rounds=5)
                self.assertTrue(agent._drop_earliest_messages(1))
                self.assertTrue(agent._drop_earliest_messages(2))
                msgs = agent._openai_messages if use_openai else agent._anthropic_messages
                assert_pairs_intact(self, agent)
                rows = self.dropped_rows(agent)
                self.assertEqual({r["fallback_round"] for r in rows}, {1, 2})
                markers = [i for i, m in enumerate(msgs) if isinstance(m.get("content"), str) and m["content"].startswith("[Context overflow:")]
                self.assertEqual(len(markers), 1)
                self.assertIn(f"{len(rows)} earlier messages", msgs[markers[0]]["content"])
                current = msgs.index({"role": "user", "content": "CURRENT REQUEST"})
                self.assertEqual(markers[0], current + 1)  # in-turn drop: trace sits right after the request

    def test_folded_memory_block_is_never_dropped(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai)
                messages = agent._openai_messages if use_openai else agent._anthropic_messages
                block = {"role": "user", "content": format_folded_memory(memory())}
                messages.append(block)
                build_history(agent)
                self.assertTrue(agent._drop_earliest_messages(1))
                self.assertEqual(messages[1 if use_openai else 0], block)
                assert_pairs_intact(self, agent)

    def test_nothing_droppable_returns_false_and_leaves_history(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai)
                messages = agent._openai_messages if use_openai else agent._anthropic_messages
                messages.append({"role": "user", "content": "huge paste " + "x" * 5000})
                before = list(messages)
                self.assertFalse(agent._drop_earliest_messages(1))
                self.assertEqual(messages, before)
                agent = make_agent(use_openai)
                messages = build_history(agent, past_turns=0, current_rounds=1)  # current request + its only round
                before = list(messages)
                self.assertFalse(agent._drop_earliest_messages(1))
                self.assertEqual(messages, before)

    def test_persist_failure_does_not_block_drop(self):
        agent = make_agent(True)
        build_history(agent)
        with patch("agents.agent.save_dropped_messages", side_effect=OSError("disk full")), \
                self.assertLogs(level="WARNING"):
            self.assertTrue(agent._drop_earliest_messages(1))
        assert_pairs_intact(self, agent)
        self.assertLess(len(agent._openai_messages), 1 + 3 * 4 + 1 + 6)


class OverflowRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def stream(self, agent: Agent, failures: list[Exception]):
        calls = []

        async def fake():
            calls.append(len(agent._openai_messages if agent.use_openai else agent._anthropic_messages))
            if failures:
                raise failures.pop(0)
            if agent.use_openai:
                return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=1))
        return fake, calls

    async def run_chat(self, agent: Agent, failures: list[Exception]):
        fake, calls = self.stream(agent, failures)
        name = "_call_openai_stream" if agent.use_openai else "_call_anthropic_stream"
        setattr(agent, name, fake)
        chat = agent._chat_openai if agent.use_openai else agent._chat_anthropic
        with patch("agents.agent.save_folded_session_memory"), patch("agents.agent.save_dropped_messages"):
            await chat("next request")
        return calls

    async def test_compaction_then_single_retry_without_dropping(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai, max_fold_limit=1)
                agent._fold_count = 1  # limit already reached: overflow must still fold
                build_history(agent)
                agent._generate_folded_session_memory = AsyncMock(return_value=memory())
                with patch.object(agent, "_drop_earliest_messages") as drop:
                    calls = await self.run_chat(agent, [overflow_error(agent)])
                self.assertEqual(drop.call_count, 0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(agent._fold_count, 2)
                self.assertEqual(agent._folded_session_memories[-1]["trigger"], "overflow")

    async def test_folding_disabled_goes_straight_to_fallback(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai, enable_folding=False)
                build_history(agent)
                agent._generate_folded_session_memory = AsyncMock(return_value=memory())
                calls = await self.run_chat(agent, [overflow_error(agent)])
                agent._generate_folded_session_memory.assert_not_awaited()
                self.assertEqual(len(calls), 2)
                self.assertLess(calls[1], calls[0])
                assert_pairs_intact(self, agent)

    async def test_failed_compaction_falls_back_to_dropping(self):
        agent = make_agent(True)
        build_history(agent)
        agent._generate_folded_session_memory = AsyncMock(return_value=None)  # e.g. the side query overflowed too
        calls = await self.run_chat(agent, [openai_error()])
        agent._generate_folded_session_memory.assert_awaited_once()
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[1], calls[0])

    async def test_still_overflowing_after_two_fallback_rounds_raises(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai)
                build_history(agent, current_rounds=6)
                agent._generate_folded_session_memory = AsyncMock(return_value=memory())
                failures = [overflow_error(agent) for _ in range(2)]
                last = failures[-1]
                with patch.object(agent, "_drop_earliest_messages", wraps=agent._drop_earliest_messages) as drop:
                    with self.assertRaises(type(last)) as ctx:
                        await self.run_chat(agent, failures)
                self.assertIs(ctx.exception, last)
                self.assertEqual(agent._fold_count, 1)  # compacted once only, no compaction loop
                self.assertEqual(drop.call_count, 1)  # after compaction only the memory remains: nothing to drop, raise

    async def test_two_fallback_rounds_then_raise_when_folding_disabled(self):
        # Mid-turn overflow (after several tool rounds), so both fallback rounds have something to drop.
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai, enable_folding=False)
                build_history(agent, current_rounds=8)
                failures = [overflow_error(agent) for _ in range(3)]
                fake, calls = self.stream(agent, failures[1:])
                with patch.object(agent, "_drop_earliest_messages", wraps=agent._drop_earliest_messages) as drop, \
                        patch("agents.agent.save_dropped_messages"):
                    with self.assertRaises(type(failures[0])) as ctx:
                        await agent._call_after_overflow(failures[0], fake)
                self.assertIs(ctx.exception, failures[-1])
                self.assertEqual([c.args[0] for c in drop.call_args_list], [1, 2])
                self.assertEqual(len(calls), 2)
                self.assertLess(calls[1], calls[0])
                assert_pairs_intact(self, agent)

    async def test_non_overflow_400_propagates_untouched(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make_agent(use_openai)
                build_history(agent)
                before = len(agent._openai_messages if use_openai else agent._anthropic_messages)
                err = openai_error(code="invalid_value", message="bad") if use_openai else anthropic_error("spend limit reached")
                agent._generate_folded_session_memory = AsyncMock(return_value=memory())
                with patch.object(agent, "_drop_earliest_messages") as drop, self.assertRaises(type(err)):
                    await self.run_chat(agent, [err])
                agent._generate_folded_session_memory.assert_not_awaited()
                self.assertEqual(drop.call_count, 0)
                self.assertEqual(len(agent._openai_messages if use_openai else agent._anthropic_messages), before + 1)

    async def test_non_overflow_error_on_retry_propagates(self):
        agent = make_agent(True)
        build_history(agent)
        agent._generate_folded_session_memory = AsyncMock(return_value=memory())
        boom = openai.RateLimitError("rate", response=_response(429), body=None)
        with patch.object(agent, "_drop_earliest_messages") as drop, self.assertRaises(openai.RateLimitError):
            await self.run_chat(agent, [openai_error(), boom])
        self.assertEqual(drop.call_count, 0)


if __name__ == "__main__":
    unittest.main()
