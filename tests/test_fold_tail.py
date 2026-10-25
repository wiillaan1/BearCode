"""Audit item #4: keep a verbatim tail after the folded memory."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai

from agents.agent import FOLD_MEMORY_RESERVE_TOKENS, Agent
from agents.context_runtime import estimate_tokens
from agents.session_memory import (
    FOLD_COMPONENT_SYSTEMS,
    build_anthropic_transcript,
    build_folding_user_prompt,
    build_openai_transcript,
    format_folded_memory,
    split_previous_memory,
)

REQUEST = "CURRENT REQUEST: finish the refactor"
UNIT_BYTES = 3000  # ~1000 estimated tokens per tool result


def memory(pad: int = 0):
    return {"episode_memory": {"task_description": "t" + "m" * pad, "key_events": [], "current_progress": "p"},
            "working_memory": {"immediate_goal": "g", "current_challenges": "", "next_actions": []},
            "tool_memory": {"tools_used": [], "derived_rules": []}}


def _resp(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1"))


def overflow(agent):
    if agent.use_openai:
        return openai.BadRequestError("too long", response=_resp(400), body={"code": "context_length_exceeded"})
    return anthropic.BadRequestError("prompt is too long", response=_resp(400),
                                     body={"type": "error", "error": {"type": "invalid_request_error"}})


def rnd(use_openai: bool, tag: str, size: int = UNIT_BYTES, name: str = "read_file") -> list[dict]:
    if use_openai:
        return [{"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"id-{tag}", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"id-{tag}", "content": f"{tag} " + "x" * size}]
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"id-{tag}", "name": name, "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"id-{tag}", "content": f"{tag} " + "x" * size}]}]


def reply(use_openai: bool, text: str) -> dict:
    return {"role": "assistant", "content": text if use_openai else [{"type": "text", "text": text}]}


def assert_pairs_intact(test, agent: Agent) -> None:
    msgs = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    if agent.use_openai:
        for i, msg in enumerate(msgs):
            if msg.get("role") == "tool":
                j = i - 1
                while msgs[j].get("role") == "tool":
                    j -= 1
                test.assertIn(msg["tool_call_id"], {c["id"] for c in msgs[j].get("tool_calls") or []}, "orphan tool message")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                following = {m.get("tool_call_id") for m in msgs[i + 1:i + 1 + len(msg["tool_calls"])]}
                test.assertEqual(following, {c["id"] for c in msg["tool_calls"]}, "unanswered tool_calls")
    else:
        test.assertEqual(agent._normalize_anthropic_messages(msgs), msgs)
        for i, msg in enumerate(msgs):
            if agent._anthropic_tool_use_ids(msg):
                test.assertLess(i + 1, len(msgs), "trailing tool_use")
                test.assertEqual(agent._anthropic_tool_use_ids(msg), agent._anthropic_tool_result_ids(msgs[i + 1]))


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

    def agent(self, use_openai: bool, *, mem=memory, **kwargs) -> Agent:
        options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                       enable_online_evolution=False, enable_folding=True, model="gpt-4o", fold_tail_tokens=3500)
        options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
        options.update(kwargs)
        agent = Agent(**options)
        agent._auto_save = lambda strict=False: None
        self.calls = []

        async def gen(transcript, previous_memory=None, tail_kept=False):
            self.calls.append({"transcript": transcript, "previous_memory": previous_memory, "tail_kept": tail_kept})
            return mem()
        agent._generate_folded_session_memory = gen
        return agent

    @staticmethod
    def msgs(agent):
        return agent._openai_messages if agent.use_openai else agent._anthropic_messages

    def history(self, agent, *, past_turns: int = 3, current_rounds: int = 4, request: str = REQUEST) -> dict:
        """Past turns (user, tool round, reply), then the current request and its tool rounds; returns the request."""
        o = agent.use_openai
        msgs = self.msgs(agent)
        for t in range(past_turns):
            msgs.append({"role": "user", "content": f"past turn {t}"})
            msgs.extend(rnd(o, f"p{t}"))
            msgs.append(reply(o, f"past reply {t}"))
        req = {"role": "user", "content": request}
        msgs.append(req)
        agent._turn_request = req
        for r in range(current_rounds):
            msgs.extend(rnd(o, f"c{r}"))
        return req

    @staticmethod
    def after_memory(agent):
        msgs = Base.msgs(agent)
        i = next(k for k, m in enumerate(msgs) if isinstance(m.get("content"), str) and m["content"].startswith("<session-folded-memory>"))
        return msgs[i + 1:]

    async def both(self, check):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                await check(use_openai)


class TailShapeTests(Base):
    async def test_tail_kept_after_memory_with_pairs_intact_and_request_first(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            req = self.history(agent)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            kept = self.after_memory(agent)
            self.assertIs(kept[0], req)  # request precedes the rounds that followed it
            self.assertGreater(len(kept), 1)
            self.assertIn("c3 ", json.dumps(kept))  # newest round kept
            self.assertNotIn("p0 ", json.dumps(kept))  # oldest history folded
            self.assertLessEqual(estimate_tokens(kept[1:]), 3500)
            assert_pairs_intact(self, agent)
            self.assertTrue(self.calls[0]["tail_kept"])
        await self.both(check)

    async def test_tail_not_in_transcript_but_request_is(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            self.history(agent)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            transcript = self.calls[0]["transcript"]
            self.assertIn(REQUEST, transcript)
            self.assertIn("p0 ", transcript)
            for m in self.after_memory(agent)[1:]:
                for tag in ("c2 ", "c3 "):
                    if tag in json.dumps(m):
                        self.assertNotIn(tag, transcript)
        await self.both(check)

    async def test_open_round_is_folded_not_kept(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            self.history(agent)
            open_round = rnd(use_openai, "open", name="compact_context")[0]  # assistant with unanswered tool call
            self.msgs(agent).append(open_round)
            self.assertTrue(await agent._compact_conversation(trigger="tool"))
            self.assertNotIn("id-open", json.dumps(self.after_memory(agent)))
            self.assertIn("id-open", self.calls[0]["transcript"].split("c0 ")[-1])  # at the end of the transcript
            assert_pairs_intact(self, agent)
        await self.both(check)

    async def test_empty_head_does_not_fold(self):
        async def check(use_openai):
            agent = self.agent(use_openai, fold_tail_tokens=50_000)
            self.history(agent, past_turns=1, current_rounds=2)
            before = list(self.msgs(agent))
            with self.assertNoLogs(level="WARNING"):
                self.assertFalse(await agent._compact_conversation(trigger="manual"))
            self.assertEqual(self.calls, [])
            self.assertEqual(agent._fold_count, 0)
            self.assertEqual(self.msgs(agent), before)
        await self.both(check)

    async def test_shrink_drops_oldest_tail_units_first_into_transcript(self):
        async def check(use_openai):
            # budget allows the whole current turn, but the threshold only fits ~2 rounds next to the reserve
            agent = self.agent(use_openai, fold_tail_tokens=20_000)
            fixed = estimate_tokens([agent._system_prompt, agent.tools]) + FOLD_MEMORY_RESERVE_TOKENS
            agent.auto_compact_threshold = (fixed + 2600) / agent.effective_window
            req = self.history(agent, current_rounds=5)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            kept = self.after_memory(agent)
            self.assertIs(kept[0], req)
            kept_text = json.dumps(kept)
            self.assertIn("c4 ", kept_text)
            self.assertNotIn("c0 ", kept_text)
            self.assertIn("c0 ", self.calls[0]["transcript"])  # shrunk units were summarised, not lost
            assert_pairs_intact(self, agent)
        await self.both(check)

    async def test_request_withdrawn_after_summary_tail_kept(self):
        async def check(use_openai):
            # real memory far above the reserve: the request (already in the transcript) is withdrawn, the tail stays
            agent = self.agent(use_openai, mem=lambda: memory(pad=60_000))
            fixed = estimate_tokens([agent._system_prompt, agent.tools]) + FOLD_MEMORY_RESERVE_TOKENS
            agent.auto_compact_threshold = (fixed + 4000) / agent.effective_window
            req = self.history(agent)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            kept = self.after_memory(agent)
            self.assertFalse(any(m is req for m in kept))
            self.assertIn("c3 ", json.dumps(kept))
            self.assertIn("too large to keep verbatim", self.out.getvalue())
            self.assertIn("next fold", self.out.getvalue())  # still above threshold: reported, nothing more deleted
            assert_pairs_intact(self, agent)
        await self.both(check)

    async def test_two_folds_do_not_resummarise(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            self.history(agent)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            first_tail = [m for m in self.after_memory(agent)[1:]]
            o = use_openai
            for r in range(4, 9):
                self.msgs(agent).extend(rnd(o, f"c{r}"))
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            second = self.calls[1]["transcript"]
            self.assertNotIn("p0 ", second)  # first head content lives only in <previous-memory>
            self.assertIsNotNone(self.calls[1]["previous_memory"])
            # first tail's tool results (content starts "cN x") appear exactly once in the second transcript
            tags = [f"c{r} x" for r in range(4) if any(f"c{r} x" in json.dumps(m) for m in first_tail)]
            self.assertTrue(tags)
            for tag in tags:
                self.assertEqual(second.count(tag), 1, tag)
        await self.both(check)

    async def test_lost_request_still_warns_with_tail(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            self.history(agent)
            agent._turn_request = {"role": "user", "content": REQUEST}
            with self.assertLogs(level="WARNING") as logs:
                self.assertTrue(await agent._compact_conversation(trigger="auto"))
            self.assertIn("turn request lost before fold", "\n".join(logs.output))
        await self.both(check)


class TailZeroParityTests(Base):
    """fold_tail_tokens=0 must reproduce #41 exactly: transcript, prompt and kept messages."""

    async def test_zero_tail_matches_41(self):
        async def check(use_openai):
            for mode in ("single_json", "three_parallel"):
                agent = self.agent(use_openai, fold_tail_tokens=0, fold_memory_mode=mode)
                req = self.history(agent)
                prompts = []

                async def side_query(system, prompt):
                    prompts.append(prompt)
                    mem = memory()
                    if system in FOLD_COMPONENT_SYSTEMS.values():
                        name = next(k for k, v in FOLD_COMPONENT_SYSTEMS.items() if v == system)
                        return json.dumps(mem[name])
                    return json.dumps(mem)
                del agent._generate_folded_session_memory  # use the real one
                from agents.agent import _sanitize_for_utf8
                _, rest = split_previous_memory(_sanitize_for_utf8(self.msgs(agent)))
                expected = (build_openai_transcript if use_openai else build_anthropic_transcript)(rest)
                with patch.object(agent, "_build_side_query", return_value=side_query):
                    self.assertTrue(await agent._compact_conversation(trigger="auto"))
                if mode == "single_json":
                    self.assertEqual(prompts, [build_folding_user_prompt(expected)])
                else:
                    self.assertTrue(all(p.endswith(expected) for p in prompts))
                self.assertTrue(all("kept verbatim" not in p for p in prompts))
                self.assertEqual(self.after_memory(agent), [req])
                self.assertIs(self.after_memory(agent)[0], req)
        await self.both(check)


class PromptNoteTests(Base):
    async def test_note_only_when_tail_kept(self):
        for mode in ("single_json", "three_parallel"):
            for tail in (0, 3500):
                with self.subTest(mode=mode, tail=tail):
                    agent = self.agent(True, fold_tail_tokens=tail, fold_memory_mode=mode)
                    self.history(agent)
                    prompts = []

                    async def side_query(system, prompt):
                        prompts.append(prompt)
                        if system in FOLD_COMPONENT_SYSTEMS.values():
                            name = next(k for k, v in FOLD_COMPONENT_SYSTEMS.items() if v == system)
                            return json.dumps(memory()[name])
                        return json.dumps(memory())
                    del agent._generate_folded_session_memory
                    with patch.object(agent, "_build_side_query", return_value=side_query):
                        self.assertTrue(await agent._compact_conversation(trigger="auto"))
                    has_note = ["kept verbatim" in p for p in prompts]
                    self.assertEqual(has_note, [bool(tail)] * len(prompts))


class DefaultsAndPlumbingTests(Base):
    def test_default_budget_and_validation(self):
        big = Agent(api_key="fake", api_base="https://example.test/v1", custom_system_prompt="t", model="gpt-4o")
        self.assertEqual(big._fold_tail_budget(), min(20_000, int(0.20 * big.effective_window)))
        small = Agent(api_key="fake", api_base="https://example.test/v1", custom_system_prompt="t", model="unknown-small")
        self.assertEqual(small._fold_tail_budget(), int(0.20 * small.effective_window))
        self.assertEqual(Agent(api_key="fake", api_base="https://example.test/v1", custom_system_prompt="t",
                               fold_tail_tokens=0)._fold_tail_budget(), 0)
        with self.assertRaises(ValueError):
            Agent(api_key="fake", api_base="https://example.test/v1", custom_system_prompt="t", fold_tail_tokens=-1)

    def test_reserve_is_calibrated_and_bounded(self):
        self.assertEqual(FOLD_MEMORY_RESERVE_TOKENS, 4652)
        self.assertLessEqual(FOLD_MEMORY_RESERVE_TOKENS, 3 * 2500 * 4 // 3)

    def test_estimate_tokens_is_the_estimator_first_term(self):
        agent = self.agent(True)
        self.history(agent)
        agent.last_input_token_count = 0
        agent._last_request_bytes = None
        self.assertEqual(agent._estimated_input_tokens(),
                         estimate_tokens([agent._system_prompt, agent.tools, agent._openai_messages]))

    def test_eval_defaults_tail_off_and_child_inherits(self):
        from agents import eval_folding
        import inspect
        self.assertEqual(inspect.signature(eval_folding._run_case).parameters["fold_tail_tokens"].default, 0)
        src = inspect.getsource(eval_folding)
        self.assertIn('"--fold-tail-tokens", type=int, default=0', src)
        self.assertIn("fold_tail_tokens=fold_tail_tokens", src)
        agent_src = inspect.getsource(Agent._run_child_agent) if hasattr(Agent, "_run_child_agent") else ""
        self.assertIn("fold_tail_tokens=self.fold_tail_tokens", inspect.getsource(Agent))


class DefaultTailPathTests(Base):
    """Already-accepted paths (#2, #38, #41) re-run with the tail on (default budget rule via a small explicit cap)."""

    async def test_2_fallback_drops_tail_history_and_keeps_request(self):
        async def check(use_openai):
            agent = self.agent(use_openai, fold_tail_tokens=9_000)
            req = self.history(agent, past_turns=12, current_rounds=0)  # more history than the tail budget holds
            self.assertTrue(await agent._compact_conversation(trigger="overflow", ignore_fold_limit=True))
            kept = self.after_memory(agent)
            self.assertIn("past turn", json.dumps(kept))  # history turns kept in the tail
            self.assertTrue(agent._drop_earliest_messages(1))  # tail turns are droppable units
            self.assertTrue(any(m is req for m in self.msgs(agent)))
            assert_pairs_intact(self, agent)
        await self.both(check)

    async def test_38_retry_request_is_memory_request_tail(self):
        agent = self.agent(False)
        req = self.history(agent)
        seen = []

        async def stream():
            seen.append(list(agent._anthropic_messages))
            stop = "model_context_window_exceeded" if len(seen) == 1 else "end_turn"
            return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text="x")],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=1))
        agent._call_anthropic_stream = stream
        agent._turn_output_buffer = []
        # drive from inside a turn: reuse the prepared history as this turn's state
        agent._overflow_stop_handled = False
        self.assertTrue(await agent._recover_from_overflow_stop("model_context_window_exceeded", agent._capture_stream_output()))
        retry = agent._anthropic_messages
        self.assertTrue(retry[0]["content"].startswith("<session-folded-memory>"))
        self.assertIs(retry[1], req)
        self.assertGreater(len(retry), 2)
        assert_pairs_intact(self, agent)

    async def test_41_request_identity_and_position_with_tail(self):
        async def check(use_openai):
            agent = self.agent(use_openai)
            req = self.history(agent)
            self.assertTrue(await agent._compact_conversation(trigger="auto"))
            kept = self.after_memory(agent)
            self.assertIs(kept[0], req)
            self.assertTrue(all(m is not req for m in kept[1:]))
        await self.both(check)


if __name__ == "__main__":
    unittest.main()
