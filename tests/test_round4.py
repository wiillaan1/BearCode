"""Round 4: #39 pruning utilisation, #20 closest snippet for edit_file misses, #42 newline before print_info."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents import ui
from agents.agent import Agent
from agents.tools import _edit_file
from types import SimpleNamespace


def make(use_openai: bool = True, **kwargs) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=True, model="gpt-4o", fold_tail_tokens=0)
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    options.update(kwargs)
    return Agent(**options)


def big_round(agent: Agent, tag: str, size: int) -> None:
    msgs = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    if agent.use_openai:
        msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": tag, "type": "function",
                  "function": {"name": "read_file", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": tag, "content": "z" * size}]
    else:
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": tag, "name": "read_file", "input": {}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tag, "content": "z" * size}]}]


class FormerStateTests(unittest.TestCase):
    """Pinned before round 4 (commit c3744f9) as CurrentStateTests; how each one changed:
    1. pruning ignored a just-appended large result  → it is budgeted in the same round (#39)
    2. edit miss was a bare error                     → same prefix, plus a wrapped closest snippet (#20)
    3. Info box shared the line with streamed text    → a newline is inserted first (#42)
    """

    def test_pruning_sees_a_just_appended_large_result(self):
        agent = make()
        agent.last_input_token_count = 0
        big_round(agent, "t1", 300_000)
        agent._run_compression_pipeline()
        self.assertLess(len(agent._openai_messages[-1]["content"]), 300_000)

    def test_edit_miss_keeps_prefix_and_adds_snippet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.py"
            path.write_text("def add(a, b):\n    return a + b\n")
            result = _edit_file({"file_path": str(path), "old_string": "return a - b", "new_string": "x"})
        self.assertTrue(result.startswith(f"Error: old_string not found in {path}\n\nClosest match"))

    def test_print_info_after_unterminated_stream_starts_a_new_line(self):
        out = io.StringIO()
        with redirect_stdout(out), patch.object(ui, "console", ui._LineAwareConsole(file=out, highlight=False, width=80)):
            ui._stream_line_open = False
            ui.print_assistant_text("Done editing.")
            ui.print_info("note")
        self.assertIn("Done editing.\n╭", out.getvalue())


# ---------------------------------------------------------------- new behaviour (round 4)

import time

from agents import untrusted
from agents.tool_outcome import is_tool_failure
from agents.tools import _SNIPPET_MAX_CHARS, _SNIPPET_MAX_LINES


def edit(content: str, old: str, new: str = "X") -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "f.py"
        path.write_text(content)
        return _edit_file({"file_path": str(path), "old_string": old, "new_string": new})


SOURCE = "def add(a, b):\n    total = a + b\n    return total\n\n\ndef sub(a, b):\n    return a - b\n"


class PruningUtilisationTests(unittest.TestCase):
    def test_just_appended_large_result_is_seen_in_the_same_round(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make(use_openai)
                agent.last_input_token_count = 0
                big_round(agent, "t1", 300_000)
                agent._run_compression_pipeline()
                msgs = agent._openai_messages if use_openai else agent._anthropic_messages
                content = msgs[-1]["content"] if use_openai else msgs[-1]["content"][0]["content"]
                self.assertLess(len(content), 300_000)
                self.assertIn("budgeted", content)

    def test_small_history_untouched(self):
        agent = make()
        big_round(agent, "t1", 3_000)
        agent._run_compression_pipeline()
        self.assertEqual(len(agent._openai_messages[-1]["content"]), 3_000)


class ClosestSnippetTests(unittest.TestCase):
    def test_miss_carries_wrapped_snippet_after_the_prefix(self):
        result = edit(SOURCE, "    total = a - b\n    return total")
        self.assertTrue(result.startswith("Error: old_string not found in "))
        self.assertTrue(is_tool_failure("edit_file", result))
        self.assertIn('<tool_output tool="edit_file">', result)
        self.assertIn("    2 |     total = a + b", result)
        self.assertIn("    3 |     return total", result)
        self.assertTrue(result.rstrip().endswith("</tool_output>"))
        self.assertIn("not applied", result)

    def test_closing_tag_in_snippet_is_escaped(self):
        content = "x = '</tool_output>'\ny = 2\n"
        result = edit(content, "x = '</tool_output>!'")
        self.assertEqual(result.lower().count("</tool_output>"), 1)
        self.assertIn("&lt;/tool_output>", result)

    def test_no_snippet_when_nothing_is_similar(self):
        result = edit(SOURCE, "completely unrelated text that is nowhere")
        self.assertTrue(result.startswith("Error: old_string not found in "))
        self.assertNotIn("<tool_output", result)

    def test_success_and_multiple_match_paths_unchanged(self):
        self.assertTrue(edit(SOURCE, "return a - b", "return b - a").startswith("Successfully edited "))
        dup = "a = 1\na = 1\n"
        self.assertRegex(edit(dup, "a = 1"), r"^Error: old_string found 2 times in .*\. Must be unique\.$")

    def test_snippet_is_bounded(self):
        long_old = "\n".join(f"line {i} = value_{i}" for i in range(40))
        content = long_old.replace("line 5 ", "line 5x ") + "\n"
        result = edit(content, long_old)
        snippet = result.split('<tool_output tool="edit_file">\n', 1)[1].rsplit("\n</tool_output>", 1)[0]
        self.assertLessEqual(len(snippet.splitlines()), _SNIPPET_MAX_LINES)
        self.assertLessEqual(len(snippet), _SNIPPET_MAX_CHARS)

    def test_large_file_stays_fast(self):
        content = "".join(f"value_{i} = compute({i}, factor=3)\n" for i in range(60_000))
        started = time.monotonic()
        result = edit(content, "value_41234 = compute(41234, factr=3)")
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertIn("41235 | value_41234 = compute(41234, factor=3)", result)

    def test_wrapping_switch_off_leaves_snippet_unwrapped(self):
        with patch.object(untrusted, "WRAP_ENABLED", False):
            result = edit(SOURCE, "    total = a - b")
        self.assertNotIn("<tool_output", result)
        self.assertIn("    2 |     total = a + b", result)


class PrintInfoNewlineTests(unittest.TestCase):
    def capture(self, actions):
        out = io.StringIO()
        with redirect_stdout(out), patch.object(ui, "console", ui._LineAwareConsole(file=out, highlight=False, width=80)):
            ui._stream_line_open = False
            actions()
        return out.getvalue()

    def test_unterminated_stream_gets_a_newline(self):
        text = self.capture(lambda: (ui.print_assistant_text("Done editing."), ui.print_info("note")))
        self.assertIn("Done editing.\n╭", text)

    def test_terminated_stream_gets_no_extra_blank_line(self):
        text = self.capture(lambda: (ui.print_assistant_text("Done editing.\n"), ui.print_info("note")))
        self.assertIn("Done editing.\n╭", text)
        self.assertNotIn("Done editing.\n\n", text)

    def test_other_console_output_closes_the_line(self):
        text = self.capture(lambda: (ui.print_assistant_text("Partial"), ui.console.print("panel"), ui.print_info("note")))
        self.assertNotIn("\n\n╭", text)



class EveryPanelNewlineTests(unittest.TestCase):
    PANELS = {
        "info": lambda: ui.print_info("note"),
        "cost": lambda: ui.print_cost(10, 2),
        "tool call": lambda: ui.print_tool_call("read_file", {"file_path": "a.py"}),
        "warning": lambda: ui.print_warning("careful"),
        "error": lambda: ui.print_error("boom"),
    }

    def run_panel(self, streamed: str, panel) -> str:
        out = io.StringIO()
        with redirect_stdout(out), patch.object(ui, "console", ui._LineAwareConsole(file=out, highlight=False, width=80)):
            ui._stream_line_open = False
            ui.print_assistant_text(streamed)
            panel()
        return out.getvalue()

    def test_each_panel_starts_a_new_line_after_unterminated_stream(self):
        for name, panel in self.PANELS.items():
            with self.subTest(panel=name):
                text = self.run_panel("Partial answer", panel)
                self.assertTrue(text.startswith("Partial answer\n"), repr(text[:40]))

    def test_each_panel_adds_no_blank_line_after_terminated_stream(self):
        for name, panel in self.PANELS.items():
            with self.subTest(panel=name):
                text = self.run_panel("Full line\n", panel)
                self.assertTrue(text.startswith("Full line\n"))
                self.assertFalse(text.startswith("Full line\n\n"), repr(text[:40]))


# ---- #36: output-limit parameter switch ----

import asyncio
import json as _json
from unittest.mock import AsyncMock

import httpx
import openai

from agents import openai_params


def _bad_request(message: str) -> openai.BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://example.test/v1/chat/completions"))
    return openai.BadRequestError(message, response=resp, body={"error": {"message": message}})


MCT = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."


class OutputLimitParamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(openai_params, "_COMPLETION_TOKENS_MODELS", set())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_switches_once_and_remembers_the_model(self):
        create = AsyncMock(side_effect=[_bad_request(MCT), "ok1", "ok2"])
        self.assertEqual(await openai_params.create_with_output_limit(create, {"model": "new-model"}, 64), "ok1")
        self.assertEqual([c.kwargs for c in create.await_args_list],
                         [{"model": "new-model", "max_tokens": 64}, {"model": "new-model", "max_completion_tokens": 64}])
        self.assertEqual(await openai_params.create_with_output_limit(create, {"model": "new-model"}, 64), "ok2")
        self.assertEqual(create.await_count, 3)  # no extra 400 the second time
        self.assertEqual(create.await_args_list[-1].kwargs, {"model": "new-model", "max_completion_tokens": 64})

    async def test_unrelated_400_and_failed_retry_raise_unchanged(self):
        other = _bad_request("Invalid value for 'messages'.")
        create = AsyncMock(side_effect=[other])
        with self.assertRaises(openai.BadRequestError) as ctx:
            await openai_params.create_with_output_limit(create, {"model": "m"}, 64)
        self.assertIs(ctx.exception, other)
        self.assertEqual(create.await_count, 1)
        second = _bad_request("Rate-limited differently")
        create = AsyncMock(side_effect=[_bad_request(MCT), second])
        with self.assertRaises(openai.BadRequestError) as ctx:
            await openai_params.create_with_output_limit(create, {"model": "m2"}, 64)
        self.assertIs(ctx.exception, second)
        self.assertEqual(create.await_count, 2)  # at most one switch

    async def test_no_limit_sends_neither_parameter(self):
        create = AsyncMock(return_value="ok")
        await openai_params.create_with_output_limit(create, {"model": "m"}, None)
        self.assertEqual(create.await_args.kwargs, {"model": "m"})

    async def test_side_query_switches_too(self):
        agent = make(True, model="new-model")
        msg = SimpleNamespace(content="hi", tool_calls=None)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        create = AsyncMock(side_effect=[_bad_request(MCT), resp])
        agent._openai_client.chat.completions.create = create
        self.assertEqual(await agent._build_side_query(max_tokens=32)("sys", "user"), "hi")
        self.assertNotIn("max_tokens", create.await_args_list[1].kwargs)
        self.assertEqual(create.await_args_list[1].kwargs["max_completion_tokens"], 32)

    async def test_main_loop_switch_happens_before_streaming_and_prints_once(self):
        bodies = []

        def handler(request):
            body = _json.loads(request.content)
            bodies.append(body)
            if "max_tokens" in body:
                return httpx.Response(400, json={"error": {"message": MCT, "type": "invalid_request_error",
                                                           "param": "max_tokens", "code": "unsupported_parameter"}})
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "new-model",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello once"}, "finish_reason": "stop"}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=("data: " + _json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

        agent = make(True, model="new-model", max_output_tokens=50, enable_folding=False)
        agent._auto_save = lambda strict=False: None
        await agent._openai_client.close()
        agent._openai_client = openai.AsyncOpenAI(api_key="fake", base_url="http://mock.test/v1", max_retries=0,
                                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        out = io.StringIO()
        with redirect_stdout(out), patch("agents.verify_on_stop._settings_files", return_value=[]):
            agent._turn_output_buffer = []
            await agent._chat_openai("hi")
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["max_completion_tokens"], 50)
        self.assertNotIn("max_tokens", bodies[1])
        self.assertEqual(out.getvalue().count("hello once"), 1)
        self.assertEqual("".join(agent._turn_output_buffer).strip(), "hello once")


if __name__ == "__main__":
    unittest.main()
