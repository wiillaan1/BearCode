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


class CurrentStateTests(unittest.TestCase):
    """Pre-round-4 behaviour, pinned before the change (see 第 4 轮实施记录 for how each one changed)."""

    def test_pruning_ignores_a_just_appended_large_result(self):
        agent = make()
        agent.last_input_token_count = 0  # previous response was small (or a fold just happened)
        big_round(agent, "t1", 300_000)  # ~100K estimated tokens on a 108K effective window
        agent._run_compression_pipeline()
        self.assertEqual(len(agent._openai_messages[-1]["content"]), 300_000)  # not budgeted

    def test_edit_miss_message_is_bare(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.py"
            path.write_text("def add(a, b):\n    return a + b\n")
            result = _edit_file({"file_path": str(path), "old_string": "return a - b", "new_string": "x"})
        self.assertEqual(result, f"Error: old_string not found in {path}")

    def test_print_info_after_unterminated_stream_shares_the_line(self):
        out = io.StringIO()
        with redirect_stdout(out), patch.object(ui, "console", ui.Console(file=out, highlight=False, width=80)):
            ui.print_assistant_text("Done editing.")
            ui.print_info("note")
        self.assertIn("Done editing.╭", out.getvalue())


if __name__ == "__main__":
    unittest.main()
