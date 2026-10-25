"""Round 3: #25 untrusted-output wrapping of tool results."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents.agent import Agent
from agents.result_storage import persist_tool_result

RAW_READ = "   1 | hello from a file"
RAW_SHELL = "shell says hi\n"


def make(use_openai: bool, **kwargs) -> Agent:
    options = dict(api_key="fake", custom_system_prompt="test", enable_mcp=False, enable_memory_prefetch=False,
                   enable_online_evolution=False, enable_folding=False)
    options.update({"api_base": "https://example.test/v1"} if use_openai else {"api_protocol": "anthropic"})
    options.update(kwargs)
    agent = Agent(**options)
    agent._auto_save = lambda strict=False: None
    return agent


class Harness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.out = io.StringIO()
        ctx = redirect_stdout(self.out)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        for target in (patch("agents.verify_on_stop._settings_files", return_value=[]),
                       patch("pathlib.Path.home", return_value=Path(self.home.name))):
            target.start()
            self.addCleanup(target.stop)

    async def drive(self, agent, rounds, results):
        """rounds: list of [(name, inp)] per model response; results: dict name -> raw tool output."""
        rounds = list(rounds)
        ids = iter(range(1000))

        async def run_checked(name, inp):
            return results[name]

        async def stream():
            step = rounds.pop(0) if rounds else None
            if step is None:
                if agent.use_openai:
                    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}]}
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")],
                                       usage=SimpleNamespace(input_tokens=1, output_tokens=1))
            calls = [(f"c{next(ids)}", n, i) for n, i in step]
            if agent.use_openai:
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": c, "type": "function", "function": {"name": n, "arguments": json.dumps(i)}} for c, n, i in calls]}}]}
            return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   content=[SimpleNamespace(type="tool_use", id=c, name=n, input=i) for c, n, i in calls])

        agent._run_checked_tool = run_checked
        setattr(agent, "_call_openai_stream" if agent.use_openai else "_call_anthropic_stream", stream)
        await (agent._chat_openai if agent.use_openai else agent._chat_anthropic)("go")

    @staticmethod
    def tool_contents(agent):
        if agent.use_openai:
            return [m["content"] for m in agent._openai_messages if m.get("role") == "tool"]
        return [b["content"] for m in agent._anthropic_messages if m.get("role") == "user" and isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "tool_result"]


class CurrentStateTests(Harness):
    """Pre-round-3 behaviour, pinned before the change (see 第 3 轮实施记录 for how each one changed)."""

    async def test_anthropic_site_stores_raw_result(self):
        agent = make(False)
        await self.drive(agent, [[("read_file", {"file_path": "a"})]], {"read_file": RAW_READ})
        self.assertEqual(self.tool_contents(agent), [RAW_READ])

    async def test_openai_concurrent_and_sequential_sites_store_raw_results(self):
        agent = make(True)
        await self.drive(agent, [[("read_file", {"file_path": "a"})], [("run_shell", {"command": "echo hi"})]],
                         {"read_file": RAW_READ, "run_shell": RAW_SHELL})
        self.assertEqual(self.tool_contents(agent), [RAW_READ, RAW_SHELL])

    async def test_recovery_reinserts_raw_checkpoint_result(self):
        for use_openai in (True, False):
            with self.subTest(use_openai=use_openai):
                agent = make(use_openai)
                msgs = agent._openai_messages if use_openai else agent._anthropic_messages
                msgs.append({"role": "user", "content": "go"})
                if use_openai:
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [
                        {"id": "k1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
                else:
                    msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": "k1", "name": "read_file", "input": {}}]})
                agent._tool_checkpoints = {"k1": {"name": "read_file", "status": "completed", "result": RAW_READ}}
                agent._repair_interrupted_tool_calls()
                self.assertEqual(self.tool_contents(agent), [RAW_READ])

    def test_large_result_preview_format(self):
        with patch("pathlib.Path.home", return_value=Path(tempfile.mkdtemp())):
            stored = persist_tool_result("run_shell", "line\n" * 10_000)["result"]
        self.assertTrue(stored.startswith("[Result too large ("))
        self.assertIn("Use read_file with offset/limit to read the required lines.]\n\nPreview (up to 200 lines / 6000 characters):\n", stored)


if __name__ == "__main__":
    unittest.main()
