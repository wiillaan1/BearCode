"""Round 2: #21 tool/operation-key split, #23 confirm fallback, #22 deny-before-bypass and the hard floor.

Every hard-floor case here is a string decision only: nothing is executed.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import tools
from agents.tools import check_permission

MODES = ("default", "acceptEdits", "plan", "dontAsk", "bypassPermissions")


def no_rules():
    return patch("agents.tools._check_permission_rules", return_value=None)


class FormerStateTests(unittest.TestCase):
    """Pinned before round 2 (commit 4c45de0) as CurrentStateTests; how each one changed:
    1. skill / agent / tool_search allowed in every mode  → unchanged (now explicit ORCHESTRATION_TOOLS; kept as is)
    2. unclassified tool fell back to allow                → confirm (dontAsk: deny)
    3. bypass ignored the user's deny rules                → deny rules apply in bypass
    4. literal catastrophic command was a normal shell call → hard floor denies it in every mode
    """

    def test_orchestration_tools_allowed_in_every_mode(self):
        with no_rules():
            for name in ("skill", "agent", "tool_search"):
                for mode in MODES:
                    self.assertEqual(check_permission(name, {}, mode)["action"], "allow", (name, mode))

    def test_unclassified_tool_no_longer_allowed(self):
        with no_rules():
            self.assertEqual(check_permission("new_tool", {}, "default")["action"], "confirm")
            self.assertEqual(check_permission("new_tool", {}, "dontAsk")["action"], "deny")

    def test_bypass_respects_deny_rules(self):
        with patch("agents.tools._check_permission_rules", return_value="deny"):
            self.assertEqual(check_permission("read_file", {"file_path": "secret.txt"}, "bypassPermissions")["action"], "deny")

    def test_literal_catastrophic_command_hits_the_floor(self):
        # string decision only, never executed
        with no_rules():
            self.assertEqual(check_permission("run_shell", {"command": "rm -rf /"}, "bypassPermissions")["action"], "deny")
            self.assertEqual(check_permission("run_shell", {"command": "rm -rf /"}, "default")["action"], "deny")


# ---------------------------------------------------------------- new behaviour (round 2)

import asyncio
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock

from agents.agent import Agent
from agents.tool_outcome import is_tool_failure
from agents.tool_schemas import tool_definitions
from agents.tools import (EDIT_TOOLS, ORCHESTRATION_TOOLS, READ_TOOLS, RUNTIME_OPERATION_KEYS,
                          hard_floor_violation)

SHOULD_DENY = {
    "rm root/home": [
        "rm -rf /", "sudo rm -fr /*", "rm -r -f ~", 'rm --recursive --force "$HOME"', "ls; rm -rf /",
        "echo $(rm -rf ~)", "r''m -rf /", "make && sudo -E rm -Rf /", "true\nrm -rf /",
        "rm -rf --no-preserve-root /", "/bin/rm -rf ${HOME}", "rm -rfv ~/*", "rm -rf /.",
        'bash -c "rm -rf /"', "sudo sh -c 'rm -rf ~'", '/bin/zsh -c "cd x && rm -rf /"',
    ],
    "mkfs device": ["mkfs.ext4 /dev/sda1", "sudo mkfs -t ext4 /dev/nvme0n1", "x && mkfs.vfat /dev/disk2"],
    "dd to block device": ["dd if=x of=/dev/sda bs=1M", "sudo dd of=/dev/nvme0n1 if=img", "dd if=/dev/zero of=/dev/disk2"],
    "fork bomb": [":(){ :|:& };:", ":() { : | : & } ; :", "bomb(){ bomb|bomb& };bomb"],
}
SHOULD_NOT_DENY = {
    "rm root/home": [
        "rm -rf ./build", "rm -rf /tmp/x", "rm -rf ~/project/dist", "rm -f /", "echo rm -rf /",
        'git commit -m "fix; rm -rf /"', 'grep "rm -rf /" notes.md', 'rm -rf "$DIR"', 'bash -c "echo hi"',
        "bash script.sh", "sh -c", "rm -rf",
    ],
    "mkfs device": ["mkfs.ext4 disk.img", "echo mkfs.ext4 /dev/sda", "man mkfs"],
    "dd to block device": ["dd if=/dev/zero of=./blob bs=1M count=1", "dd if=x of=/dev/null", "dd if=/dev/sda of=backup.img"],
    "fork bomb": ['echo "(){}"', "f(){ g|h& };f", "f(){ echo hi; }; f", "a|b"],
}
KNOWN_MISSES = [  # documented in 第 2 轮实施记录: literal floor, not a security boundary
    "rm -rf $DIR", "rm -r /usr", "find / -delete", "xargs rm -rf", "sudo -u root rm -rf /",
    'bash -c "$(echo cm0gLXJmIC8= | base64 -d)"', "python -c \"import shutil; shutil.rmtree('/')\"",
    'git commit -m "fix; rm -rf /', "mkfs.ext4 $DEV", "mke2fs /dev/sda", "cat img > /dev/sda", "dd of=$DEV",
    'bash -c "bash -c \\"rm -rf /\\""',  # only one level of -c is unwrapped
]


class HardFloorStringTests(unittest.TestCase):
    def test_should_deny(self):
        for rule, commands in SHOULD_DENY.items():
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(hard_floor_violation(command), rule)

    def test_should_not_deny(self):
        for rule, commands in SHOULD_NOT_DENY.items():
            for command in commands:
                with self.subTest(command=command):
                    self.assertIsNone(hard_floor_violation(command))

    def test_known_misses_stay_documented(self):
        for command in KNOWN_MISSES:
            with self.subTest(command=command):
                self.assertIsNone(hard_floor_violation(command))

    def test_fork_bomb_pipe_is_a_literal(self):
        # If "|" in the pattern were an unescaped alternation, these would match on fragments.
        for text in ("a|b", ":", "x&y", "f(){", "};f", "bomb|bomb"):
            with self.subTest(text=text):
                self.assertIsNone(hard_floor_violation(text))
        for text in SHOULD_DENY["fork bomb"]:
            self.assertEqual(hard_floor_violation(text), "fork bomb")
        self.assertIn(r"\|", tools._FORK_BOMB.pattern)


class CheckPermissionTests(unittest.TestCase):
    def test_hard_floor_in_every_mode_and_over_allow_rules(self):
        for rule_result in (None, "allow"):
            with patch("agents.tools._check_permission_rules", return_value=rule_result):
                for mode in MODES:
                    with self.subTest(mode=mode, rule=rule_result):
                        decision = check_permission("run_shell", {"command": "rm -rf /"}, mode)
                        self.assertEqual(decision["action"], "deny")
                        self.assertTrue(decision["message"].startswith("Hard floor: rm root/home"))
                        self.assertIn("not a security boundary", decision["message"])

    def test_hard_floor_checks_mcp_command(self):
        with no_rules():
            self.assertEqual(check_permission("mcp__box__exec", {"command": "rm -rf ~"}, "bypassPermissions")["action"], "deny")
            self.assertEqual(check_permission("mcp__box__exec", {"command": "ls"}, "default")["action"], "confirm")
            self.assertEqual(check_permission("mcp__box__exec", {"command": 3}, "default")["action"], "confirm")

    def test_deny_rules_apply_in_bypass(self):
        with patch("agents.tools._check_permission_rules", return_value="deny"):
            self.assertEqual(check_permission("read_file", {"file_path": "secret.txt"}, "bypassPermissions")["action"], "deny")
        with no_rules():
            self.assertEqual(check_permission("run_shell", {"command": "ls"}, "bypassPermissions")["action"], "allow")

    def test_unclassified_tool_needs_confirmation(self):
        with no_rules():
            for mode, expected in (("default", "confirm"), ("acceptEdits", "confirm"), ("plan", "confirm"),
                                   ("dontAsk", "deny"), ("bypassPermissions", "allow")):
                with self.subTest(mode=mode):
                    decision = check_permission("new_tool", {}, mode)
                    self.assertEqual(decision["action"], expected)
                    if expected != "allow":
                        self.assertIn("Unclassified tool: new_tool", decision["message"])

    def test_every_registered_tool_is_classified(self):
        classified = READ_TOOLS | EDIT_TOOLS | ORCHESTRATION_TOOLS | {"run_shell", "enter_plan_mode", "exit_plan_mode"}
        registered = {tool["name"] for tool in tool_definitions}
        self.assertEqual(registered - classified, set(), "register a tool's permission class in tools.py")
        unknown = (READ_TOOLS | EDIT_TOOLS | ORCHESTRATION_TOOLS) - registered - RUNTIME_OPERATION_KEYS
        self.assertEqual(unknown, set())
        self.assertTrue(RUNTIME_OPERATION_KEYS.isdisjoint(registered))


class LoopIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, **kwargs):
        options = dict(api_key="fake", api_base="https://example.test/v1", custom_system_prompt="t", enable_mcp=False,
                       enable_memory_prefetch=False, enable_online_evolution=False)
        options.update(kwargs)
        return Agent(**options)

    async def test_confirmed_hard_floor_still_denied_and_never_executed(self):
        for mode in ("default", "bypassPermissions"):
            with self.subTest(mode=mode):
                agent = self.agent(permission_mode=mode, confirm_fn=AsyncMock(return_value=True))
                with no_rules(), patch("agents.agent.execute_tool", AsyncMock(return_value="ran")) as run, \
                        redirect_stdout(io.StringIO()):
                    result = await agent._run_checked_tool("run_shell", {"command": "rm -rf /"})
                run.assert_not_awaited()
                self.assertTrue(result.startswith("Action denied: Hard floor: rm root/home"))
                self.assertTrue(is_tool_failure("run_shell", result))

    async def test_authorize_operation_respects_floor(self):
        agent = self.agent(permission_mode="bypassPermissions")
        with no_rules():
            self.assertFalse(await agent.authorize_operation("run_shell", {"command": "mkfs.ext4 /dev/sda"}))

    async def test_one_shot_orchestration_tools_still_run_without_prompting(self):
        agent = self.agent()  # default mode, no confirm_fn: one-shot
        self.assertIsNone(agent.confirm_fn)
        with no_rules(), patch("builtins.input", side_effect=AssertionError("must not prompt")), \
                patch.object(agent, "_execute_tool_call", AsyncMock(return_value="ok")) as run, \
                redirect_stdout(io.StringIO()):
            for name, inp in (("tool_search", {"query": "x"}), ("skill", {"skill_name": "demo"}),
                              ("agent", {"prompt": "do x", "description": "d"})):
                with self.subTest(tool=name):
                    self.assertEqual(await agent._run_checked_tool(name, inp), "ok")
        self.assertEqual(run.await_count, 3)


class DiscoveryFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tool_has_no_permission_mode_parameter(self):
        import inspect
        self.assertNotIn("permission_mode", inspect.signature(tools.execute_tool).parameters)
        self.assertNotIn("permission_mode", inspect.signature(tools._discovery_path_filter).parameters)


if __name__ == "__main__":
    unittest.main()
