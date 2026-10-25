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


class CurrentStateTests(unittest.TestCase):
    """Pre-round-2 behaviour, pinned before the change (see 第 2 轮实施记录 for how each one changed)."""

    def test_orchestration_tools_allowed_in_every_mode(self):
        with no_rules():
            for name in ("skill", "agent", "tool_search"):
                for mode in MODES:
                    self.assertEqual(check_permission(name, {}, mode)["action"], "allow", (name, mode))

    def test_unclassified_tool_falls_back_to_allow(self):
        with no_rules():
            for mode in ("default", "dontAsk"):
                self.assertEqual(check_permission("new_tool", {}, mode)["action"], "allow")

    def test_bypass_ignores_deny_rules(self):
        with patch("agents.tools._check_permission_rules", return_value="deny"):
            self.assertEqual(check_permission("read_file", {"file_path": "secret.txt"}, "bypassPermissions")["action"], "allow")

    def test_literal_catastrophic_command_is_only_a_normal_shell_call(self):
        # string decision only, never executed
        with no_rules():
            self.assertEqual(check_permission("run_shell", {"command": "rm -rf /"}, "bypassPermissions")["action"], "allow")
            self.assertEqual(check_permission("run_shell", {"command": "rm -rf /"}, "default")["action"], "confirm")


if __name__ == "__main__":
    unittest.main()
