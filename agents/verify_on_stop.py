"""verify-on-stop guard (audit item #3; Hermes, Harness Engineering §6.2).

When the model is about to stop after changing files without a verification run afterwards, the loop asks it
once to find and run the project's own checks. Conservative on purpose: unknown writes and unknown checks are
not guessed at, so the guard prefers missing a case over nagging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Bear runs on other people's projects, so the table lists common commands per ecosystem rather than this repo's.
VERIFY_COMMAND_KEYWORDS: tuple[str, ...] = (
    # Python
    "pytest", "unittest", "tox", "nox",
    # JavaScript / TypeScript
    "jest", "vitest", "mocha", "npm test", "npm run test", "yarn test", "pnpm test", "node --test",
    # Go / Rust / JVM / .NET / Ruby / make
    "go test", "cargo test", "cargo check", "mvn test", "gradle test", "dotnet test", "rspec",
    "make test", "make check",
    # lint / type checks
    "ruff", "flake8", "mypy", "pyright", "eslint", "tsc", "golangci-lint", "clippy", "node --check",
    "git diff --check",
)

WRITE_TOOLS = frozenset({"write_file", "edit_file"})

VERIFY_ON_STOP_MARKER = "[verify-on-stop]"
# Same prefix as agent.OVERFLOW_MARKER_TEXT (#2); a test pins the two together.
CONTEXT_OVERFLOW_MARKER = "[Context overflow:"
_HARNESS_MARKERS = (VERIFY_ON_STOP_MARKER, CONTEXT_OVERFLOW_MARKER)

_NOT_RUN_PREFIXES = ("Action denied", "User denied this action.", "Command timed out")
_MAX_LISTED_FILES = 5


def is_harness_injected_message(msg: Any) -> bool:
    """A user-role message the harness wrote itself; it never counts as the user's request."""
    if not isinstance(msg, dict) or msg.get("role") != "user" or not isinstance(msg.get("content"), str):
        return False
    return msg["content"].startswith(_HARNESS_MARKERS)


def is_successful_write(name: str, result: str) -> bool:
    # Not _looks_like_tool_failure: it flags any result containing "error", and a successful write echoes the
    # diff or file preview, so editing code that mentions "error" would read as a failed write.
    return name in WRITE_TOOLS and str(result).startswith("Successfully ")


def is_verification(name: str, inp: Any, result: str, keywords: tuple[str, ...]) -> bool:
    """A shell command matching the keyword table that actually ran; the exit code does not matter."""
    if name != "run_shell" or not isinstance(inp, dict) or not isinstance(inp.get("command"), str):
        return False
    if str(result).startswith(_NOT_RUN_PREFIXES):
        return False
    command = inp["command"].lower()
    return any(keyword.lower() in command for keyword in keywords)


def _settings_files() -> list[tuple[Path, bool]]:
    from .project_trust import is_project_trusted
    return [(Path.home() / ".bear" / "settings.json", True),
            (Path.cwd() / ".bear" / "settings.json", is_project_trusted())]


def load_verify_command_keywords() -> tuple[str, ...]:
    """Built-in table plus settings.json `verifyOnStop.extraCommands`; project entries only for trusted projects."""
    from .tools import _load_settings
    extra: list[str] = []
    for path, trusted in _settings_files():
        settings = _load_settings(path)
        if not trusted or not isinstance(settings, dict) or not isinstance(settings.get("verifyOnStop"), dict):
            continue
        commands = settings["verifyOnStop"].get("extraCommands")
        if isinstance(commands, list):
            # An empty keyword would match every command and silence the guard.
            extra.extend(c.strip() for c in commands if isinstance(c, str) and c.strip())
    return VERIFY_COMMAND_KEYWORDS + tuple(extra)


def verify_on_stop_prompt(files: list[str]) -> str:
    listed = ", ".join(files[:_MAX_LISTED_FILES])
    if len(files) > _MAX_LISTED_FILES:
        listed += f", +{len(files) - _MAX_LISTED_FILES} more"
    return (
        f"{VERIFY_ON_STOP_MARKER} You changed files in this turn ({listed}) but have not run any tests, linters or "
        "type checks since the last change. Before finishing, find this project's own verification command (look at "
        "the README, CI config, Makefile, package.json, pyproject.toml or similar) and run it. If verification does "
        "not apply — for example the change is documentation only, the project has no tests, or the user asked you "
        "not to run them — say so briefly and finish."
    )
