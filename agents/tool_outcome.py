"""Tool-call outcome signals for stuck detection (audit item #5; Harness Engineering Table 5, §7.3).

Pure functions over the *raw* tool result: results over 30KB are replaced by a preview whose text starts with
"[Result too large" and names a uuid file, so the stored text can neither show the failure prefix nor repeat.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

STUCK_WARNING_MARKER = "[stuck-warning]"

_FAILURE_PREFIXES = ("Tool not available in this run:", "Error", "Action denied:", "User denied this action.")
_TOOL_FAILURE_PREFIXES = {
    "run_shell": ("Command failed (exit code", "Command timed out after"),
    # "modified externally" warning: the write did not happen.
    "write_file": ("Warning:",),
    "edit_file": ("Warning:",),
    "compact_context": ("Compaction skipped:", "Context folding is disabled", "Context folding limit reached"),
}


def is_tool_failure(name: str, raw_result: str) -> bool:
    # Prefix rules only, not a search for "error" anywhere: successful reads and diffs routinely contain that word,
    # and a failed write's "modified externally" warning contains none of the old keywords.
    text = str(raw_result)
    return text.startswith(_FAILURE_PREFIXES) or text.startswith(_TOOL_FAILURE_PREFIXES.get(name, ()))


def call_signature(name: str, args: Any, raw_arguments: str | None = None) -> str:
    """Same tool + same arguments (key order ignored, no whitespace/quote normalisation).

    OpenAI calls whose JSON failed to parse use their raw argument string; Anthropic's parse fallback is `{}`.
    """
    payload = args if isinstance(args, dict) else {"__raw_arguments__": raw_arguments if raw_arguments is not None else args}
    encoded = json.dumps([name, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()


def result_digest(raw_result: str) -> str:
    return hashlib.sha256(str(raw_result).encode("utf-8", errors="replace")).hexdigest()


def stuck_warning(name: str, count: int) -> str:
    return (f"\n\n{STUCK_WARNING_MARKER} This exact call ({name}, same arguments) has now run {count} times in a row "
            "without progress (same failure or same result). Repeating it will not help: try a different approach, "
            "check the cause, or ask the user.")


def stuck_stop_message(name: str, count: int, last_result: str) -> str:
    first_line = str(last_result).strip().splitlines()[0][:200] if str(last_result).strip() else "(empty result)"
    return (f"Stopped: the same call ({name}, same arguments) ran {count} times in a row without progress. "
            f"Last result: {first_line}")
