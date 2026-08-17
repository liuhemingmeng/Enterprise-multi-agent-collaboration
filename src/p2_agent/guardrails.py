from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict


class Severity(str, Enum):  # noqa: UP042 - keep (str, Enum) for broad Python compatibility
    info = "info"
    warning = "warning"
    critical = "critical"


class GuardrailFinding(BaseModel):
    """A single guardrail observation attached to a task.

    Severity drives the action: ``critical`` input findings block execution at
    the service boundary; everything else is detection-only and recorded for
    audit / the evaluation harness.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = ""
    field: str
    rule: str
    severity: Severity
    message: str
    action: str = "recorded"


# Conservative, high-precision injection signatures.  We deliberately avoid
# fuzzy heuristics so ordinary business text ("写一份智慧城市方案") never trips
# a false positive.  Only explicit instruction-override phrases match.
INJECTION_PATTERNS = [
    "忽略以上",
    "忽略上述",
    "忽略前面的",
    "ignore the above",
    "ignore previous",
    "ignore all previous",
    "disregard",
    "忘记你的指令",
    "forget your instructions",
    "system prompt",
    "你现在是",
    "假装你是",
    "jailbreak",
    "作为 dan",
    "as dan",
    "new instructions",
    "重新设定",
    "override",
    "越权",
    "忽略之前",
]


def check_input(text: str) -> list[GuardrailFinding]:
    """Scan user-supplied input for prompt-injection attempts."""
    findings: list[GuardrailFinding] = []
    low = (text or "").lower()
    for pat in INJECTION_PATTERNS:
        if pat.lower() in low:
            findings.append(
                GuardrailFinding(
                    field="user_goal",
                    rule="prompt_injection",
                    severity=Severity.critical,
                    message=f"检测到疑似提示注入关键词：{pat}",
                    action="blocked",
                )
            )
            break
    if len(text or "") > 4000:
        findings.append(
            GuardrailFinding(
                field="user_goal",
                rule="input_too_long",
                severity=Severity.warning,
                message="输入长度超过 4000 字符，已标记",
                action="recorded",
            )
        )
    return findings


def check_output(field: str, value: Any) -> list[GuardrailFinding]:
    """Validate a produced artifact against its expected shape.

    This is schema-level, not semantic: we confirm the artifact is non-empty
    and carries the structural markers the rest of the pipeline depends on
    (e.g. citations in the draft).  Semantic quality is the reviewer's job.
    """
    findings: list[GuardrailFinding] = []
    text = value if isinstance(value, str) else str(value)
    if field == "draft":
        if not (text or "").strip():
            findings.append(
                GuardrailFinding(
                    field="draft",
                    rule="empty_draft",
                    severity=Severity.warning,
                    message="草稿为空",
                    action="recorded",
                )
            )
        elif not any(marker in text for marker in ("[", "来源", "引用")):
            findings.append(
                GuardrailFinding(
                    field="draft",
                    rule="no_citation",
                    severity=Severity.warning,
                    message="草稿缺少引用标记（[id] / 来源 / 引用）",
                    action="recorded",
                )
            )
    elif field == "review":
        if "decision" not in text:
            findings.append(
                GuardrailFinding(
                    field="review",
                    rule="review_shape",
                    severity=Severity.warning,
                    message="评审结果缺少 decision 字段",
                    action="recorded",
                )
            )
    return findings


def check_tool_result(item: Any) -> list[GuardrailFinding]:
    """Inspect a single tool/retrieval result for injection or abnormality."""
    findings: list[GuardrailFinding] = []
    text = getattr(item, "text", None)
    if text is None:
        text = item if isinstance(item, str) else str(item)
    if len(text or "") > 5000:
        findings.append(
            GuardrailFinding(
                field="tool_result",
                rule="tool_result_oversized",
                severity=Severity.warning,
                message="工具返回超长（>5000 字符），已标记截断",
                action="flagged",
            )
        )
    low = (text or "").lower()
    for pat in INJECTION_PATTERNS:
        if pat.lower() in low:
            findings.append(
                GuardrailFinding(
                    field="tool_result",
                    rule="tool_result_injection",
                    severity=Severity.warning,
                    message=f"工具结果疑似包含注入内容：{pat}",
                    action="flagged",
                )
            )
            break
    return findings


class GuardrailStore:
    """In-process finding buffer keyed by task_id (audit trail)."""

    def __init__(self) -> None:
        self._findings: dict[str, list[GuardrailFinding]] = {}
        self._lock = Lock()

    def add(self, finding: GuardrailFinding) -> None:
        with self._lock:
            self._findings.setdefault(finding.task_id, []).append(finding)

    def list(self, task_id: str) -> list[GuardrailFinding]:
        with self._lock:
            return list(self._findings.get(str(task_id), []))

    def reset(self) -> None:
        with self._lock:
            self._findings.clear()


guardrail_store = GuardrailStore()


def run_node_guardrails(node_name: str, state: Any, out_dict: dict) -> None:
    """Detection-only guardrails applied to a node's output.

    Findings are recorded (never block) so the workflow keeps running while we
    still get an audit trail.  The only *blocking* guardrail is the input
    check at the service boundary (see ``service.submit`` / ``create_and_run``).
    """
    tid = str(state.task_id)
    findings: list[GuardrailFinding] = []
    if node_name == "planner":
        findings += check_output("plan", out_dict.get("plan"))
    elif node_name == "retriever":
        for ev in out_dict.get("evidence", []) or []:
            findings += check_tool_result(ev)
    elif node_name == "writer":
        findings += check_output("draft", out_dict.get("draft", ""))
    elif node_name == "reviewer":
        findings += check_output("review", out_dict.get("review"))
    for f in findings:
        f.task_id = tid
        guardrail_store.add(f)
