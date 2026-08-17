from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DOMAINS = [
    "制造业预测性维护",
    "零售业库存优化",
    "金融风控建模",
    "医疗影像辅助诊断",
    "物流路径规划",
    "教育个性化推荐",
    "能源负荷预测",
    "智能客服问答",
    "供应链中断预警",
    "智能招聘筛选",
]

# Each domain yields the same fixed set of task templates so the dataset is
# fully deterministic and reproducible across runs (no random seed needed).
CATEGORY = Literal["normal", "complex", "revise", "no_evidence"]


@dataclass
class EvalTask:
    task_id: str
    goal: str
    category: CATEGORY
    domain: str

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "category": self.category,
            "domain": self.domain,
        }


def _templates_for(domain: str) -> list[tuple[CATEGORY, str]]:
    """Five normal, three complex, one revise, one no_evidence per domain."""
    return [
        ("normal", f"为{domain}场景设计企业实施方案"),
        ("normal", f"面向中小制造企业的{domain}落地方案"),
        ("normal", f"为{domain}撰写可交付的客户方案初稿"),
        ("normal", f"围绕{domain}做一份内部可行性分析"),
        ("normal", f"为{domain}项目制定三个月推进计划"),
        ("complex", f"为{domain}客户设计包含成本测算与风险预案的详细方案"),
        ("complex", f"面向集团客户的{domain}整体规划与实施路线图"),
        ("complex", f"为{domain}设计含 KPI 与验收标准的完整提案"),
        ("revise", f"需要修订的{domain}方案"),
        ("no_evidence", f"__no_evidence__关于{domain}的机密内部资料方案"),
    ]


def build_evaluation_set() -> list[EvalTask]:
    """Build a deterministic evaluation set of 100 tasks (10 domains x 10)."""
    tasks: list[EvalTask] = []
    index = 0
    for domain in DOMAINS:
        for category, goal in _templates_for(domain):
            index += 1
            tasks.append(
                EvalTask(
                    task_id=f"task-{index:03d}",
                    goal=goal,
                    category=category,
                    domain=domain,
                )
            )
    return tasks


def save_dataset(path: str | Path, tasks: list[EvalTask] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = tasks or build_evaluation_set()
    description = (
        "确定性合成评测集：10 个业务域 × 10 条任务"
        "（5 normal / 3 complex / 1 revise / 1 no_evidence）"
    )
    payload = {
        "size": len(tasks),
        "description": description,
        "tasks": [t.to_dict() for t in tasks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_dataset(path: str | Path) -> list[EvalTask]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalTask(
            task_id=t["task_id"],
            goal=t["goal"],
            category=t["category"],
            domain=t["domain"],
        )
        for t in raw["tasks"]
    ]
