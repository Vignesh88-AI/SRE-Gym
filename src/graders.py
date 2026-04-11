"""
SRE-Gym graders — score a completed episode on a 0.0–1.0 scale.

Each grader loads the ground-truth task JSON for reference data
(red-herring flags, cascade graph, initial user counts) and
evaluates the agent's ``episode_state`` dict (produced by
``SREGymEnvironment.state()``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

_INCIDENTS_DIR = Path(__file__).parent / "incidents"


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_task(task_id: str) -> dict:
    """Load the ground-truth task JSON."""
    path = _INCIDENTS_DIR / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _clamp(value: float) -> float:
    """Clamp a score to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


# ── graders ──────────────────────────────────────────────────────────────────


def grade_task1(episode_state: Dict[str, Any]) -> float:
    """Task 1 grader — did the agent fix the highest-impact bugs?

    Score components
    ~~~~~~~~~~~~~~~~
    * **60 %** — Were the top 3 bugs by ``affected_users`` fixed? (20 % each)
    * **20 %** — Budget efficiency (``budget_remaining / initial_budget × 0.2``)
    * **20 %** — No unnecessary actions (deduct 0.05 per wasteful action)

    Returns a float clamped to ``[0.0, 1.0]``.
    """
    task = _load_task("task1_easy")

    # ── ground truth ──
    bugs_by_impact = sorted(
        task["bugs"], key=lambda b: b["affected_users"], reverse=True
    )
    top_3_ids: Set[str] = {b["bug_id"] for b in bugs_by_impact[:3]}
    initial_budget: int = task["budget"]

    # ── episode results ──
    fixed: Set[str] = set(episode_state["fixed_bugs"])
    investigated: Set[str] = set(episode_state["investigated_bugs"])
    budget_remaining: int = episode_state["budget_remaining"]
    step_number: int = episode_state["step_number"]

    # ── 60 %: top-3 impact bugs fixed ──
    impact_score = sum(0.2 for bid in top_3_ids if bid in fixed)

    # ── 20 %: budget efficiency ──
    if initial_budget > 0:
        efficiency_score = (budget_remaining / initial_budget) * 0.2
    else:
        efficiency_score = 0.0

    # ── 20 %: wasteful-action penalty ──
    # Useful actions = investigations + fixes.  Everything else is waste.
    useful_actions = len(investigated) + len(fixed)
    wasteful_actions = max(0, step_number - useful_actions)
    waste_penalty = min(0.2, wasteful_actions * 0.05)
    waste_score = 0.2 - waste_penalty

    return _clamp(impact_score + efficiency_score + waste_score)


def grade_task2(episode_state: Dict[str, Any]) -> float:
    """Task 2 grader — did the agent avoid red herrings and find real issues?

    Score components
    ~~~~~~~~~~~~~~~~
    * **40 %** — Real high-impact bugs fixed
      (non-red-herring with ``affected_users > 100``)
    * **30 %** — Red herrings *not* fixed (−0.10 per red herring fixed)
    * **20 %** — Root cause handled correctly in the cascade
    * **10 %** — Investigate-before-fix discipline for ambiguous bugs

    Returns a float clamped to ``[0.0, 1.0]``.
    """
    task = _load_task("task2_medium")

    # ── ground truth ──
    bug_map: Dict[str, dict] = {b["bug_id"]: b for b in task["bugs"]}
    red_herrings: Set[str] = {
        bid for bid, b in bug_map.items() if b["is_red_herring"]
    }
    real_high_impact: Set[str] = {
        bid
        for bid, b in bug_map.items()
        if not b["is_red_herring"] and b["affected_users"] > 100
    }
    cascade_children: Set[str] = {
        bid
        for bid, b in bug_map.items()
        if b.get("root_cause_bug_id") is not None
    }

    # ── episode results ──
    fixed: Set[str] = set(episode_state["fixed_bugs"])
    investigated: Set[str] = set(episode_state["investigated_bugs"])

    # ── 40 %: real high-impact bugs fixed ──
    if real_high_impact:
        real_fixed_ratio = len(fixed & real_high_impact) / len(real_high_impact)
    else:
        real_fixed_ratio = 1.0
    impact_score = real_fixed_ratio * 0.4

    # ── 30 %: red herrings NOT fixed ──
    rh_fixed_count = len(fixed & red_herrings)
    rh_score = max(0.0, 0.3 - rh_fixed_count * 0.1)

    # ── 20 %: cascade root-cause discipline ──
    # Full marks if every fixed symptom also has its root cause fixed.
    # Deduct 0.10 for each symptom fixed without its root.
    cascade_score = 0.2
    for bid in cascade_children:
        root_id = bug_map[bid].get("root_cause_bug_id")
        if root_id and bid in fixed and root_id not in fixed:
            cascade_score -= 0.10
    cascade_score = max(0.0, cascade_score)

    # ── 10 %: investigate-before-fix ──
    # For every non-red-herring bug that was fixed, check whether
    # it was investigated first.  Ratio → 10 %.
    fixable = fixed - red_herrings
    if fixable:
        invest_ratio = len(fixable & investigated) / len(fixable)
    else:
        invest_ratio = 1.0
    invest_score = invest_ratio * 0.1

    return _clamp(impact_score + rh_score + cascade_score + invest_score)


def grade_task3(episode_state: Dict[str, Any]) -> float:
    """Task 3 grader — did the agent solve cascading failures optimally?

    Score components
    ~~~~~~~~~~~~~~~~
    * **50 %** — Root-cause bugs fixed
      (``BUG_H001``, ``BUG_H010``, ``BUG_H015`` — ~16.7 % each)
    * **20 %** — Cascade chains fully resolved (~6.7 % per chain)
    * **20 %** — Users unblocked as a fraction of total initial users at risk
    * **10 %** — Budget efficiency

    Penalties
    ~~~~~~~~~
    * **−0.15** per symptom fixed while its root cause remained unfixed
    * **−0.10** per red herring fixed

    Returns a float clamped to ``[0.0, 1.0]``.
    """
    task = _load_task("task3_hard")

    # ── ground truth ──
    bug_map: Dict[str, dict] = {b["bug_id"]: b for b in task["bugs"]}
    root_causes: Set[str] = {"BUG_H001", "BUG_H010", "BUG_H015"}
    chains: Dict[str, list] = {
        "database": ["BUG_H001", "BUG_H002", "BUG_H003", "BUG_H004"],
        "auth": ["BUG_H010", "BUG_H011", "BUG_H012"],
        "pipeline": ["BUG_H015", "BUG_H016"],
    }
    red_herrings: Set[str] = {
        bid for bid, b in bug_map.items() if b["is_red_herring"]
    }
    cascade_children: Set[str] = {
        bid
        for bid, b in bug_map.items()
        if b.get("root_cause_bug_id") is not None
    }

    # Total initial users at risk across all non-red-herring bugs
    total_initial_users = sum(
        b["affected_users"]
        for b in task["bugs"]
        if not b["is_red_herring"]
    )

    initial_budget: int = task["budget"]

    # ── episode results ──
    fixed: Set[str] = set(episode_state["fixed_bugs"])
    budget_remaining: int = episode_state["budget_remaining"]
    bugs_state: Dict[str, dict] = episode_state["bugs"]

    # ── 50 %: root causes fixed ──
    roots_fixed = len(fixed & root_causes)
    root_score = (roots_fixed / len(root_causes)) * 0.5

    # ── 20 %: cascade chains fully resolved ──
    chain_score = 0.0
    per_chain = 0.2 / len(chains)
    for _chain_name, chain_bugs in chains.items():
        if set(chain_bugs).issubset(fixed):
            chain_score += per_chain

    # ── 20 %: users unblocked ──
    # Sum initial affected_users for every fixed, non-red-herring bug.
    users_saved = sum(
        bug_map[bid]["affected_users"]
        for bid in fixed
        if bid in bug_map and not bug_map[bid]["is_red_herring"]
    )
    if total_initial_users > 0:
        unblock_ratio = min(1.0, users_saved / total_initial_users)
    else:
        unblock_ratio = 1.0
    unblock_score = unblock_ratio * 0.2

    # ── 10 %: budget efficiency ──
    if initial_budget > 0:
        efficiency_score = (budget_remaining / initial_budget) * 0.1
    else:
        efficiency_score = 0.0

    score = root_score + chain_score + unblock_score + efficiency_score

    # ── penalties ──

    # −0.15 per symptom fixed while its root cause remained unfixed
    for bid in cascade_children:
        root_id = bug_map[bid].get("root_cause_bug_id")
        if root_id and bid in fixed and root_id not in fixed:
            score -= 0.15

    # −0.10 per red herring fixed
    rh_fixed_count = len(fixed & red_herrings)
    score -= rh_fixed_count * 0.1

    return _clamp(score)


# ── router ───────────────────────────────────────────────────────────────────


def run_grader(task_id: str, episode_state: Dict[str, Any]) -> float:
    """Route to the correct grader function by *task_id*.

    Raises :class:`ValueError` for an unrecognised task.
    """
    graders = {
        "task1_easy": grade_task1,
        "task2_medium": grade_task2,
        "task3_hard": grade_task3,
    }
    grader = graders.get(task_id)
    if grader is None:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. "
            f"Available: {sorted(graders)}"
        )
    return grader(episode_state)
