"""
SRE-Gym: Core OpenEnv-compliant environment for SRE bug triage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from src.models import (
    Action,
    BugReport,
    Observation,
    PublicBugReport,
    ResetResult,
    StepResult,
)

# ── constants ───────────────────────────────────────────────────────────────

_URGENCY_WEIGHT: Dict[str, float] = {
    "CRITICAL": 2.0,
    "HIGH": 1.5,
    "MEDIUM": 1.0,
    "LOW": 0.5,
}

_INCIDENTS_DIR = Path(__file__).parent / "incidents"


# ── environment ─────────────────────────────────────────────────────────────


class SREGymEnvironment:
    """OpenEnv-compliant SRE bug triage environment.

    Lifecycle::

        env = SREGymEnvironment("task1_easy")
        result = env.reset()
        while not result.observation.done:
            action = agent.act(result.observation)
            result = env.step(action)
    """

    # ── constructor ──────────────────────────────────────────────────────

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._task_path = _INCIDENTS_DIR / f"{task_id}.json"
        if not self._task_path.exists():
            raise FileNotFoundError(
                f"Task file not found: {self._task_path}"
            )
        self._raw_task: dict = json.loads(
            self._task_path.read_text(encoding="utf-8")
        )
        self._initialize_state()

    # ── public API ───────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reload the task JSON and reinitialise all internal state."""
        self._raw_task = json.loads(
            self._task_path.read_text(encoding="utf-8")
        )
        self._initialize_state()
        return ResetResult(
            observation=self._build_observation(),
            info={"task_id": self._task_id, "event": "reset"},
        )

    def step(self, action: Action) -> StepResult:
        """Process one agent action and advance the environment by one tick.

        Returns a :class:`StepResult` containing the new observation,
        scalar reward for this step, a ``done`` flag, and an ``info`` dict.
        """
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already finished."},
            )

        # ---- 1. execute action ----
        reward, result_msg = self._process_action(action)

        # ---- 2. post-step bookkeeping ----
        #   a) return temporarily-fixed bugs whose root cause is still active
        self._return_temporary_fixes()

        #   b) apply spread to all unfixed non-red-herring bugs
        self._apply_spread()

        #   c) advance step counter
        self._step_number += 1

        #   d) check terminal conditions
        self._check_done()

        #   e) add final bonus when episode ends
        if self._done:
            reward += self._calculate_final_bonus()

        # ---- 3. clip reward to [-2.0, 5.0] ----
        reward = max(-2.0, min(5.0, reward))

        # ---- 4. accumulate total reward ----
        self._total_reward += reward
        self._last_action_result = result_msg

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=self._done,
            info={
                "action": action.model_dump(),
                "result": result_msg,
                "total_reward": round(self._total_reward, 4),
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return the full current state as a plain dict (debugging / logging)."""
        return {
            "task_id": self._task_id,
            "step_number": self._step_number,
            "budget_remaining": self._budget_remaining,
            "max_steps": self._max_steps,
            "done": self._done,
            "total_reward": round(self._total_reward, 4),
            "fixed_bugs": sorted(self._fixed_bugs),
            "investigated_bugs": sorted(self._investigated_bugs),
            "returning_bugs": sorted(self._returning_bugs),
            "bugs": {
                bid: pub.model_dump()
                for bid, pub in self._public_bugs.items()
            },
        }

    # ── initialisation ───────────────────────────────────────────────────

    def _initialize_state(self) -> None:
        """(Re)set every mutable field from the raw task dict."""
        task = self._raw_task

        self._goal: str = task["goal"]
        self._max_steps: int = task["max_steps"]
        self._budget_remaining: int = task["budget"]
        self._step_number: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._last_action_result: str = ""

        self._bugs: Dict[str, BugReport] = {}
        self._public_bugs: Dict[str, PublicBugReport] = {}
        self._fixed_bugs: Set[str] = set()
        self._investigated_bugs: Set[str] = set()
        self._returning_bugs: Set[str] = set()

        for raw in task["bugs"]:
            bug = BugReport(
                bug_id=raw["bug_id"],
                error_message=raw["error_message"],
                severity=raw["severity"],
                frequency=raw["frequency"],
                affected_users=raw["affected_users"],
                stack_trace=raw["stack_trace"],
                service=raw["service"],
                is_red_herring=raw["is_red_herring"],
                root_cause_bug_id=raw.get("root_cause_bug_id"),
                child_bug_ids=raw.get("child_bug_ids", []),
                spread_rate=raw.get("spread_rate", 0.0),
            )
            self._bugs[bug.bug_id] = bug

            self._public_bugs[bug.bug_id] = PublicBugReport(
                bug_id=bug.bug_id,
                error_message=bug.error_message,
                severity=bug.severity,
                frequency=bug.frequency,
                affected_users=bug.affected_users,
                service=bug.service,
                investigated=False,
                fixed=False,
                stack_trace=None,
            )

    # ── action dispatch ──────────────────────────────────────────────────

    def _process_action(self, action: Action) -> Tuple[float, str]:
        """Route *action* to the right handler; return ``(reward, message)``."""
        handlers = {
            "investigate": self._handle_investigate,
            "fix": self._handle_fix,
            "escalate": self._handle_escalate,
            "ignore": self._handle_ignore,
            "noop": self._handle_noop,
        }

        handler = handlers.get(action.action_type)
        if handler is None:
            return -0.1, f"Unknown action type: {action.action_type}"

        # Actions that require a valid bug_id
        if action.action_type in ("investigate", "fix", "escalate"):
            if action.bug_id is None:
                return -0.1, f"Action '{action.action_type}' requires a bug_id."
            if action.bug_id not in self._bugs:
                return -0.1, f"Invalid bug_id: {action.bug_id}"

        return handler(action)

    # ── action handlers ──────────────────────────────────────────────────

    def _handle_investigate(self, action: Action) -> Tuple[float, str]:
        """Reveal the stack trace for a bug.  **Cost: 1 budget.**"""
        bug_id: str = action.bug_id  # type: ignore[assignment]
        cost = 1

        if self._budget_remaining < cost:
            return -0.05, "Not enough budget to investigate."
        self._budget_remaining -= cost

        if bug_id in self._investigated_bugs:
            return -0.05, f"Already investigated {bug_id}."

        # Reveal
        self._investigated_bugs.add(bug_id)
        full = self._bugs[bug_id]
        pub = self._public_bugs[bug_id]
        pub.investigated = True
        pub.stack_trace = full.stack_trace

        preview = full.stack_trace[:100].replace("\n", " ")
        return 0.05, f"Investigated {bug_id}: {preview}..."

    def _handle_fix(self, action: Action) -> Tuple[float, str]:
        """Attempt to fix a bug.  **Cost: 2 budget.**"""
        bug_id: str = action.bug_id  # type: ignore[assignment]
        cost = 2

        if self._budget_remaining < cost:
            return -0.05, "Not enough budget to fix."
        self._budget_remaining -= cost

        full = self._bugs[bug_id]
        pub = self._public_bugs[bug_id]

        # ---- already fixed ----
        if bug_id in self._fixed_bugs:
            return -0.1, f"{bug_id} is already fixed."

        # ---- red herring — wasted effort ----
        if full.is_red_herring:
            self._fixed_bugs.add(bug_id)
            pub.fixed = True
            return -0.2, (
                f"Fixed {bug_id} — but this was a red herring. "
                "Budget wasted on noise."
            )

        # ---- symptom with unfixed root cause ----
        root_id = full.root_cause_bug_id
        if root_id is not None and root_id not in self._fixed_bugs:
            self._fixed_bugs.add(bug_id)
            pub.fixed = True
            self._returning_bugs.add(bug_id)
            return 0.1, (
                f"Fixed symptom {bug_id} but root cause "
                f"{root_id} still active — bug will return."
            )

        # ---- genuine fix (root cause OR root already resolved) ----
        self._fixed_bugs.add(bug_id)
        pub.fixed = True

        # cascade: automatically fix all children
        self._fix_cascade(bug_id)

        weight = _URGENCY_WEIGHT.get(full.severity, 1.0)
        reward = (full.affected_users / 1000.0) * weight
        return reward, (
            f"Fixed {bug_id} — {full.affected_users} users unblocked."
        )

    def _handle_escalate(self, action: Action) -> Tuple[float, str]:
        """Escalate to the on-call team.  **Cost: 1 budget.**"""
        bug_id: str = action.bug_id  # type: ignore[assignment]
        cost = 1

        if self._budget_remaining < cost:
            return -0.05, "Not enough budget to escalate."
        self._budget_remaining -= cost

        full = self._bugs[bug_id]
        if full.severity == "CRITICAL" and bug_id not in self._fixed_bugs:
            return 0.15, f"Escalated {bug_id} to on-call team."
        return -0.1, f"Escalated {bug_id} to on-call team (unnecessary)."

    def _handle_ignore(self, action: Action) -> Tuple[float, str]:
        """Deliberately ignore a bug.  **Cost: 0 budget.**"""
        bug_id = action.bug_id
        if bug_id is None:
            return -0.1, "ignore action requires a bug_id."
        if bug_id not in self._bugs:
            return -0.1, f"Invalid bug_id: {bug_id}"

        full = self._bugs[bug_id]
        if full.is_red_herring:
            return 0.1, f"Ignored {bug_id} — correct, this was noise."
        return -0.15, f"Ignored {bug_id} — but this was a real bug!"

    def _handle_noop(self, _action: Action) -> Tuple[float, str]:
        """Do nothing.  **Cost: 0 budget.**"""
        return -0.01, "No operation."

    # ── cascade helpers ──────────────────────────────────────────────────

    def _fix_cascade(self, bug_id: str) -> None:
        """Recursively mark all child bugs as fixed."""
        full = self._bugs[bug_id]
        for child_id in full.child_bug_ids:
            if child_id in self._bugs and child_id not in self._fixed_bugs:
                self._fixed_bugs.add(child_id)
                self._public_bugs[child_id].fixed = True
                # If the child was scheduled to return, cancel that
                self._returning_bugs.discard(child_id)
                # Continue down the tree
                self._fix_cascade(child_id)

    def _return_temporary_fixes(self) -> None:
        """Un-fix every bug whose root cause is still active."""
        for bug_id in list(self._returning_bugs):
            root_id = self._bugs[bug_id].root_cause_bug_id
            if root_id is not None and root_id not in self._fixed_bugs:
                self._fixed_bugs.discard(bug_id)
                self._public_bugs[bug_id].fixed = False
        self._returning_bugs.clear()

    # ── spread / tick ────────────────────────────────────────────────────

    def _apply_spread(self) -> None:
        """For every unfixed real bug, grow ``affected_users`` by ``spread_rate``."""
        for bug_id, bug in self._bugs.items():
            if bug_id in self._fixed_bugs:
                continue
            if bug.is_red_herring:
                continue
            if bug.spread_rate > 0:
                added = int(bug.spread_rate)
                bug.affected_users += added
                self._public_bugs[bug_id].affected_users = bug.affected_users

    # ── terminal conditions ──────────────────────────────────────────────

    def _check_done(self) -> None:
        """Set ``self._done`` if any terminal condition is met."""
        # Budget exhausted
        if self._budget_remaining <= 0:
            self._done = True
            return

        # Max steps reached
        if self._step_number >= self._max_steps:
            self._done = True
            return

        # All non-red-herring bugs fixed
        real_bug_ids = {
            bid for bid, b in self._bugs.items() if not b.is_red_herring
        }
        if real_bug_ids.issubset(self._fixed_bugs):
            self._done = True

    # ── observation builder ──────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """Assemble the current :class:`Observation` from internal state."""
        return Observation(
            bugs=list(self._public_bugs.values()),
            step_number=self._step_number,
            budget_remaining=self._budget_remaining,
            total_affected_users=sum(
                pub.affected_users
                for pub in self._public_bugs.values()
                if not pub.fixed
            ),
            goal=self._goal,
            last_action_result=self._last_action_result,
            task_id=self._task_id,
            done=self._done,
        )

    # ── final scoring ────────────────────────────────────────────────────

    def _calculate_final_bonus(self) -> float:
        """Compute end-of-episode bonus points.

        * **+0.5** if every non-red-herring CRITICAL bug is fixed.
        * **+0.2** if no budget was wasted fixing red herrings.
        * **+0.05 × remaining budget** for efficiency.
        """
        bonus = 0.0

        # All real critical bugs fixed?
        critical_real = {
            bid
            for bid, b in self._bugs.items()
            if b.severity == "CRITICAL" and not b.is_red_herring
        }
        if critical_real and critical_real.issubset(self._fixed_bugs):
            bonus += 0.5

        # No red herrings wasted on?
        red_herring_ids = {
            bid for bid, b in self._bugs.items() if b.is_red_herring
        }
        if not red_herring_ids.intersection(self._fixed_bugs):
            bonus += 0.2

        # Remaining budget efficiency
        bonus += self._budget_remaining * 0.05

        return bonus
