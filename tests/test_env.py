"""
SRE-Gym smoke tests.

Run with:
    PYTHONPATH=. pytest tests/test_env.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import SREGymEnvironment
from src.graders import run_grader
from src.models import Action


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def env_easy() -> SREGymEnvironment:
    return SREGymEnvironment("task1_easy")


@pytest.fixture
def env_medium() -> SREGymEnvironment:
    return SREGymEnvironment("task2_medium")


@pytest.fixture
def env_hard() -> SREGymEnvironment:
    return SREGymEnvironment("task3_hard")


# ── Task 1: basics ──────────────────────────────────────────────────────────


class TestResetTask1:
    def test_reset_returns_correct_observation(self, env_easy: SREGymEnvironment):
        result = env_easy.reset()
        obs = result.observation

        assert len(obs.bugs) == 5
        assert obs.budget_remaining == 8
        assert obs.step_number == 0
        assert obs.done is False
        assert obs.task_id == "task1_easy"

    def test_all_bugs_start_uninvestigated(self, env_easy: SREGymEnvironment):
        result = env_easy.reset()
        for bug in result.observation.bugs:
            assert bug.investigated is False
            assert bug.fixed is False
            assert bug.stack_trace is None


class TestInvestigateAction:
    def test_investigate_reveals_stack_trace(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        action = Action(action_type="investigate", bug_id="BUG001")
        result = env_easy.step(action)

        assert result.reward > 0
        # Find BUG001 in observation
        bug = next(b for b in result.observation.bugs if b.bug_id == "BUG001")
        assert bug.investigated is True
        assert bug.stack_trace is not None
        assert len(bug.stack_trace) > 0

    def test_double_investigate_penalised(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        action = Action(action_type="investigate", bug_id="BUG001")
        env_easy.step(action)
        result2 = env_easy.step(action)

        assert result2.reward < 0  # penalty for redundant investigation


class TestFixAction:
    def test_fix_marks_bug_as_fixed(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        # Investigate first
        env_easy.step(Action(action_type="investigate", bug_id="BUG001"))
        # Fix
        result = env_easy.step(
            Action(action_type="fix", bug_id="BUG001", fix_strategy="hotfix")
        )

        assert result.reward > 0
        bug = next(b for b in result.observation.bugs if b.bug_id == "BUG001")
        assert bug.fixed is True

    def test_fix_without_investigate_still_works(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        result = env_easy.step(
            Action(action_type="fix", bug_id="BUG001", fix_strategy="rollback")
        )
        bug = next(b for b in result.observation.bugs if b.bug_id == "BUG001")
        assert bug.fixed is True


class TestBudgetDepletion:
    def test_budget_decreases_on_actions(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        # 4 fix actions × 2 budget each = 8 budget (full depletion)
        bugs = ["BUG001", "BUG002", "BUG003", "BUG004"]
        for bug_id in bugs:
            result = env_easy.step(
                Action(action_type="fix", bug_id=bug_id, fix_strategy="hotfix")
            )

        assert result.observation.budget_remaining == 0
        assert result.done is True  # budget exhausted


# ── Task 2: red herrings ────────────────────────────────────────────────────


class TestRedHerringPenalty:
    def test_fixing_red_herring_gives_negative_reward(
        self, env_medium: SREGymEnvironment
    ):
        env_medium.reset()

        # BUG_M004 is a red herring (CRITICAL, search-service, CI test index)
        result = env_medium.step(
            Action(action_type="fix", bug_id="BUG_M004", fix_strategy="hotfix")
        )

        assert result.reward < 0  # -0.2 penalty for red herring

    def test_ignoring_red_herring_gives_positive_reward(
        self, env_medium: SREGymEnvironment
    ):
        env_medium.reset()

        result = env_medium.step(
            Action(action_type="ignore", bug_id="BUG_M004")
        )

        assert result.reward > 0  # +0.1 for correctly ignoring noise


# ── Task 3: cascading failures ──────────────────────────────────────────────


class TestCascadeReturns:
    def test_symptom_fix_returns_when_root_unfixed(
        self, env_hard: SREGymEnvironment
    ):
        env_hard.reset()

        # BUG_H002 is a symptom of BUG_H001
        # Fix the symptom WITHOUT fixing the root cause
        result = env_hard.step(
            Action(action_type="fix", bug_id="BUG_H002", fix_strategy="hotfix")
        )

        # Bug should temporarily be fixed
        assert "root cause" in result.info["result"].lower() or \
               "will return" in result.info["result"].lower()

        # After the step, the bug should return (un-fix itself)
        bug_h002 = next(
            b for b in result.observation.bugs if b.bug_id == "BUG_H002"
        )
        assert bug_h002.fixed is False  # returned because root unfixed

    def test_root_cause_fix_cascades_to_children(
        self, env_hard: SREGymEnvironment
    ):
        env_hard.reset()

        # Fix the root cause BUG_H001
        result = env_hard.step(
            Action(action_type="fix", bug_id="BUG_H001", fix_strategy="hotfix")
        )

        # BUG_H001's child BUG_H002 should also be fixed (cascade)
        bug_h002 = next(
            b for b in result.observation.bugs if b.bug_id == "BUG_H002"
        )
        assert bug_h002.fixed is True


# ── state() ─────────────────────────────────────────────────────────────────


class TestStateEndpoint:
    def test_state_returns_required_keys(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        state = env_easy.state()

        required_keys = {
            "task_id",
            "step_number",
            "budget_remaining",
            "max_steps",
            "done",
            "total_reward",
            "fixed_bugs",
            "investigated_bugs",
            "returning_bugs",
            "bugs",
        }
        assert required_keys.issubset(state.keys())

    def test_state_reflects_actions(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        env_easy.step(Action(action_type="investigate", bug_id="BUG001"))
        state = env_easy.state()

        assert "BUG001" in state["investigated_bugs"]
        assert state["step_number"] == 1


# ── grader ───────────────────────────────────────────────────────────────────


class TestGrader:
    def test_grader_returns_valid_score_task1(self, env_easy: SREGymEnvironment):
        env_easy.reset()
        # Play a simple episode: investigate and fix top 2 bugs
        for bug_id in ["BUG001", "BUG002"]:
            env_easy.step(Action(action_type="investigate", bug_id=bug_id))
            env_easy.step(
                Action(action_type="fix", bug_id=bug_id, fix_strategy="hotfix")
            )

        state = env_easy.state()
        score = run_grader("task1_easy", state)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grader_returns_valid_score_task2(self, env_medium: SREGymEnvironment):
        env_medium.reset()
        # Minimal episode — just investigate one bug
        env_medium.step(Action(action_type="investigate", bug_id="BUG_M001"))

        state = env_medium.state()
        score = run_grader("task2_medium", state)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grader_returns_valid_score_task3(self, env_hard: SREGymEnvironment):
        env_hard.reset()
        # Minimal episode
        env_hard.step(Action(action_type="noop"))

        state = env_hard.state()
        score = run_grader("task3_hard", state)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_play_scores_higher(self, env_easy: SREGymEnvironment):
        """An agent that fixes all bugs should score higher than one that does nothing."""
        # Episode A: fix top bugs
        env_easy.reset()
        for bug_id in ["BUG001", "BUG002", "BUG003"]:
            env_easy.step(
                Action(action_type="fix", bug_id=bug_id, fix_strategy="hotfix")
            )
        state_a = env_easy.state()
        score_a = run_grader("task1_easy", state_a)

        # Episode B: do nothing
        env_easy.reset()
        env_easy.step(Action(action_type="noop"))
        state_b = env_easy.state()
        score_b = run_grader("task1_easy", state_b)

        assert score_a > score_b


# ── edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_bug_id_returns_negative_reward(
        self, env_easy: SREGymEnvironment
    ):
        env_easy.reset()
        result = env_easy.step(
            Action(action_type="investigate", bug_id="NONEXISTENT")
        )
        assert result.reward < 0

    def test_step_after_done_returns_zero_reward(
        self, env_easy: SREGymEnvironment
    ):
        env_easy.reset()
        # Burn all budget: 4 fixes × 2 = 8
        for bug_id in ["BUG001", "BUG002", "BUG003", "BUG004"]:
            env_easy.step(
                Action(action_type="fix", bug_id=bug_id, fix_strategy="hotfix")
            )
        # Episode should be done now
        result = env_easy.step(Action(action_type="noop"))
        assert result.done is True
        assert result.reward == 0.0

    def test_noop_gives_small_negative_reward(
        self, env_easy: SREGymEnvironment
    ):
        env_easy.reset()
        result = env_easy.step(Action(action_type="noop"))
        assert result.reward < 0

    def test_spread_increases_affected_users(
        self, env_easy: SREGymEnvironment
    ):
        env_easy.reset()
        # Get initial affected users for BUG001
        initial_bug = next(
            b
            for b in env_easy.reset().observation.bugs
            if b.bug_id == "BUG001"
        )
        initial_users = initial_bug.affected_users

        # Take a noop — BUG001 has spread_rate 50.0
        result = env_easy.step(Action(action_type="noop"))
        bug_after = next(
            b for b in result.observation.bugs if b.bug_id == "BUG001"
        )

        assert bug_after.affected_users > initial_users
