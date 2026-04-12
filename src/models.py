"""
SRE-Gym: OpenEnv-compliant Pydantic v2 models for bug triage environment.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class BugReport(BaseModel):
    """Internal ground-truth representation of a bug (hidden from the agent)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bug_id: str
    error_message: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    frequency: int  # how many times the alert fired
    affected_users: int
    stack_trace: str  # hidden until the bug is investigated
    service: str  # e.g. "payment-service"
    is_red_herring: bool  # hidden field – true if the bug is not actionable
    root_cause_bug_id: Optional[str] = None  # parent in the cascade graph
    child_bug_ids: List[str] = []  # bugs this one masks
    spread_rate: float = 0.0  # additional users affected per step


class PublicBugReport(BaseModel):
    """Agent-visible view of a bug report (sanitised)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bug_id: str
    error_message: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    frequency: int
    affected_users: int
    service: str
    investigated: bool = False
    fixed: bool = False
    stack_trace: Optional[str] = None  # only revealed after investigate()


class Action(BaseModel):
    """An action the agent can take each step."""

    action_type: Literal["investigate", "fix", "escalate", "ignore", "noop"]
    bug_id: Optional[str] = None
    fix_strategy: Optional[Literal["hotfix", "rollback", "restart", "patch"]] = None


class Observation(BaseModel):
    """Everything the agent observes after each step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bugs: List[PublicBugReport]
    step_number: int
    budget_remaining: int
    total_affected_users: int
    goal: str
    last_action_result: str = ""
    task_id: str
    done: bool = False


class StepResult(BaseModel):
    """Returned by ``env.step()``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class ResetResult(BaseModel):
    """Returned by ``env.reset()``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Observation
    info: dict[str, Any]
