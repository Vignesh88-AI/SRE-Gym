"""
SRE-Gym baseline inference script.

Runs an LLM agent (via OpenAI-compatible API) against all three
SRE-Gym tasks and reports scores.

Mandatory environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``HF_TOKEN``    — Your Hugging Face / API key (no default)
* ``API_BASE_URL`` — The API endpoint for the LLM
* ``MODEL_NAME``  — The model identifier to use for inference

Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

Optional:
* ``ENV_URL`` — SRE-Gym server URL (default: https://argonite3-sre-gym.hf.space)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── configuration ────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: Optional[str] = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
API_KEY: Optional[str] = HF_TOKEN or os.getenv("API_KEY")
ENV_URL: str = os.getenv("ENV_URL", "https://argonite3-sre-gym.hf.space")

MAX_STEPS: int = 15
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 400

# ── system prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = (
    "You are an expert SRE (Site Reliability Engineer) triaging production incidents.\n"
    "You receive a live incident queue with bugs, severities, affected user counts, and services.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Bugs have CASCADE relationships — some are SYMPTOMS of a root cause.\n"
    "   Fixing a symptom before its root cause = bug RETURNS next step (wasted budget!).\n"
    "   Always investigate() suspicious bugs first to reveal stack trace and root cause.\n"
    "2. Some CRITICAL bugs are RED HERRINGS (flaky tests, noise). Do NOT fix them.\n"
    "   investigate() to reveal the truth, then ignore() them for +0.1 reward.\n"
    "3. Some LOW severity bugs silently corrupt real user data — higher priority than they look.\n"
    "4. Fixed budget: investigate() costs 1, fix() costs 2. Plan carefully.\n"
    "5. Unfixed bugs spread every step (affected_users grows). Speed matters.\n"
    "\n"
    "STRATEGY:\n"
    "- Scan for suspiciously high-frequency CRITICAL bugs (red herring candidates).\n"
    "- investigate() ambiguous bugs before committing budget to fix().\n"
    "- Fix ROOT CAUSES first. Check last_action_result for cascade warnings.\n"
    "- ignore() confirmed noise to save budget and earn +0.1.\n"
    "- Fix real bugs by (affected_users x urgency) priority.\n"
    "\n"
    "Available actions (JSON only — no explanation, no markdown):\n"
    '{"action_type": "investigate", "bug_id": "BUG001"}\n'
    '{"action_type": "fix", "bug_id": "BUG001", "fix_strategy": "hotfix"}\n'
    '{"action_type": "ignore", "bug_id": "BUG001"}\n'
    '{"action_type": "escalate", "bug_id": "BUG001"}\n'
    '{"action_type": "noop"}\n'
    "\n"
    "fix_strategy options: hotfix, rollback, restart, patch\n"
    "Reply with ONLY a single JSON object. No explanation. No markdown."
)


# ── logging helpers ──────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── OpenAI client ────────────────────────────────────────────────────────────


def _get_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required. Set it as an environment variable."
        )
    if not MODEL_NAME:
        raise RuntimeError(
            "MODEL_NAME is required. Set it as an environment variable."
        )
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── env helpers ──────────────────────────────────────────────────────────────


def call_env(
    endpoint: str,
    method: str = "GET",
    data: Optional[dict] = None,
) -> dict:
    """Call the SRE-Gym server and return the JSON response."""
    url = f"{ENV_URL}{endpoint}"
    if method.upper() == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=data or {}, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(
            f"ENV error {resp.status_code} on {method} {endpoint}: {resp.text}"
        )
    return resp.json()


# ── prompt builder ───────────────────────────────────────────────────────────


def build_user_prompt(obs: dict) -> str:
    """Format an observation dict into a human-readable prompt for the LLM."""
    lines: List[str] = [
        "=== SRE Incident Queue ===",
        f"Step: {obs['step_number']}  |  Budget remaining: {obs['budget_remaining']}",
        f"Total affected users: {obs['total_affected_users']}",
        f"Goal: {obs['goal']}",
    ]

    if obs.get("last_action_result"):
        lines.append(f"Last action result: {obs['last_action_result']}")

    lines.append("")
    lines.append("--- Active Bugs ---")

    for bug in obs["bugs"]:
        if bug.get("fixed"):
            continue
        parts = [
            f"  [{bug['severity']}] {bug['bug_id']}",
            f"    Service: {bug['service']}",
            f"    Error: {bug['error_message'][:120]}",
            f"    Affected users: {bug['affected_users']}  |  Frequency: {bug['frequency']}",
            f"    Investigated: {bug.get('investigated', False)}",
        ]
        stack = bug.get("stack_trace")
        if stack:
            preview = stack[:300].replace("\n", "\n      ")
            parts.append(f"    Stack trace:\n      {preview}...")
        lines.extend(parts)
        lines.append("")

    fixed = [b for b in obs["bugs"] if b.get("fixed")]
    if fixed:
        lines.append(f"--- Fixed Bugs ({len(fixed)}) ---")
        for bug in fixed:
            lines.append(f"  [FIXED] {bug['bug_id']} ({bug['service']})")
        lines.append("")

    lines.append("Respond with a single JSON action object.")
    return "\n".join(lines)


# ── response parser ──────────────────────────────────────────────────────────


def parse_action(response_text: str) -> dict:
    """Extract a JSON action object from the LLM response."""
    text = response_text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Find JSON object in text
    match = re.search(r"\{[^{}]*\"action_type\"[^{}]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Fallback
    print(f"  [WARN] Could not parse LLM response, using noop: {text[:100]}")
    return {"action_type": "noop"}


# ── episode runner ───────────────────────────────────────────────────────────


def run_task(task_id: str) -> Dict[str, Any]:
    """Run a full episode for one task and return the results dict."""
    client = _get_client()
    log_start(task=task_id, env="sre-gym", model=MODEL_NAME or "unknown")

    rewards: List[float] = []
    step_count = 0
    score = 0.0
    total_reward = 0.0

    try:
        # 1. Reset environment
        reset_resp = call_env("/reset", method="POST", data={"task_id": task_id})
        obs = reset_resp["observation"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        # 2. Episode loop
        while not done and step_count < MAX_STEPS:
            user_prompt = build_user_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                llm_response = completion.choices[0].message.content or ""
            except Exception as exc:
                llm_response = '{"action_type": "noop"}'
                print(f"  [WARN] LLM call failed: {exc}", file=sys.stderr)

            messages.append({"role": "assistant", "content": llm_response})

            # Parse and step
            action = parse_action(llm_response)

            try:
                step_resp = call_env("/step", method="POST", data=action)
                obs = step_resp["observation"]
                reward = step_resp["reward"]
                done = step_resp["done"]
                step_count = obs["step_number"]
                rewards.append(reward)
                total_reward += reward

                action_str = f"{action['action_type']}({action.get('bug_id', '')})"
                log_step(step=step_count, action=action_str, reward=reward, done=done)
            except Exception as e:
                log_step(step=step_count, action="error", reward=0.0, done=True, error=str(e))
                break

            # Keep message history manageable
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

        # 3. Get final state and grade
        state = call_env("/state", method="GET")
        total_reward = state.get("total_reward", total_reward)
        grader_resp = call_env(
            "/grader",
            method="POST",
            data={"task_id": task_id, "episode_state": state},
        )
        score = grader_resp["score"]

    except Exception as e:
        print(f"  [ERROR] Episode failed: {e}", file=sys.stderr)
    finally:
        success = score >= 0.1
        log_end(success=success, steps=step_count, score=score, rewards=rewards)

    # FIXED: include total_reward so main() summary never crashes
    return {
        "task_id": task_id,
        "score": score,
        "steps": step_count,
        "total_reward": total_reward,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all three tasks in sequence and print a summary."""
    tasks = ["task1_easy", "task2_medium", "task3_hard"]
    results: List[Dict[str, Any]] = []

    for task_id in tasks:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Task {task_id} failed: {exc}")
            results.append(
                {"task_id": task_id, "score": 0.0, "total_reward": 0.0, "steps": 0}
            )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    print(f"\n{'=' * 40}")
    print(f"  BASELINE SUMMARY")
    print(f"{'=' * 40}")
    for r in results:
        print(
            f"  {r['task_id']:15s}  "
            f"score={r['score']:.4f}  "
            f"reward={r['total_reward']:.4f}  "
            f"steps={r['steps']}"
        )
    print(f"{'─' * 40}")
    print(f"  {'Average score':15s}  {avg_score:.4f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
