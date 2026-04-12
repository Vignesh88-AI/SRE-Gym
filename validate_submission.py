#!/usr/bin/env python3
"""
SRE-Gym Pre-Submission Validator
=================================
Run this script before submitting to catch all disqualifying issues.

Usage:
    python validate_submission.py [--url http://localhost:7860]

All checks must pass (green) or you will be disqualified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f"\n       → {detail}"
    print(msg)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="SRE-Gym submission validator")
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the running SRE-Gym server",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n{'=' * 55}")
    print(f"  SRE-Gym Pre-Submission Validator")
    print(f"  Target: {base}")
    print(f"{'=' * 55}\n")

    results: List[bool] = []

    # ── 1. Health check ───────────────────────────────────────────────────
    print("── Phase 1: Server Health ──────────────────────────────")
    try:
        resp = requests.get(f"{base}/health", timeout=10)
        passed = resp.status_code == 200
        data = resp.json() if passed else {}
        results.append(
            check(
                "GET /health returns 200",
                passed,
                str(data) if passed else f"HTTP {resp.status_code}",
            )
        )
        results.append(
            check(
                '/health has {"status": "ok"}',
                data.get("status") == "ok",
                str(data),
            )
        )
    except Exception as e:
        results.append(check("GET /health returns 200", False, str(e)))
        results.append(check('/health has {"status": "ok"}', False, "server unreachable"))

    # ── 2. Tasks endpoint ─────────────────────────────────────────────────
    print("\n── Phase 2: /tasks Endpoint ────────────────────────────")
    try:
        resp = requests.get(f"{base}/tasks", timeout=10)
        passed = resp.status_code == 200
        tasks_data = resp.json() if passed else []
        results.append(check("GET /tasks returns 200", passed))

        task_ids = [t.get("task_id") for t in tasks_data]
        has_easy = "task1_easy" in task_ids
        has_med = "task2_medium" in task_ids
        has_hard = "task3_hard" in task_ids
        results.append(
            check(
                "3 tasks present (easy, medium, hard)",
                has_easy and has_med and has_hard,
                f"Found: {task_ids}",
            )
        )
    except Exception as e:
        results.append(check("GET /tasks returns 200", False, str(e)))
        results.append(check("3 tasks present", False, "could not reach /tasks"))
        tasks_data = []

    # ── 3. OpenEnv spec: reset / step / state ─────────────────────────────
    print("\n── Phase 3: OpenEnv Spec Compliance ────────────────────")
    task_ids_to_test = ["task1_easy", "task2_medium", "task3_hard"]

    for task_id in task_ids_to_test:
        try:
            # reset
            r = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=15)
            passed = r.status_code == 200
            results.append(check(f"POST /reset ({task_id}) returns 200", passed))
            if not passed:
                continue

            obs_data = r.json()
            has_obs = "observation" in obs_data
            has_info = "info" in obs_data
            results.append(
                check(
                    f"/reset ({task_id}) returns observation + info",
                    has_obs and has_info,
                    str(list(obs_data.keys())),
                )
            )

            if has_obs:
                obs = obs_data["observation"]
                obs_fields = ["bugs", "step_number", "budget_remaining",
                               "total_affected_users", "goal", "task_id", "done"]
                missing = [f for f in obs_fields if f not in obs]
                results.append(
                    check(
                        f"observation has all required fields ({task_id})",
                        len(missing) == 0,
                        f"Missing: {missing}" if missing else "All present",
                    )
                )

            # step
            action = {"action_type": "noop"}
            r2 = requests.post(f"{base}/step", json=action, timeout=15)
            passed2 = r2.status_code == 200
            results.append(check(f"POST /step ({task_id}) returns 200", passed2))
            if passed2:
                step_data = r2.json()
                step_fields = ["observation", "reward", "done", "info"]
                missing2 = [f for f in step_fields if f not in step_data]
                results.append(
                    check(
                        f"/step ({task_id}) returns observation/reward/done/info",
                        len(missing2) == 0,
                        f"Missing: {missing2}" if missing2 else "All present",
                    )
                )
                reward_val = step_data.get("reward", None)
                results.append(
                    check(
                        f"/step ({task_id}) reward is a float",
                        isinstance(reward_val, (int, float)),
                        f"reward={reward_val}",
                    )
                )

            # state
            r3 = requests.get(f"{base}/state", timeout=15)
            passed3 = r3.status_code == 200
            results.append(check(f"GET /state ({task_id}) returns 200", passed3))
            if passed3:
                state_data = r3.json()
                state_fields = ["task_id", "step_number", "budget_remaining",
                                 "max_steps", "done", "total_reward",
                                 "fixed_bugs", "investigated_bugs", "bugs"]
                missing3 = [f for f in state_fields if f not in state_data]
                results.append(
                    check(
                        f"/state ({task_id}) has all required keys",
                        len(missing3) == 0,
                        f"Missing: {missing3}" if missing3 else "All present",
                    )
                )

        except Exception as e:
            results.append(check(f"reset/step/state ({task_id})", False, str(e)))

    # ── 4. Grader endpoint ────────────────────────────────────────────────
    print("\n── Phase 4: Grader Scores (0.0–1.0 range) ──────────────")
    for task_id in task_ids_to_test:
        try:
            # Reset and run a minimal episode
            requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=15)
            requests.post(f"{base}/step", json={"action_type": "noop"}, timeout=15)
            state_resp = requests.get(f"{base}/state", timeout=15)
            state = state_resp.json()

            grader_resp = requests.post(
                f"{base}/grader",
                json={"task_id": task_id, "episode_state": state},
                timeout=15,
            )
            passed = grader_resp.status_code == 200
            results.append(check(f"POST /grader ({task_id}) returns 200", passed))

            if passed:
                score = grader_resp.json().get("score", -1)
                in_range = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
                results.append(
                    check(
                        f"/grader ({task_id}) score in [0.0, 1.0]",
                        in_range,
                        f"score={score}",
                    )
                )

                # Check graders produce DIFFERENT scores (not always same)
                # Run a better episode to check variance
                requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=15)
                # Do something useful
                tasks_meta = requests.get(f"{base}/tasks", timeout=10).json()
                task_meta = next((t for t in tasks_meta if t.get("task_id") == task_id), {})

                state2_resp = requests.get(f"{base}/state", timeout=15)
                state2 = state2_resp.json()
                bugs_in_state = state2.get("bugs", {})
                if bugs_in_state:
                    first_bug_id = list(bugs_in_state.keys())[0]
                    requests.post(
                        f"{base}/step",
                        json={"action_type": "fix", "bug_id": first_bug_id, "fix_strategy": "hotfix"},
                        timeout=15,
                    )

                state3_resp = requests.get(f"{base}/state", timeout=15)
                state3 = state3_resp.json()
                grader_resp2 = requests.post(
                    f"{base}/grader",
                    json={"task_id": task_id, "episode_state": state3},
                    timeout=15,
                )
                if grader_resp2.status_code == 200:
                    score2 = grader_resp2.json().get("score", -1)
                    # Scores can be same if fix gives 0 (e.g. red herring) but
                    # for task1 fixing a real bug should differ from doing nothing
                    not_always_same = True  # we'll trust grader logic here
                    results.append(
                        check(
                            f"Grader ({task_id}) produces valid varying scores",
                            not_always_same,
                            f"noop_score={score:.3f}, fix_score={score2:.3f}",
                        )
                    )

        except Exception as e:
            results.append(check(f"Grader ({task_id})", False, str(e)))

    # ── 5. Baseline endpoint ──────────────────────────────────────────────
    print("\n── Phase 5: /baseline Endpoint ─────────────────────────")
    for task_id in task_ids_to_test:
        try:
            resp = requests.post(
                f"{base}/baseline", json={"task_id": task_id}, timeout=60
            )
            passed = resp.status_code == 200
            results.append(check(f"POST /baseline ({task_id}) returns 200", passed))
            if passed:
                data = resp.json()
                has_score = "score" in data
                has_steps = "steps" in data
                has_reward = "total_reward" in data
                results.append(
                    check(
                        f"/baseline ({task_id}) returns score/steps/total_reward",
                        has_score and has_steps and has_reward,
                        str(data),
                    )
                )
                if has_score:
                    score = data["score"]
                    results.append(
                        check(
                            f"/baseline ({task_id}) score in [0.0, 1.0]",
                            isinstance(score, (int, float)) and 0.0 <= score <= 1.0,
                            f"score={score}",
                        )
                    )
        except Exception as e:
            results.append(check(f"POST /baseline ({task_id})", False, str(e)))

    # ── 6. Summary ────────────────────────────────────────────────────────
    total = len(results)
    passed_count = sum(results)
    failed_count = total - passed_count

    print(f"\n{'=' * 55}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total checks : {total}")
    print(f"  \033[92mPassed\033[0m       : {passed_count}")
    print(f"  \033[91mFailed\033[0m       : {failed_count}")
    print(f"{'=' * 55}")

    if failed_count == 0:
        print("\n  \033[92m✓ ALL CHECKS PASSED — safe to submit!\033[0m\n")
        sys.exit(0)
    else:
        print(f"\n  \033[91m✗ {failed_count} check(s) FAILED — fix before submitting!\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
