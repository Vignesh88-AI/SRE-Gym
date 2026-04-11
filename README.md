---
title: SRE-Gym
emoji: 🚨
colorFrom: red
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - incident-response
  - bug-triage
---

# SRE-Gym 🚨

**Production incident triage environment for training AI agents on real SRE workflows.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-brightgreen)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## Overview

SRE-Gym is an OpenEnv-compliant reinforcement learning environment that simulates real-world Site Reliability Engineering (SRE) on-call workflows. An AI agent faces a queue of production bugs and must triage, investigate, and fix them under a strict action budget.

Unlike simple classification tasks, SRE-Gym tests **sequential decision-making under uncertainty**:

1. **Hidden information** — stack traces are only revealed after spending budget to investigate
2. **Cascading failures** — bugs form dependency graphs; fixing symptoms without root causes wastes budget
3. **Red herrings** — CRITICAL-severity alerts that are actually known noise (flaky tests, misconfigured monitors)
4. **Dynamic spread** — unfixed bugs grow in impact every step, punishing slow triage
5. **Budget constraints** — the agent cannot investigate everything and must prioritise

These mechanics mirror the real challenges faced by SRE teams during production incidents.

---

## Environment Description

### Observation Space

| Field | Type | Description |
|---|---|---|
| `bugs` | `List[PublicBugReport]` | Visible bug reports (stack traces hidden until investigated) |
| `step_number` | `int` | Current step in the episode |
| `budget_remaining` | `int` | Action points left |
| `total_affected_users` | `int` | Sum of `affected_users` across all unfixed bugs |
| `goal` | `str` | Human-readable objective |
| `last_action_result` | `str` | Feedback from previous action |
| `task_id` | `str` | Current task identifier |
| `done` | `bool` | Whether the episode has ended |

Each `PublicBugReport` contains:

| Field | Type | Description |
|---|---|---|
| `bug_id` | `str` | Unique identifier |
| `error_message` | `str` | Alert description |
| `severity` | `CRITICAL\|HIGH\|MEDIUM\|LOW` | Declared severity (may be misleading) |
| `frequency` | `int` | How many times the alert fired |
| `affected_users` | `int` | Users currently impacted |
| `service` | `str` | Originating microservice |
| `investigated` | `bool` | Whether `investigate()` has been called |
| `fixed` | `bool` | Whether the bug is fixed |
| `stack_trace` | `str \| null` | Only revealed after `investigate()` |

### Action Space

| Action | Budget Cost | Description |
|---|---|---|
| `investigate` | 1 | Reveal stack trace for a bug |
| `fix` | 2 | Fix a bug (if root cause is unfixed, bug returns next step) |
| `escalate` | 1 | Escalate to on-call team (+0.15 reward if CRITICAL) |
| `ignore` | 0 | Skip a bug (+0.1 reward if red herring, −0.15 if real) |
| `noop` | 0 | Do nothing (−0.01 reward) |

### Reward Function

```
Per-step reward:
  fix (genuine):     (affected_users / 1000) × urgency_weight
  fix (red herring): −0.2
  fix (symptom):     +0.1  (but bug returns — wasted budget)
  investigate:       +0.05
  ignore (correct):  +0.1
  ignore (wrong):    −0.15

Urgency weights:
  CRITICAL = 2.0  |  HIGH = 1.5  |  MEDIUM = 1.0  |  LOW = 0.5

End-of-episode bonus:
  All CRITICALs fixed: +0.5
  No red herrings wasted: +0.2
  Remaining budget: +0.05 per point
```

---

## Tasks

| Task ID | Difficulty | Bugs | Budget | Max Steps | Key Challenge |
|---|---|---|---|---|---|
| `task1_easy` | 🟢 Easy | 5 | 8 | 10 | Clear severity signals, straightforward priority triage |
| `task2_medium` | 🟡 Medium | 15 | 12 | 15 | 3 red herrings, 2 hidden data-corruption bugs, 1 cascade |
| `task3_hard` | 🔴 Hard | 25 | 15 | 20 | 3 cascade chains (depth 4), 5 red herrings, symptom-return mechanic |

---

## What Makes This Hard

| Mechanic | Description |
|---|---|
| **Information asymmetry** | Stack traces are hidden. The agent must spend budget to `investigate()` before it knows if a bug is real or noise. |
| **Cascading failures** | Bug dependency graphs mean fixing symptoms wastes 2 budget points — the bug returns next step if the root cause is unfixed. |
| **Red herrings** | CRITICAL alerts with high frequency that are actually flaky tests or misconfigured monitors. Fixing them costs budget and yields negative reward. |
| **Dynamic spread** | Every unfixed real bug increases `affected_users` by `spread_rate` each step. Delayed triage compounds the damage. |
| **Constrained budget** | The agent cannot investigate all bugs. Task 3 has 25 bugs but only 15 budget points — forcing strategic prioritisation. |

---

## Setup & Installation

### Local

```bash
git clone https://github.com/your-username/SRE-gym.git
cd SRE-gym
pip install -r requirements.txt
```

### Docker

```bash
docker build -t sre-gym .
docker run -p 7860:7860 sre-gym
```

### Verify

```bash
curl http://localhost:7860/health
# → {"status": "ok", "env": "sre-gym"}
```

---

## Usage

### Python Example

```python
import requests

BASE = "http://localhost:7860"

# Reset environment with a task
reset = requests.post(f"{BASE}/reset", json={"task_id": "task1_easy"}).json()
obs = reset["observation"]

print(f"Bugs: {len(obs['bugs'])}, Budget: {obs['budget_remaining']}")

# Investigate the first bug
step1 = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "bug_id": "BUG001"
}).json()

print(f"Reward: {step1['reward']}, Result: {step1['info']['result']}")

# Fix the bug
step2 = requests.post(f"{BASE}/step", json={
    "action_type": "fix",
    "bug_id": "BUG001",
    "fix_strategy": "hotfix"
}).json()

print(f"Reward: {step2['reward']}, Done: {step2['done']}")

# Get final state and grade
state = requests.get(f"{BASE}/state").json()
grade = requests.post(f"{BASE}/grader", json={
    "task_id": "task1_easy",
    "episode_state": state
}).json()

print(f"Score: {grade['score']}")
```

---

## Baseline Scores

| Task | Heuristic Baseline | Random Agent | LLM Agent |
|---|---|---|---|
| `task1_easy` | TBD | TBD | TBD |
| `task2_medium` | TBD | TBD | TBD |
| `task3_hard` | TBD | TBD | TBD |

Run the built-in heuristic baseline:

```bash
curl -X POST http://localhost:7860/baseline -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy"}'
```

---

## Running Inference

The `inference.py` script runs an LLM agent against all three tasks:

```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export ENV_URL="http://localhost:7860"

# Start the server first
python src/server.py &

# Run the LLM agent
python inference.py
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` / `API_KEY` | ✅ | — | Bearer token for LLM provider |
| `MODEL_NAME` | ✅ | — | Model identifier |
| `API_BASE_URL` | ❌ | `https://router.huggingface.co/v1` | LLM endpoint |
| `ENV_URL` | ❌ | `http://localhost:7860` | SRE-Gym server URL |

---

## Architecture

```
SRE-gym/
├── openenv.yaml              # OpenEnv environment specification
├── inference.py               # LLM baseline inference script
├── Dockerfile                 # Container build
├── requirements.txt           # Python dependencies
├── README.md
├── src/
│   ├── models.py              # Pydantic v2 data models
│   ├── environment.py         # Core SREGymEnvironment (reset/step)
│   ├── graders.py             # Task-specific scoring (0.0–1.0)
│   ├── server.py              # FastAPI server (7 endpoints)
│   └── incidents/             # Task JSON files
│       ├── task1_easy.json    #   5 bugs — easy
│       ├── task2_medium.json  #  15 bugs — medium
│       └── task3_hard.json    #  25 bugs — hard
└── tests/
    └── test_env.py            # Pytest smoke tests
```

---

## License

MIT
