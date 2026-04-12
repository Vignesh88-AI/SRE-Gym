"""
SRE-Gym: FastAPI server exposing the OpenEnv bug triage environment.
Gradio UI mounted at / for HF Spaces compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.environment import SREGymEnvironment
from src.graders import run_grader
from src.models import Action, Observation, ResetResult, StepResult

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SRE-Gym",
    description="OpenEnv-compliant SRE production incident triage environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (HTTP sessions) ─────────────────────────────────────────────

sessions: Dict[str, SREGymEnvironment] = {}

# ── Incident data ─────────────────────────────────────────────────────────────

_INCIDENTS_DIR = Path(__file__).parent / "incidents"

_TASK_META = [
    {"task_id": "task1_easy",   "difficulty": "easy"},
    {"task_id": "task2_medium", "difficulty": "medium"},
    {"task_id": "task3_hard",   "difficulty": "hard"},
]

_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["investigate", "fix", "escalate", "ignore", "noop"],
            "description": "The type of action to perform",
        },
        "bug_id": {
            "type": "string",
            "description": "Required for investigate, fix, escalate, ignore",
        },
        "fix_strategy": {
            "type": "string",
            "enum": ["hotfix", "rollback", "restart", "patch"],
            "description": "Required when action_type is fix",
        },
    },
}

# ── Request / Response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_easy"

class GraderRequest(BaseModel):
    task_id: str
    episode_state: dict

class BaselineRequest(BaseModel):
    task_id: str = "task1_easy"

class GraderResponse(BaseModel):
    score: float

class BaselineResponse(BaseModel):
    task_id: str
    score: float
    steps: int
    total_reward: float

class TaskInfo(BaseModel):
    task_id: str
    description: str
    budget: int
    max_steps: int
    difficulty: str
    action_schema: dict

class HealthResponse(BaseModel):
    status: str
    env: str

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
def reset(body: Optional[ResetRequest] = None) -> ResetResult:
    """Start a new episode. Returns initial observation.
    body is optional — defaults to task1_easy if not provided.
    """
    task_id = (body.task_id if body else None) or "task1_easy"
    try:
        env = SREGymEnvironment(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = env.reset()
    sessions["default"] = env
    print(f"[START] task={task_id} env=sre-gym model=server", flush=True)
    return result


@app.post("/step", response_model=StepResult)
def step(action: Optional[Action] = None) -> StepResult:
    """Execute one action. Returns observation, reward, done, info."""
    if action is None:
        action = Action(action_type="noop")
    env = sessions.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call POST /reset first.")
    result = env.step(action)
    obs = result.observation
    print(
        f"[STEP] step={obs.step_number} action={action.action_type} "
        f"bug_id={action.bug_id} reward={result.reward:.4f} done={result.done}",
        flush=True,
    )
    return result


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Return full internal state for the current session."""
    env = sessions.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call POST /reset first.")
    return env.state()


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    """Return all tasks with descriptions and action schema."""
    results: List[TaskInfo] = []
    for meta in _TASK_META:
        path = _INCIDENTS_DIR / f"{meta['task_id']}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        results.append(TaskInfo(
            task_id=meta["task_id"],
            description=raw.get("description", ""),
            budget=raw.get("budget", 0),
            max_steps=raw.get("max_steps", 0),
            difficulty=meta["difficulty"],
            action_schema=_ACTION_SCHEMA,
        ))
    return results


@app.post("/grader", response_model=GraderResponse)
def grade(body: GraderRequest) -> GraderResponse:
    """Score a completed episode. Returns score in [0.0, 1.0]."""
    try:
        score = run_grader(body.task_id, body.episode_state)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    print(f"[END] task_id={body.task_id} score={score:.4f}", flush=True)
    return GraderResponse(score=score)


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(body: BaselineRequest) -> BaselineResponse:
    """Run the heuristic agent and return its score."""
    try:
        env = SREGymEnvironment(body.task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reset_result = env.reset()
    obs = reset_result.observation
    done = obs.done
    while not done:
        action = _heuristic_action(obs)
        step_result = env.step(action)
        obs = step_result.observation
        done = step_result.done
    state = env.state()
    score = run_grader(body.task_id, state)
    print(
        f"[BASELINE] task_id={body.task_id} score={score:.4f} "
        f"steps={state['step_number']} total_reward={state['total_reward']:.4f}",
        flush=True,
    )
    return BaselineResponse(
        task_id=body.task_id,
        score=score,
        steps=state["step_number"],
        total_reward=state["total_reward"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", env="sre-gym")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket — each connection gets its own isolated environment instance."""
    await websocket.accept()
    ws_env: Optional[SREGymEnvironment] = None
    client_id = id(websocket)
    print(f"[WS] Client {client_id} connected", flush=True)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "reset":
                task_id = data.get("task_id", "task1_easy")
                try:
                    ws_env = SREGymEnvironment(task_id)
                    result = ws_env.reset()
                    sessions["default"] = ws_env
                    print(f"[WS] Client {client_id} reset task={task_id}", flush=True)
                    await websocket.send_json(result.model_dump())
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            elif msg_type == "step":
                if ws_env is None:
                    await websocket.send_json({"error": "No active session. Send reset first."})
                else:
                    try:
                        action = Action(**data.get("action", {}))
                        result = ws_env.step(action)
                        obs = result.observation
                        print(
                            f"[WS] Client {client_id} step={obs.step_number} "
                            f"action={action.action_type} reward={result.reward:.4f}",
                            flush=True,
                        )
                        await websocket.send_json(result.model_dump())
                    except Exception as e:
                        await websocket.send_json({"error": str(e)})

            elif msg_type == "state":
                if ws_env is None:
                    await websocket.send_json({"error": "No active session."})
                else:
                    await websocket.send_json(ws_env.state())

            elif msg_type == "grader":
                env_to_grade = ws_env
                task_id = data.get("task_id")
                episode_state = data.get("episode_state")
                try:
                    if task_id and episode_state:
                        score = run_grader(task_id, episode_state)
                    elif env_to_grade:
                        state = env_to_grade.state()
                        score = run_grader(state.get("task_id", "task1_easy"), state)
                    else:
                        await websocket.send_json({"error": "No active session."})
                        continue
                    await websocket.send_json({"score": score})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected", flush=True)
    except Exception as e:
        print(f"[WS] Client {client_id} error: {e}", flush=True)


# ── Heuristic baseline agent ───────────────────────────────────────────────────

def _heuristic_action(obs: Observation) -> Action:
    """Greedy heuristic: investigate then fix the highest-impact unfixed bug."""
    unfixed = [b for b in obs.bugs if not b.fixed]
    if not unfixed:
        return Action(action_type="noop")
    unfixed.sort(key=lambda b: b.affected_users, reverse=True)
    target = unfixed[0]
    if not target.investigated and obs.budget_remaining >= 1:
        return Action(action_type="investigate", bug_id=target.bug_id)
    if obs.budget_remaining >= 2:
        return Action(action_type="fix", bug_id=target.bug_id, fix_strategy="hotfix")
    critical = [b for b in unfixed if b.severity == "CRITICAL"]
    if critical and obs.budget_remaining >= 1:
        return Action(action_type="escalate", bug_id=critical[0].bug_id)
    return Action(action_type="noop")


# ── Gradio UI ──────────────────────────────────────────────────────────────────
# Gradio runs server-side — no browser fetch calls, no CORS/iframe issues.
# This is why buttons work when mounted at / on HF Spaces.

def _build_gradio_app():
    """Build fully interactive Gradio UI for SRE-Gym."""
    try:
        import gradio as gr

        # ── Server-side handlers (all Python, no HTTP) ─────────────────────

        def do_reset(task_id: str, env_state: dict):
            try:
                env = SREGymEnvironment(task_id)
                result = env.reset()
                obs = result.observation
                new_state = {
                    "task_id": task_id,
                    "env": env,
                    "active": True,
                    "log": [
                        f"Episode started: {task_id}",
                        f"Goal: {obs.goal}",
                        f"Budget: {obs.budget_remaining} | Bugs: {len(obs.bugs)}",
                        "-" * 60,
                    ],
                }
                return (
                    new_state,
                    _fmt_bugs(obs.bugs),
                    _fmt_stats(obs),
                    "\n".join(new_state["log"]),
                    gr.update(interactive=True),   # step_btn
                    gr.update(interactive=True),   # heuristic_btn
                    f"Episode started. {len(obs.bugs)} bugs in queue. Budget: {obs.budget_remaining}",
                    "",
                )
            except Exception as e:
                return (
                    env_state, "", "",
                    f"Error during reset: {e}",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    f"Error: {e}", "",
                )

        def do_step(action_type: str, bug_id: str, fix_strategy: str, env_state: dict):
            if not env_state.get("active") or not env_state.get("env"):
                return (
                    env_state, "", "",
                    "\n".join(env_state.get("log", ["Click Reset first."])),
                    gr.update(), gr.update(),
                    "Click Reset to start an episode.", "",
                )
            try:
                env = env_state["env"]
                bug_id_clean = (bug_id or "").strip() or None
                action = Action(
                    action_type=action_type,
                    bug_id=bug_id_clean,
                    fix_strategy=fix_strategy if action_type == "fix" else None,
                )
                result = env.step(action)
                obs = result.observation
                reward = result.reward
                sign = "+" if reward >= 0 else ""
                log_line = (
                    f"Step {obs.step_number}: {action_type}"
                    f"({bug_id_clean or ''})  reward={sign}{reward:.3f}"
                )
                if result.info.get("result"):
                    log_line += f"\n  {result.info['result'][:100]}"
                env_state["log"].append(log_line)
                score_text = ""

                if obs.done:
                    env_state["active"] = False
                    state = env.state()
                    score = run_grader(env_state["task_id"], state)
                    score_text = f"Score: {score:.3f} / 1.0  ({int(score * 100)}/100)"
                    env_state["log"] += ["-" * 60, score_text]
                    return (
                        env_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                        "\n".join(env_state["log"]),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        result.info.get("result", f"Episode done. Reward: {sign}{reward:.3f}"),
                        score_text,
                    )

                return (
                    env_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                    "\n".join(env_state["log"]),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    result.info.get("result", f"{action_type} done. Reward: {sign}{reward:.3f}"),
                    score_text,
                )
            except Exception as e:
                return (
                    env_state, "", "",
                    "\n".join(env_state.get("log", [])),
                    gr.update(), gr.update(),
                    f"Error: {e}", "",
                )

        def do_heuristic(task_id: str, env_state: dict):
            try:
                env = SREGymEnvironment(task_id)
                r = env.reset()
                obs = r.observation
                log = [
                    f"Heuristic agent running on {task_id}",
                    f"Goal: {obs.goal}",
                    f"Budget: {obs.budget_remaining} | Bugs: {len(obs.bugs)}",
                    "-" * 60,
                ]
                done = obs.done
                step = 0
                while not done and step < 25:
                    action = _heuristic_action(obs)
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    step += 1
                    sign = "+" if result.reward >= 0 else ""
                    log.append(
                        f"Step {step:2d}: {action.action_type}"
                        f"({action.bug_id or ''})  "
                        f"reward={sign}{result.reward:.3f}  "
                        f"budget={obs.budget_remaining}"
                    )
                    if result.info.get("result"):
                        log.append(f"  {result.info['result'][:90]}")

                state = env.state()
                score = run_grader(task_id, state)
                score_text = f"Score: {score:.3f} / 1.0  ({int(score * 100)}/100)"
                log += [
                    "-" * 60,
                    score_text,
                    f"Total reward: {state['total_reward']:.3f}",
                    f"Bugs fixed: {len(state['fixed_bugs'])} / {len(state['bugs'])}",
                ]
                new_state = {"task_id": task_id, "env": None, "active": False, "log": log}
                return (
                    new_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                    "\n".join(log),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    score_text, score_text,
                )
            except Exception as e:
                return (
                    env_state, "", "", f"Error: {e}",
                    gr.update(), gr.update(),
                    f"Error: {e}", "",
                )

        def do_get_state(env_state: dict) -> str:
            if not env_state.get("env"):
                return "No active session. Click Reset first."
            try:
                s = env_state["env"].state()
                return (
                    f"Step: {s['step_number']}  |  "
                    f"Budget: {s['budget_remaining']}  |  "
                    f"Fixed: {len(s['fixed_bugs'])}  |  "
                    f"Total reward: {s['total_reward']:.3f}"
                )
            except Exception as e:
                return f"Error: {e}"

        def _fmt_bugs(bugs) -> str:
            if not bugs:
                return "No bugs in queue."
            header = f"{'Severity':<10} {'Bug ID':<16} {'Service':<22} {'Users':<8} Status"
            sep = "-" * 72
            rows = [header, sep]
            for b in bugs:
                status = "FIXED" if b.fixed else ("investigated" if b.investigated else "active")
                rows.append(
                    f"{b.severity:<10} {b.bug_id:<16} {b.service:<22} "
                    f"{b.affected_users:<8} {status}"
                )
            return "\n".join(rows)

        def _fmt_stats(obs) -> str:
            bugs = obs.bugs or []
            fixed = sum(1 for b in bugs if b.fixed)
            return (
                f"Step: {obs.step_number}   |   "
                f"Budget: {obs.budget_remaining}   |   "
                f"Affected Users: {obs.total_affected_users:,}   |   "
                f"Fixed: {fixed} / {len(bugs)}"
            )

        # ── Gradio Blocks UI ───────────────────────────────────────────────

        with gr.Blocks(
            title="SRE-Gym | Production Incident Triage",
            theme=gr.themes.Soft(
                primary_hue="green",
                secondary_hue="blue",
                neutral_hue="slate",
            ),
            css="""
                footer { display: none !important; }
                .monospace textarea, .monospace input { font-family: monospace !important; font-size: 13px !important; }
            """,
        ) as demo:

            # Per-session state — each user gets their own environment
            env_state = gr.State({
                "task_id": "task1_easy",
                "env": None,
                "active": False,
                "log": [],
            })

            gr.Markdown(
                "# SRE-Gym\n"
                "**Production Incident Triage Environment for AI Agents** — OpenEnv Compliant\n\n"
                "Train AI agents to triage production bugs like real Site Reliability Engineers."
            )

            with gr.Tabs():

                # ── Playground Tab ─────────────────────────────────────────
                with gr.Tab("Playground"):
                    with gr.Row():

                        # Left column: controls
                        with gr.Column(scale=1, min_width=300):

                            gr.Markdown("### Episode Setup")
                            task_dd = gr.Dropdown(
                                choices=["task1_easy", "task2_medium", "task3_hard"],
                                value="task1_easy",
                                label="Task",
                                info="Easy=5 bugs, Medium=15 bugs, Hard=25 bugs with cascades",
                            )
                            with gr.Row():
                                reset_btn = gr.Button("Reset", variant="primary", scale=1)
                                state_btn = gr.Button("Get State", variant="secondary", scale=1)

                            gr.Markdown("### Take Action")
                            action_dd = gr.Dropdown(
                                choices=["investigate", "fix", "ignore", "escalate", "noop"],
                                value="investigate",
                                label="Action Type",
                                info="investigate=1pt  fix=2pt  escalate=1pt  ignore/noop=free",
                            )
                            bug_id_box = gr.Textbox(
                                label="Bug ID",
                                placeholder="e.g. BUG001 or BUG_H001",
                                info="Required for investigate, fix, ignore, escalate",
                            )
                            strategy_dd = gr.Dropdown(
                                choices=["hotfix", "rollback", "restart", "patch"],
                                value="hotfix",
                                label="Fix Strategy",
                                info="Only used when action is 'fix'",
                            )
                            with gr.Row():
                                step_btn = gr.Button(
                                    "Step", variant="primary", interactive=False, scale=1
                                )
                                heuristic_btn = gr.Button(
                                    "Run Heuristic Agent", variant="secondary",
                                    interactive=False, scale=1
                                )

                            result_box = gr.Textbox(
                                label="Last Action Result",
                                value="Click Reset to start an episode.",
                                interactive=False,
                                lines=3,
                            )
                            score_box = gr.Textbox(
                                label="Episode Score",
                                value="",
                                interactive=False,
                            )
                            state_info = gr.Textbox(
                                label="State Info",
                                value="",
                                interactive=False,
                            )

                        # Right column: live state
                        with gr.Column(scale=2):
                            stats_box = gr.Textbox(
                                label="Episode Stats",
                                value="",
                                interactive=False,
                                elem_classes=["monospace"],
                            )
                            bug_table_box = gr.Textbox(
                                label="Bug Queue",
                                value="Click Reset to load bugs",
                                lines=16,
                                interactive=False,
                                elem_classes=["monospace"],
                            )

                    log_box = gr.Textbox(
                        label="Action Log",
                        value="No episode started.",
                        lines=10,
                        interactive=False,
                        elem_classes=["monospace"],
                    )

                    # Wire up all buttons
                    reset_btn.click(
                        fn=do_reset,
                        inputs=[task_dd, env_state],
                        outputs=[env_state, bug_table_box, stats_box, log_box,
                                 step_btn, heuristic_btn, result_box, score_box],
                    )
                    step_btn.click(
                        fn=do_step,
                        inputs=[action_dd, bug_id_box, strategy_dd, env_state],
                        outputs=[env_state, bug_table_box, stats_box, log_box,
                                 step_btn, heuristic_btn, result_box, score_box],
                    )
                    heuristic_btn.click(
                        fn=do_heuristic,
                        inputs=[task_dd, env_state],
                        outputs=[env_state, bug_table_box, stats_box, log_box,
                                 step_btn, heuristic_btn, result_box, score_box],
                    )
                    state_btn.click(
                        fn=do_get_state,
                        inputs=[env_state],
                        outputs=[state_info],
                    )

                # ── Overview Tab ───────────────────────────────────────────
                with gr.Tab("Overview"):
                    gr.Markdown("""
## What is SRE-Gym?

SRE-Gym simulates the daily on-call workflow of a Site Reliability Engineer.
An AI agent faces a live queue of production bugs and must triage, investigate,
and fix them under a strict action budget — exactly as human SREs do during incidents.

## 5 Unique Mechanics

**1. Cascading Failure Graph**
Bugs have parent-child relationships. Fix a symptom without fixing its root cause
and the bug returns next step, wasting your budget. Mirrors real production cascades.

**2. Dynamic Spread**
Every step, unfixed bugs grow their `affected_users` count. Slow triage means
more users impacted. Forces speed vs thoroughness trade-offs.

**3. Fixed Action Budget**
`investigate()` costs 1 point. `fix()` costs 2 points. You cannot investigate
every bug. Strategic planning beats greedy picking every time.

**4. Misleading Red Herrings**
Some CRITICAL-severity alerts are known flaky tests — completely harmless.
Fixing them wastes budget (-0.2 reward). `investigate()` then `ignore()` for +0.1.

**5. Information Asymmetry**
Stack traces hidden until `investigate()` is called. Agent must decide: spend
budget to gather information, or act on partial knowledge?

## Reward Function

```
fix genuine bug:    (affected_users / 1000) × urgency_weight
                    CRITICAL=2.0  HIGH=1.5  MEDIUM=1.0  LOW=0.5
fix red herring:   -0.2
fix symptom only:  +0.1  (bug returns next step — root cause still active)
investigate:       +0.05
ignore correct:    +0.1  (correctly identified as noise)
ignore real bug:   -0.15
noop:              -0.01

End-of-episode bonus:
  All CRITICAL real bugs fixed:      +0.5
  No budget wasted on red herrings:  +0.2
  Remaining budget efficiency:       +0.05 per point left
```

## Task Difficulty

| Task | Bugs | Budget | Max Steps | Challenge |
|---|---|---|---|---|
| task1_easy | 5 | 8 pts | 10 | Clear severity, sort by priority |
| task2_medium | 15 | 12 pts | 15 | 3 red herrings, 2 cascades |
| task3_hard | 25 | 15 pts | 20 | 3 cascade chains, 5 red herrings |

## Dataset Integration

SRE-Gym incident data is grounded in real-world production bug patterns:

- **Incident structure** based on real PagerDuty/Opsgenie alert formats
- **Stack traces** reflect real Java, Python, Go production errors
- **Cascading patterns** based on published incident post-mortems
- **Severity signals** mirror real monitoring tool output (Datadog, Prometheus)
""")

                # ── Quick Start Tab ────────────────────────────────────────
                with gr.Tab("Quick Start"):
                    gr.Markdown("""
## Connect to SRE-Gym

### Python (HTTP)

```python
import requests

ENV = "https://argonite3-sre-gym.hf.space"

# 1. Start episode
obs = requests.post(f"{ENV}/reset", json={"task_id": "task1_easy"}).json()["observation"]
print(obs["goal"])
print(f"Bugs: {len(obs['bugs'])}  Budget: {obs['budget_remaining']}")

# 2. Investigate the highest-impact bug
result = requests.post(f"{ENV}/step", json={
    "action_type": "investigate",
    "bug_id": obs["bugs"][0]["bug_id"]
}).json()
print(result["observation"]["last_action_result"])

# 3. Fix it
result = requests.post(f"{ENV}/step", json={
    "action_type": "fix",
    "bug_id": obs["bugs"][0]["bug_id"],
    "fix_strategy": "hotfix"
}).json()
print(f"Reward: {result['reward']}")

# 4. Get final score
state = requests.get(f"{ENV}/state").json()
score = requests.post(f"{ENV}/grader", json={
    "task_id": "task1_easy",
    "episode_state": state
}).json()["score"]
print(f"Score: {score:.3f} / 1.0")
```

### Python (WebSocket)

```python
import asyncio, websockets, json

async def run():
    async with websockets.connect("wss://argonite3-sre-gym.hf.space/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "task_id": "task1_easy"}))
        obs = json.loads(await ws.recv())["observation"]

        # Step
        await ws.send(json.dumps({
            "type": "step",
            "action": {"action_type": "investigate", "bug_id": "BUG001"}
        }))
        result = json.loads(await ws.recv())

asyncio.run(run())
```

## API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/reset` | `{"task_id": "task1_easy"}` | Start new episode |
| POST | `/step` | `{"action_type": "fix", "bug_id": "BUG001", "fix_strategy": "hotfix"}` | Take action |
| GET | `/state` | — | Full episode state |
| GET | `/tasks` | — | All tasks + action schema |
| POST | `/grader` | `{"task_id": "...", "episode_state": {...}}` | Score episode |
| POST | `/baseline` | `{"task_id": "task1_easy"}` | Run heuristic agent |
| GET | `/health` | — | Health check |
| GET | `/docs` | — | Swagger UI |
| WS | `/ws` | — | WebSocket persistent session |

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| task1_easy | ~0.72 | Heuristic handles clear signals well |
| task2_medium | ~0.51 | Red herrings trip the greedy heuristic |
| task3_hard | ~0.31 | Cascade reasoning required |
""")

        return demo

    except ImportError:
        return None


# ── Mount Gradio at / for HF Spaces compatibility ─────────────────────────────
# Gradio handles the HF Spaces iframe correctly when mounted at root.
# FastAPI routes (/reset, /step, /state, etc.) remain fully accessible.

try:
    import gradio as gr
    from gradio.routes import mount_gradio_app
    _demo = _build_gradio_app()
    if _demo is not None:
        app = mount_gradio_app(app, _demo, path="/")
        print("[INFO] Gradio UI mounted at /", flush=True)
    else:
        print("[WARN] Gradio UI build failed", flush=True)
except Exception as _err:
    print(f"[WARN] Gradio not mounted: {_err}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
