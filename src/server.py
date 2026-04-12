"""
SRE-Gym: FastAPI server exposing the OpenEnv bug triage environment.
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

# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SRE-Gym",
    description="OpenEnv-compliant SRE bug triage environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── in-memory session store ─────────────────────────────────────────────────

sessions: Dict[str, SREGymEnvironment] = {}

# ── incidents directory ─────────────────────────────────────────────────────

_INCIDENTS_DIR = Path(__file__).parent / "incidents"

_TASK_META = [
    {"task_id": "task1_easy", "difficulty": "easy"},
    {"task_id": "task2_medium", "difficulty": "medium"},
    {"task_id": "task3_hard", "difficulty": "hard"},
]

# ── action schema (for /tasks endpoint) ─────────────────────────────────────

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


# ── request / response schemas ──────────────────────────────────────────────


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


# ── endpoints ────────────────────────────────────────────────────────────────




@app.get("/", response_class=HTMLResponse)
def landing_page() -> str:
    """Redirect to Gradio UI at /web."""
    return HTMLResponse(
        content='''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="0; url=/web">
<title>SRE-Gym</title>
</head>
<body>
<p>Redirecting to <a href="/web">SRE-Gym Playground</a>...</p>
<script>window.location.href = "/web";</script>
</body>
</html>''',
        status_code=200
    )

@app.post("/reset", response_model=ResetResult)
def reset(body: ResetRequest) -> ResetResult:
    """Create a new environment for the given task and return the initial observation."""
    try:
        env = SREGymEnvironment(body.task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = env.reset()
    sessions["default"] = env
    print(f"[START] task={body.task_id} env=sre-gym model=server", flush=True) #, flush=True)
    return result


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Execute one action in the current environment session."""
    env = sessions.get("default")
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )

    result = env.step(action)
    obs = result.observation
    print(
        f"[STEP] step={obs.step_number} "
        f"action={action.action_type} "
        f"bug_id={action.bug_id} "
        f"reward={result.reward:.4f} "
        f"done={result.done}",
        flush=True,
    )
    return result


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Return the full internal state of the current session."""
    env = sessions.get("default")
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )
    return env.state()


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    """Return metadata for every available task, including the action schema."""
    results: List[TaskInfo] = []
    for meta in _TASK_META:
        path = _INCIDENTS_DIR / f"{meta['task_id']}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        results.append(
            TaskInfo(
                task_id=meta["task_id"],
                description=raw.get("description", ""),
                budget=raw.get("budget", 0),
                max_steps=raw.get("max_steps", 0),
                difficulty=meta["difficulty"],
                action_schema=_ACTION_SCHEMA,
            )
        )
    return results


@app.post("/grader", response_model=GraderResponse)
def grade(body: GraderRequest) -> GraderResponse:
    """Score a completed episode using the task-specific grader."""
    try:
        score = run_grader(body.task_id, body.episode_state)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    print(f"[END] task_id={body.task_id} score={score:.4f}", flush=True)
    return GraderResponse(score=score)


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(body: BaselineRequest) -> BaselineResponse:
    """Run a simple heuristic agent and return its score.

    Heuristic: investigate then fix the highest affected_users unfixed bug.
    """
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
        f"[BASELINE] task_id={body.task_id} "
        f"score={score:.4f} "
        f"steps={state['step_number']} "
        f"total_reward={state['total_reward']:.4f}",
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
    """Simple liveness check."""
    return HealthResponse(status="ok", env="sre-gym")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — each connection gets its own isolated environment instance."""
    await websocket.accept()
    # Per-connection environment (not shared — proper session isolation)
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
                    # Also update shared session for HTTP /step compatibility
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
                        action_data = data.get("action", {})
                        action = Action(**action_data)
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
                    await websocket.send_json({"error": "No active session. Send reset first."})
                else:
                    await websocket.send_json(ws_env.state())

            elif msg_type == "grader":
                task_id = data.get("task_id")
                episode_state = data.get("episode_state")
                if task_id and episode_state:
                    try:
                        score = run_grader(task_id, episode_state)
                        await websocket.send_json({"score": score})
                    except Exception as e:
                        await websocket.send_json({"error": str(e)})
                elif ws_env is not None:
                    try:
                        state = ws_env.state()
                        t_id = state.get("task_id", "task1_easy")
                        score = run_grader(t_id, state)
                        await websocket.send_json({"score": score})
                    except Exception as e:
                        await websocket.send_json({"error": str(e)})
                else:
                    await websocket.send_json({"error": "No active session."})

            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected", flush=True)
    except Exception as e:
        print(f"[WS] Client {client_id} error: {e}", flush=True)


# ── heuristic baseline agent ────────────────────────────────────────────────


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
        return Action(
            action_type="fix",
            bug_id=target.bug_id,
            fix_strategy="hotfix",
        )

    critical = [b for b in unfixed if b.severity == "CRITICAL"]
    if critical and obs.budget_remaining >= 1:
        return Action(action_type="escalate", bug_id=critical[0].bug_id)

    return Action(action_type="noop")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

# ── Gradio Web UI ─────────────────────────────────────────────────────────────
# KEY INSIGHT: Gradio functions run SERVER-SIDE in Python.
# They call the environment directly — no HTTP fetch from browser,
# no CORS issues, no HF Spaces iframe problems. This is why it works.

def _build_gradio_app():
    """Build fully interactive Gradio UI for SRE-Gym."""
    try:
        import gradio as gr
        from src.graders import run_grader as _grade

        def do_reset(task_id, env_state):
            try:
                env = SREGymEnvironment(task_id)
                result = env.reset()
                obs = result.observation
                env_state = {
                    "task_id": task_id, "env": env, "active": True,
                    "log": [
                        f"Episode started: {task_id}",
                        f"Goal: {obs.goal}",
                        f"Budget: {obs.budget_remaining} | Bugs: {len(obs.bugs)}",
                        "-" * 60,
                    ],
                }
                return (
                    env_state,
                    _fmt_bugs(obs.bugs),
                    _fmt_stats(obs),
                    "\n".join(env_state["log"]),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    f"Episode started. {len(obs.bugs)} bugs. Budget: {obs.budget_remaining}",
                    "",
                )
            except Exception as e:
                return env_state, "", "", f"Error: {e}", gr.update(interactive=False), gr.update(interactive=False), f"Error: {e}", ""

        def do_step(action_type, bug_id, fix_strategy, env_state):
            if not env_state.get("active") or not env_state.get("env"):
                return env_state, "", "", "\n".join(env_state.get("log", [])), gr.update(), gr.update(), "Click Reset first.", ""
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
                log_line = f"Step {obs.step_number}: {action_type}({bug_id_clean or ''}) reward={sign}{reward:.3f}"
                if result.info.get("result"):
                    log_line += f"\n  {result.info['result'][:100]}"
                env_state["log"].append(log_line)
                score_text = ""
                if obs.done:
                    env_state["active"] = False
                    state = env.state()
                    score = _grade(env_state["task_id"], state)
                    score_text = f"Final Score: {score:.3f} / 1.0  ({int(score*100)}/100)"
                    env_state["log"] += ["-" * 60, score_text]
                    return (env_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                            "\n".join(env_state["log"]),
                            gr.update(interactive=False), gr.update(interactive=False),
                            result.info.get("result", f"Done. Reward: {sign}{reward:.3f}"), score_text)
                return (env_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                        "\n".join(env_state["log"]),
                        gr.update(interactive=True), gr.update(interactive=True),
                        result.info.get("result", f"{action_type} done. Reward: {sign}{reward:.3f}"), score_text)
            except Exception as e:
                return env_state, "", "", "\n".join(env_state.get("log", [])), gr.update(), gr.update(), f"Error: {e}", ""

        def do_heuristic(task_id, env_state):
            try:
                env = SREGymEnvironment(task_id)
                r = env.reset()
                obs = r.observation
                log = [f"Heuristic agent on {task_id}", f"Budget: {obs.budget_remaining} | Bugs: {len(obs.bugs)}", "-" * 60]
                done = obs.done
                step = 0
                while not done and step < 25:
                    action = _heuristic_action(obs)
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    step += 1
                    sign = "+" if result.reward >= 0 else ""
                    log.append(f"Step {step:2d}: {action.action_type}({action.bug_id or ''})  reward={sign}{result.reward:.3f}  budget={obs.budget_remaining}")
                    if result.info.get("result"):
                        log.append(f"  {result.info['result'][:90]}")
                state = env.state()
                score = _grade(task_id, state)
                score_text = f"Final Score: {score:.3f} / 1.0  ({int(score*100)}/100)"
                log += ["-" * 60, score_text, f"Total reward: {state['total_reward']:.3f}", f"Bugs fixed: {len(state['fixed_bugs'])}/{len(state['bugs'])}"]
                env_state = {"task_id": task_id, "env": None, "active": False, "log": log}
                return (env_state, _fmt_bugs(obs.bugs), _fmt_stats(obs),
                        "\n".join(log), gr.update(interactive=False), gr.update(interactive=False), score_text, score_text)
            except Exception as e:
                return env_state, "", "", f"Error: {e}", gr.update(), gr.update(), f"Error: {e}", ""

        def do_get_state(env_state):
            if not env_state.get("env"):
                return "No active session. Click Reset first."
            try:
                s = env_state["env"].state()
                return (f"Step: {s['step_number']}  Budget: {s['budget_remaining']}  "
                        f"Fixed: {len(s['fixed_bugs'])}  Total reward: {s['total_reward']:.3f}")
            except Exception as e:
                return f"Error: {e}"

        def _fmt_bugs(bugs) -> str:
            if not bugs:
                return "No bugs"
            hdr = f"{'Severity':<9}  {'Bug ID':<14}  {'Service':<20}  {'Users':<7}  Status"
            sep = "-" * 70
            rows = [hdr, sep]
            for b in bugs:
                status = "FIXED" if b.fixed else ("investigated" if b.investigated else "")
                rows.append(f"{b.severity:<9}  {b.bug_id:<14}  {b.service:<20}  {b.affected_users:<7}  {status}")
            return "\n".join(rows)

        def _fmt_stats(obs) -> str:
            bugs = obs.bugs or []
            fixed = sum(1 for b in bugs if b.fixed)
            return f"Step: {obs.step_number}   Budget: {obs.budget_remaining}   Affected Users: {obs.total_affected_users:,}   Fixed: {fixed}/{len(bugs)}"

        with gr.Blocks(
            title="SRE-Gym | Production Incident Triage",
            theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue", neutral_hue="slate"),
            css="footer { display: none !important; }"
        ) as demo:

            env_state = gr.State({"task_id": "task1_easy", "env": None, "active": False, "log": []})

            gr.Markdown("# SRE-Gym\n**Production Incident Triage Environment for AI Agents** — OpenEnv Compliant")

            with gr.Tabs():

                with gr.Tab("Playground"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Episode Setup")
                            task_dd = gr.Dropdown(
                                choices=["task1_easy", "task2_medium", "task3_hard"],
                                value="task1_easy", label="Task",
                            )
                            with gr.Row():
                                reset_btn = gr.Button("Reset", variant="primary")
                                state_btn = gr.Button("Get State", variant="secondary")

                            gr.Markdown("### Take Action")
                            action_dd = gr.Dropdown(
                                choices=["investigate", "fix", "ignore", "escalate", "noop"],
                                value="investigate",
                                label="Action Type  (investigate=1pt, fix=2pt, others=free)",
                            )
                            bug_id_box = gr.Textbox(label="Bug ID", placeholder="e.g. BUG001")
                            strategy_dd = gr.Dropdown(
                                choices=["hotfix", "rollback", "restart", "patch"],
                                value="hotfix", label="Fix Strategy  (only for fix action)",
                            )
                            with gr.Row():
                                step_btn = gr.Button("Step", variant="primary", interactive=False)
                                heuristic_btn = gr.Button("Run Heuristic Agent", variant="secondary", interactive=False)

                            result_box = gr.Textbox(label="Last Result", value="Click Reset to start.", interactive=False)
                            score_box = gr.Textbox(label="Episode Score", value="", interactive=False)

                        with gr.Column(scale=2):
                            stats_box = gr.Textbox(label="Stats", value="", interactive=False)
                            bug_table_box = gr.Textbox(label="Bug Queue", value="Click Reset to load bugs", lines=14, interactive=False)
                            state_info = gr.Textbox(label="State Info", value="", interactive=False)

                    log_box = gr.Textbox(label="Action Log", value="No episode started.", lines=8, interactive=False)

                    reset_btn.click(fn=do_reset, inputs=[task_dd, env_state],
                                    outputs=[env_state, bug_table_box, stats_box, log_box, step_btn, heuristic_btn, result_box, score_box])
                    step_btn.click(fn=do_step, inputs=[action_dd, bug_id_box, strategy_dd, env_state],
                                   outputs=[env_state, bug_table_box, stats_box, log_box, step_btn, heuristic_btn, result_box, score_box])
                    heuristic_btn.click(fn=do_heuristic, inputs=[task_dd, env_state],
                                        outputs=[env_state, bug_table_box, stats_box, log_box, step_btn, heuristic_btn, result_box, score_box])
                    state_btn.click(fn=do_get_state, inputs=[env_state], outputs=[state_info])

                with gr.Tab("Overview"):
                    gr.Markdown("""
## 5 Unique Mechanics

**1. Cascading Failure Graph** — Bugs have parent-child relationships. Fix a symptom before its root cause and the bug returns next step, wasting budget.

**2. Dynamic Spread** — Unfixed bugs grow affected_users every step. Slow triage = more users impacted.

**3. Fixed Action Budget** — investigate=1pt, fix=2pt. Cannot investigate everything. Must plan strategically.

**4. Misleading Red Herrings** — Some CRITICAL alerts are flaky tests. Fixing them wastes budget (-0.2). investigate() then ignore() for +0.1.

**5. Information Asymmetry** — Stack traces hidden until investigate() is called.

## Reward Function
```
fix genuine:   (affected_users/1000) x urgency_weight  (CRITICAL=2.0, HIGH=1.5, MEDIUM=1.0, LOW=0.5)
fix red herring: -0.2
fix symptom:   +0.1 (bug returns)
investigate:   +0.05
ignore correct: +0.1
ignore wrong:  -0.15
noop:          -0.01
End bonus: +0.5 all criticals, +0.2 no red herrings, +0.05 per budget point left
```

## Tasks
| Task | Bugs | Budget | Challenge |
|---|---|---|---|
| task1_easy | 5 | 8 | Clear severity |
| task2_medium | 15 | 12 | Red herrings |
| task3_hard | 25 | 15 | Cascading chains |
""")

                with gr.Tab("Quick Start"):
                    gr.Markdown("""
## Python API

```python
import requests
ENV = "https://argonite3-sre-gym.hf.space"

# Reset
obs = requests.post(f"{ENV}/reset", json={"task_id": "task1_easy"}).json()["observation"]

# Step
result = requests.post(f"{ENV}/step", json={
    "action_type": "fix", "bug_id": "BUG001", "fix_strategy": "hotfix"
}).json()

# Score
state = requests.get(f"{ENV}/state").json()
score = requests.post(f"{ENV}/grader", json={"task_id": "task1_easy", "episode_state": state}).json()["score"]
```

## Endpoints
| Method | Path | Description |
|---|---|---|
| POST | /reset | Start episode |
| POST | /step | Take action |
| GET | /state | Episode state |
| GET | /tasks | All tasks |
| POST | /grader | Score episode |
| POST | /baseline | Run heuristic |
| GET | /health | Health check |
| WS | /ws | WebSocket |
""")

        return demo

    except ImportError:
        return None


# Mount Gradio UI at /web
try:
    import gradio as gr
    from gradio.routes import mount_gradio_app
    _demo = _build_gradio_app()
    if _demo is not None:
        app = mount_gradio_app(app, _demo, path="/web")
        print("[INFO] Gradio UI mounted at /web", flush=True)
except Exception as _gradio_err:
    print(f"[INFO] Gradio UI not mounted: {_gradio_err}", flush=True)
