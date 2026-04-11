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
    """Return a dark-themed landing page."""
    tasks = list_tasks()

    task_cards_html = ""
    for t in tasks:
        task_cards_html += f"""
        <div class="card task-card">
            <div class="task-difficulty {t.difficulty}">{t.difficulty.upper()}</div>
            <h3>{t.task_id.replace("_", " ").title()}</h3>
            <p>{t.description}</p>
            <div class="task-meta">
                <span>Budget: <strong>{t.budget} pts</strong></span>
                &nbsp;|&nbsp;
                <span>Max Steps: <strong>{t.max_steps}</strong></span>
            </div>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SRE-Gym | Production Incident Triage</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0d1117; --card-bg: #161b22; --text: #c9d1d9;
                --text-muted: #8b949e; --accent: #00ff88;
                --border: #30363d; --easy: #3fb950;
                --medium: #d29922; --hard: #f85149;
            }}
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: 'Inter', sans-serif;
                   background-color: var(--bg); color: var(--text);
                   display: flex; flex-direction: column; align-items: center;
                   min-height: 100vh; padding: 2rem; }}
            .container {{ max-width: 1000px; width: 100%; }}
            header {{ text-align: center; margin-bottom: 3rem; }}
            h1 {{ font-size: 3rem; margin: 0; color: #fff; letter-spacing: -1px; }}
            .subtitle {{ font-size: 1.2rem; color: var(--text-muted); margin-top: 0.5rem; }}
            .status-badge {{ display: inline-flex; align-items: center;
                background: var(--card-bg); padding: 0.5rem 1rem;
                border-radius: 20px; border: 1px solid var(--border);
                margin-top: 1.5rem; font-size: 0.9rem; font-weight: 600; }}
            .status-dot {{ width: 8px; height: 8px; background-color: var(--accent);
                border-radius: 50%; margin-right: 8px;
                box-shadow: 0 0 10px var(--accent); animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem; margin-bottom: 4rem; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border);
                border-radius: 12px; padding: 1.5rem;
                transition: transform 0.2s, border-color 0.2s; }}
            .card:hover {{ transform: translateY(-4px); border-color: var(--accent); }}
            .stat-card h4 {{ margin: 0; color: var(--text-muted); font-size: 0.8rem;
                text-transform: uppercase; letter-spacing: 1px; }}
            .stat-card .value {{ font-size: 1.8rem; font-weight: 700;
                color: var(--accent); margin-top: 0.5rem; }}
            h2 {{ font-size: 1.8rem; margin-bottom: 1.5rem;
                border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
            .task-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem; margin-bottom: 4rem; }}
            .task-card {{ position: relative; overflow: hidden; }}
            .task-difficulty {{ position: absolute; top: 0; right: 0; padding: 0.4rem 0.8rem;
                font-size: 0.7rem; font-weight: 800; border-bottom-left-radius: 12px; color: #fff; }}
            .task-difficulty.easy {{ background: var(--easy); }}
            .task-difficulty.medium {{ background: var(--medium); }}
            .task-difficulty.hard {{ background: var(--hard); }}
            .task-card h3 {{ margin-top: 0.5rem; font-size: 1.3rem; }}
            .task-card p {{ color: var(--text-muted); line-height: 1.5; font-size: 0.95rem; }}
            .task-meta {{ margin-top: 1.5rem; padding-top: 1rem;
                border-top: 1px solid var(--border); font-size: 0.9rem; }}
            footer {{ text-align: center; margin-top: auto; padding-top: 2rem;
                border-top: 1px solid var(--border); width: 100%; color: var(--text-muted); }}
            .links {{ margin-bottom: 1rem; }}
            .links a {{ color: var(--accent); text-decoration: none; margin: 0 1rem;
                font-weight: 600; transition: opacity 0.2s; }}
            .links a:hover {{ opacity: 0.8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🚨 SRE-Gym</h1>
                <div class="subtitle">Production Incident Triage Environment for AI Agents</div>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    Live — OpenEnv Compliant
                </div>
            </header>
            <div class="stats-grid">
                <div class="card stat-card">
                    <h4>Available Tasks</h4><div class="value">3 Tasks</div>
                </div>
                <div class="card stat-card">
                    <h4>Grader Score Range</h4><div class="value">0.0 – 1.0</div>
                </div>
                <div class="card stat-card">
                    <h4>Unique Mechanics</h4><div class="value">5 Mechanics</div>
                </div>
            </div>
            <h2>Environment Tasks</h2>
            <div class="task-grid">
                {task_cards_html}
            </div>
            <footer>
                <div class="links">
                    <a href="/web">🎛 Live Demo</a>
                    <a href="/docs">API Docs</a>
                    <a href="/health">Health Check</a>
                    <a href="/tasks">Tasks + Schema</a>
                </div>
                <p>&copy; 2026 SRE-Gym — OpenEnv Hackathon Submission</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html_content


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

def _build_gradio_app():
    """Build the Gradio live demo UI for SRE-Gym."""
    try:
        import gradio as gr

        def run_demo_episode(task_id: str) -> str:
            """Run the heuristic agent on a task and stream the log."""
            try:
                from src.graders import run_grader as _grade
                env = SREGymEnvironment(task_id)
                reset_result = env.reset()
                obs = reset_result.observation
                lines = [
                    f"🚨 SRE-Gym Demo — {task_id}",
                    f"Goal: {obs.goal}",
                    f"Budget: {obs.budget_remaining}  |  Bugs in queue: {len(obs.bugs)}",
                    "─" * 52,
                ]
                done = obs.done
                step = 0
                while not done and step < 25:
                    action = _heuristic_action(obs)
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    step += 1
                    bug_str = action.bug_id or ""
                    lines.append(
                        f"Step {step:2d}: {action.action_type}({bug_str})"
                        f"  →  reward={result.reward:+.2f}"
                        f"  |  budget={obs.budget_remaining}"
                    )
                    if result.info.get("result"):
                        lines.append(f"         ↳ {result.info['result'][:80]}")
                state = env.state()
                score = _grade(task_id, state)
                lines += [
                    "─" * 52,
                    f"✅ Done in {step} steps",
                    f"📊 Final score : {score:.3f} / 1.0",
                    f"💰 Total reward: {state['total_reward']:.3f}",
                    f"🐛 Bugs fixed  : {len(state['fixed_bugs'])} / {len(state['bugs'])}",
                ]
                return "\n".join(lines)
            except Exception as exc:
                return f"Error running demo: {exc}"

        def get_env_status() -> str:
            try:
                tasks = list_tasks()
                lines = ["## 🟢 SRE-Gym is Live\n"]
                lines.append("| Task | Difficulty | Budget | Max Steps |")
                lines.append("|------|-----------|--------|-----------|")
                for t in tasks:
                    lines.append(
                        f"| `{t.task_id}` | **{t.difficulty}** | {t.budget} pts | {t.max_steps} |"
                    )
                lines += [
                    "\n**5 Unique Mechanics:**",
                    "1. 🔗 Cascading failure graphs — fix wrong one = symptom returns",
                    "2. 📈 Dynamic spread — affected_users grows every step",
                    "3. 💰 Fixed action budget — must plan, not just react",
                    "4. 🎭 Misleading red herrings — CRITICAL bugs that are noise",
                    "5. 🔍 Investigate-before-fix — information asymmetry",
                ]
                return "\n".join(lines)
            except Exception as exc:
                return f"Status error: {exc}"

        with gr.Blocks(
            title="SRE-Gym | Production Incident Triage",
            theme=gr.themes.Base(),
        ) as demo:
            gr.Markdown(
                "# 🚨 SRE-Gym\n"
                "**Production Incident Triage Environment for AI Agents**\n\n"
                "> OpenEnv-compliant RL environment — agents learn to triage bugs like real SRE engineers."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Environment Status")
                    status_md = gr.Markdown(get_env_status())
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    refresh_btn.click(fn=get_env_status, outputs=status_md)

                with gr.Column(scale=1):
                    gr.Markdown("## Run Demo Episode")
                    task_dd = gr.Dropdown(
                        choices=["task1_easy", "task2_medium", "task3_hard"],
                        value="task1_easy",
                        label="Select Task",
                    )
                    run_btn = gr.Button("▶ Run Heuristic Agent", variant="primary")
                    episode_log = gr.Textbox(
                        label="Episode Log",
                        lines=22,
                        placeholder="Click 'Run Heuristic Agent' to watch a demo episode...",
                    )
                    run_btn.click(fn=run_demo_episode, inputs=task_dd, outputs=episode_log)

            gr.Markdown("""
## API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/reset` | `{"task_id": "task1_easy"}` | Start new episode |
| POST | `/step` | `{"action_type": "fix", "bug_id": "BUG001", "fix_strategy": "hotfix"}` | Take action |
| GET | `/state` | — | Current episode state |
| GET | `/tasks` | — | All tasks + action schema |
| POST | `/grader` | `{"task_id": "...", "episode_state": {...}}` | Score episode |
| POST | `/baseline` | `{"task_id": "task1_easy"}` | Run heuristic agent |
| GET | `/health` | — | Health check |
| GET | `/docs` | — | Swagger UI |

**Action types:** `investigate` (cost 1) · `fix` (cost 2) · `escalate` (cost 1) · `ignore` (free) · `noop` (free)
""")

        return demo

    except ImportError:
        return None


# Mount Gradio UI at /web if gradio is installed
try:
    import gradio as gr
    from gradio.routes import mount_gradio_app
    _demo = _build_gradio_app()
    if _demo is not None:
        app = mount_gradio_app(app, _demo, path="/web")
        print("[INFO] Gradio UI mounted at /web", flush=True)
except Exception as _gradio_err:
    print(f"[INFO] Gradio UI not mounted: {_gradio_err}", flush=True)
