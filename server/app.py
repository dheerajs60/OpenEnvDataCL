from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.environment import DataCleanerEnv
from models import Observation, Action
from server.grader import EasyGrader, MediumGrader, HardGrader

app = FastAPI(title="Data Cleaning Environment")
env_instance = DataCleanerEnv()

# ---------------------------------------------------------------------------
# Grader registry — one instance per task, exposed to the OpenEnv validator
# ---------------------------------------------------------------------------
GRADERS = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}

TASK_META = [
    {
        "id": "easy",
        "description": (
            "Fill missing values in a simple 5-row customer dataset. "
            "Tests dtype-aware imputation (numeric and text columns)."
        ),
        "grader": "server.grader.EasyGrader",
    },
    {
        "id": "medium",
        "description": (
            "Remove exact duplicate rows and normalise mixed date formats "
            "(ISO, US slash, natural-language) to strict YYYY-MM-DD."
        ),
        "grader": "server.grader.MediumGrader",
    },
    {
        "id": "hard",
        "description": (
            "Full CRM data cleanup: fix nulls, remove duplicates, standardise "
            "dates, correct invalid enum values, and rename columns with illegal "
            "characters to valid snake_case identifiers."
        ),
        "grader": "server.grader.HardGrader",
    },
]

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: dict
    done: bool
    info: dict

class GradeRequest(BaseModel):
    state: dict

# ---------------------------------------------------------------------------
# Environment endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = None):
    try:
        diff = req.difficulty if req else None
        return env_instance.reset(diff)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
            "reward": reward.model_dump() if hasattr(reward, "model_dump") else reward.dict(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    try:
        return env_instance.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# OpenEnv-required: task discovery + grader endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    """
    Returns all tasks and their grader class paths.
    Required by the OpenEnv validator (Phase 2).
    """
    return TASK_META


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Return metadata for a single task."""
    task = next((t for t in TASK_META if t["id"] == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@app.post("/tasks/{task_id}/grade")
def grade_task(task_id: str, req: GradeRequest):
    """
    Grade an episode state for a given task.
    The OpenEnv validator calls this to verify each grader works.
    """
    if task_id not in GRADERS:
        raise HTTPException(status_code=404, detail=f"No grader for task '{task_id}'")
    result = GRADERS[task_id].grade(req.state)
    return result


# ---------------------------------------------------------------------------
# Health + UI
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenEnv — Data Cleaning Benchmark</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f1f5f9; line-height: 1.6; }
  a { color: #38bdf8; text-decoration: none; }
  code { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85em; background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #7dd3fc; }
  .wrapper { max-width: 860px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }
  .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; padding: 4px 12px; border-radius: 99px; background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; margin-bottom: 1.25rem; }
  .badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #818cf8; display: inline-block; }
  h1 { font-size: 2rem; font-weight: 700; line-height: 1.25; margin-bottom: 0.75rem; }
  h1 span { color: #38bdf8; }
  .tagline { font-size: 1rem; color: #94a3b8; max-width: 640px; margin-bottom: 1.5rem; }
  .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 2.5rem; }
  .tag { font-size: 11px; padding: 3px 10px; border-radius: 99px; border: 1px solid #334155; color: #64748b; background: #1e293b; font-weight: 500; }
  .divider { height: 1px; background: #1e293b; margin: 2rem 0; }
  .section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1px; background: #1e293b; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; margin-bottom: 2.5rem; }
  .feat { background: #0f172a; padding: 1.1rem 1.25rem; }
  .feat-icon { font-size: 18px; margin-bottom: 8px; }
  .feat h3 { font-size: 14px; font-weight: 600; margin-bottom: 4px; color: #e2e8f0; }
  .feat p { font-size: 13px; color: #64748b; line-height: 1.5; }
  .pipeline { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 2.5rem; }
  .step { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 500; color: #cbd5e1; }
  .arrow { color: #334155; font-size: 18px; font-weight: 300; }
  .endpoints { display: flex; flex-direction: column; gap: 1px; background: #1e293b; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; margin-bottom: 2.5rem; }
  .ep { display: flex; align-items: flex-start; gap: 12px; background: #0f172a; padding: 1rem 1.25rem; }
  .method { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 6px; min-width: 46px; text-align: center; flex-shrink: 0; margin-top: 2px; font-family: monospace; letter-spacing: 0.04em; }
  .post { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .get { background: #0c1a2e; color: #38bdf8; border: 1px solid #0369a1; }
  .ep-path { font-family: monospace; font-size: 14px; font-weight: 600; color: #f1f5f9; }
  .ep-desc { font-size: 12px; color: #64748b; margin-top: 2px; }
  .code-block { background: #0a0f1a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; line-height: 2; margin-bottom: 2.5rem; overflow-x: auto; }
  .op { color: #818cf8; font-weight: 600; }
  .arg { color: #34d399; }
  .cmt { color: #334155; }
  .rtable { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 2.5rem; }
  .rtable th { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #475569; text-align: left; padding: 0 0 10px; border-bottom: 1px solid #1e293b; }
  .rtable td { padding: 10px 0; border-bottom: 1px solid #0f172a; color: #94a3b8; vertical-align: middle; }
  .pill { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 99px; font-family: monospace; }
  .pos { background: #052e16; color: #4ade80; }
  .neg { background: #1c0a0a; color: #f87171; }
  .neutral { background: #1e293b; color: #94a3b8; }
  .score-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-bottom: 2.5rem; }
  .score-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.1rem; }
  .score-card .lbl { font-size: 12px; color: #64748b; margin-bottom: 6px; }
  .score-card .val { font-size: 24px; font-weight: 700; color: #38bdf8; }
  .curl { background: #0a0f1a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.9; overflow-x: auto; }
  .cmd { color: #94a3b8; }
  .flag { color: #818cf8; }
  .str { color: #34d399; }
  .cm { color: #334155; }
</style>
</head>
<body>
<div class="wrapper">
  <div class="badge">OpenEnv Challenge · AgentX-AgentBeats 2026</div>
  <h1>🧹 OpenEnv <span>/</span> Data Cleaning Benchmark</h1>
  <p class="tagline">A multi-step RL environment for evaluating LLM agents on structured CRM data cleaning — with dynamic reward shaping and three difficulty tiers.</p>
  <div class="tag-row">
    <span class="tag">reinforcement-learning</span>
    <span class="tag">llm-agents</span>
    <span class="tag">data-cleaning</span>
    <span class="tag">crm</span>
    <span class="tag">openenv</span>
    <span class="tag">fastapi</span>
    <span class="tag">benchmark</span>
  </div>

  <p class="section-label">What gets tested</p>
  <div class="grid">
    <div class="feat"><div class="feat-icon">🔧</div><h3>Missing value repair</h3><p>Dtype-aware fill — numeric, datetime, and text columns handled correctly.</p></div>
    <div class="feat"><div class="feat-icon">🗂️</div><h3>Duplicate removal</h3><p>Exact-match deduplication without destroying valid unique records.</p></div>
    <div class="feat"><div class="feat-icon">📅</div><h3>Date normalisation</h3><p>Unify ISO, US, and natural-language formats into strict YYYY-MM-DD.</p></div>
    <div class="feat"><div class="feat-icon">🔡</div><h3>Text standardisation</h3><p>Strip whitespace, fix casing — title-case all name and label columns.</p></div>
    <div class="feat"><div class="feat-icon">🏷️</div><h3>Category correction</h3><p>Replace invalid enum values with valid substitutes via fix_category.</p></div>
    <div class="feat"><div class="feat-icon">📋</div><h3>Schema repair</h3><p>Rename columns with spaces or hyphens to clean snake_case identifiers.</p></div>
  </div>

  <p class="section-label">Tasks &amp; Graders</p>
  <table class="rtable">
    <thead><tr><th>Task ID</th><th>Grader Class</th><th>Primary Challenge</th></tr></thead>
    <tbody>
      <tr><td><code>easy</code></td><td><code>EasyGrader</code></td><td>Null imputation (60% weight)</td></tr>
      <tr><td><code>medium</code></td><td><code>MediumGrader</code></td><td>Dedup + date normalisation (60% weight)</td></tr>
      <tr><td><code>hard</code></td><td><code>HardGrader</code></td><td>All five dimensions equally weighted</td></tr>
    </tbody>
  </table>

  <p class="section-label">Agent loop</p>
  <div class="pipeline">
    <span class="step">POST /reset</span>
    <span class="arrow">→</span>
    <span class="step">Read Observation</span>
    <span class="arrow">→</span>
    <span class="step">POST /step</span>
    <span class="arrow">→</span>
    <span class="step">Receive Reward</span>
    <span class="arrow">→</span>
    <span class="step">Repeat ↺ / stop</span>
  </div>

  <p class="section-label">API endpoints</p>
  <div class="endpoints">
    <div class="ep"><span class="method post">POST</span><div><div class="ep-path">/reset</div><div class="ep-desc">Load a task: <code>"easy"</code>, <code>"medium"</code>, or <code>"hard"</code>.</div></div></div>
    <div class="ep"><span class="method post">POST</span><div><div class="ep-path">/step</div><div class="ep-desc">Apply a cleaning action. Returns Observation, Reward, done flag.</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/state</div><div class="ep-desc">Current episode metadata and progress score.</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/tasks</div><div class="ep-desc">List all 3 tasks with their grader class paths (OpenEnv validator).</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/tasks/{task_id}</div><div class="ep-desc">Get metadata for a single task.</div></div></div>
    <div class="ep"><span class="method post">POST</span><div><div class="ep-path">/tasks/{task_id}/grade</div><div class="ep-desc">Grade an episode state for a task (OpenEnv validator).</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/health</div><div class="ep-desc">Liveness check — returns <code>{"status": "ok"}</code>.</div></div></div>
  </div>

  <p class="section-label">Action space</p>
  <div class="code-block">
    <span class="op">fill_missing</span>     <span class="arg">column</span>=<span class="str">"age"</span>          <span class="arg">value</span>=<span class="str">"0"</span>            <span class="cmt"># fills NaN with dtype-cast value</span><br>
    <span class="op">remove_duplicates</span>                                                   <span class="cmt"># drops exact duplicate rows</span><br>
    <span class="op">standardize_date</span>  <span class="arg">column</span>=<span class="str">"Signup-Date"</span>                        <span class="cmt"># → YYYY-MM-DD</span><br>
    <span class="op">normalize_text</span>    <span class="arg">column</span>=<span class="str">"First Name"</span>                         <span class="cmt"># strip + title-case</span><br>
    <span class="op">rename_column</span>     <span class="arg">column</span>=<span class="str">"Cst_ID"</span>       <span class="arg">value</span>=<span class="str">"customer_id"</span>   <span class="cmt"># schema repair</span><br>
    <span class="op">fix_category</span>      <span class="arg">column</span>=<span class="str">"status"</span>  <span class="arg">value</span>=<span class="str">"UNKNOWN_STATUS:ACTIVE"</span><br>
    <span class="op">stop</span>                                                                  <span class="cmt"># terminate when clean</span>
  </div>

  <p class="section-label">Reward shaping</p>
  <table class="rtable">
    <thead><tr><th>Event</th><th>Reward</th></tr></thead>
    <tbody>
      <tr><td>Issue resolved (per issue fixed)</td><td><span class="pill pos">+0.25 each</span></td></tr>
      <tr><td>Exact duplicates removed</td><td><span class="pill pos">+0.25</span></td></tr>
      <tr><td>Valid stop — all issues resolved</td><td><span class="pill pos">+0.15</span></td></tr>
      <tr><td>Minor state change</td><td><span class="pill neutral">+0.01</span></td></tr>
      <tr><td>Repeated / useless action</td><td><span class="pill neg">+0.01</span></td></tr>
      <tr><td>Premature stop</td><td><span class="pill neg">+0.01</span></td></tr>
      <tr><td>Excessive row loss (&gt;50%)</td><td><span class="pill neg">−0.50 (final)</span></td></tr>
    </tbody>
  </table>
</div>
</body>
</html>"""


import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
