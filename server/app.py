from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from env.cleaner_env import DataCleanerEnv
from env.models import Observation, Action

app = FastAPI(title="Data Cleaning Environment")
env_instance = DataCleanerEnv()

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: dict
    done: bool
    info: dict
@app.post("/reset", response_model= Observation)
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
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    try:
        return env_instance.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

  /* hero */
  .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; padding: 4px 12px; border-radius: 99px; background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; margin-bottom: 1.25rem; }
  .badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #818cf8; display: inline-block; }
  h1 { font-size: 2rem; font-weight: 700; line-height: 1.25; margin-bottom: 0.75rem; }
  h1 span { color: #38bdf8; }
  .tagline { font-size: 1rem; color: #94a3b8; max-width: 640px; margin-bottom: 1.5rem; }
  .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 2.5rem; }
  .tag { font-size: 11px; padding: 3px 10px; border-radius: 99px; border: 1px solid #334155; color: #64748b; background: #1e293b; font-weight: 500; }
  .divider { height: 1px; background: #1e293b; margin: 2rem 0; }

  /* section titles */
  .section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 1rem; }

  /* feature cards */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1px; background: #1e293b; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; margin-bottom: 2.5rem; }
  .feat { background: #0f172a; padding: 1.1rem 1.25rem; }
  .feat-icon { font-size: 18px; margin-bottom: 8px; }
  .feat h3 { font-size: 14px; font-weight: 600; margin-bottom: 4px; color: #e2e8f0; }
  .feat p { font-size: 13px; color: #64748b; line-height: 1.5; }

  /* pipeline */
  .pipeline { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 2.5rem; }
  .step { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 500; color: #cbd5e1; }
  .arrow { color: #334155; font-size: 18px; font-weight: 300; }

  /* endpoints */
  .endpoints { display: flex; flex-direction: column; gap: 1px; background: #1e293b; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; margin-bottom: 2.5rem; }
  .ep { display: flex; align-items: flex-start; gap: 12px; background: #0f172a; padding: 1rem 1.25rem; }
  .method { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 6px; min-width: 46px; text-align: center; flex-shrink: 0; margin-top: 2px; font-family: monospace; letter-spacing: 0.04em; }
  .post { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .get { background: #0c1a2e; color: #38bdf8; border: 1px solid #0369a1; }
  .ep-path { font-family: monospace; font-size: 14px; font-weight: 600; color: #f1f5f9; }
  .ep-desc { font-size: 12px; color: #64748b; margin-top: 2px; }

  /* action space */
  .code-block { background: #0a0f1a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; line-height: 2; margin-bottom: 2.5rem; overflow-x: auto; }
  .op { color: #818cf8; font-weight: 600; }
  .arg { color: #34d399; }
  .cmt { color: #334155; }

  /* reward table */
  .rtable { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 2.5rem; }
  .rtable th { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #475569; text-align: left; padding: 0 0 10px; border-bottom: 1px solid #1e293b; }
  .rtable td { padding: 10px 0; border-bottom: 1px solid #0f172a; color: #94a3b8; vertical-align: middle; }
  .pill { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 99px; font-family: monospace; }
  .pos { background: #052e16; color: #4ade80; }
  .neg { background: #1c0a0a; color: #f87171; }
  .neutral { background: #1e293b; color: #94a3b8; }

  /* score cards */
  .score-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-bottom: 2.5rem; }
  .score-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.1rem; }
  .score-card .lbl { font-size: 12px; color: #64748b; margin-bottom: 6px; }
  .score-card .val { font-size: 24px; font-weight: 700; color: #38bdf8; }

  /* curl */
  .curl { background: #0a0f1a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.9; overflow-x: auto; }
  .cmd { color: #94a3b8; }
  .flag { color: #818cf8; }
  .str { color: #34d399; }
  .cm { color: #334155; }
</style>
</head>
<body>
<div class="wrapper">

  <div class="badge">Scalar &times; Meta Hackathon 2026</div>
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
    <div class="feat"><div class="feat-icon">📅</div><h3>Date normalization</h3><p>Unify ISO, US, and natural-language formats into strict YYYY-MM-DD.</p></div>
    <div class="feat"><div class="feat-icon">🔡</div><h3>Text standardization</h3><p>Strip whitespace, fix casing — title-case all name and label columns.</p></div>
    <div class="feat"><div class="feat-icon">🏷️</div><h3>Category correction</h3><p>Replace invalid enum values with valid substitutes via fix_category.</p></div>
    <div class="feat"><div class="feat-icon">📋</div><h3>Schema repair</h3><p>Rename columns with spaces or hyphens to clean snake_case identifiers.</p></div>
  </div>

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
  <p style="font-size:13px;color:#64748b;margin-bottom:2.5rem;">Each episode runs up to 20 steps. The agent receives a structured <code>Observation</code> (row preview, schema, detected issues, step count) and emits an <code>Action</code> (operation + optional column + value). The grader returns a shaped reward per step and a final <code>[0, 1]</code> score on episode end.</p>

  <p class="section-label">API endpoints</p>
  <div class="endpoints">
    <div class="ep"><span class="method post">POST</span><div><div class="ep-path">/reset</div><div class="ep-desc">Load a task: <code>"easy"</code>, <code>"medium"</code>, or <code>"hard"</code>. Returns initial Observation with schema and detected issues.</div></div></div>
    <div class="ep"><span class="method post">POST</span><div><div class="ep-path">/step</div><div class="ep-desc">Apply a cleaning action. Returns Observation, shaped Reward, done flag, and final score info on termination.</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/state</div><div class="ep-desc">Current episode metadata — difficulty level, step count, done flag, dataframe shape.</div></div></div>
    <div class="ep"><span class="method get">GET</span><div><div class="ep-path">/health</div><div class="ep-desc">Liveness check — returns <code>{"status": "ok"}</code>.</div></div></div>
  </div>

  <p class="section-label">Action space</p>
  <div class="code-block">
    <span class="op">fill_missing</span>     <span class="arg">column</span>=<span class="str">"age"</span>          <span class="arg">value</span>=<span class="str">"0"</span>            <span class="cmt"># fills NaN with dtype-cast value</span><br>
    <span class="op">remove_duplicates</span>                                                   <span class="cmt"># drops exact duplicate rows</span><br>
    <span class="op">standardize_date</span>  <span class="arg">column</span>=<span class="str">"Signup-Date"</span>                        <span class="cmt"># → YYYY-MM-DD</span><br>
    <span class="op">normalize_text</span>    <span class="arg">column</span>=<span class="str">"First Name"</span>                         <span class="cmt"># strip + title-case</span><br>
    <span class="op">rename_column</span>     <span class="arg">column</span>=<span class="str">"Cst_ID"</span>       <span class="arg">value</span>=<span class="str">"customer_id"</span>   <span class="cmt"># schema repair</span><br>
    <span class="op">fix_category</span>      <span class="arg">column</span>=<span class="str">"status_cat"</span>  <span class="arg">value</span>=<span class="str">"UNKNOWN_STATUS:active"</span><br>
    <span class="op">stop</span>                                                                  <span class="cmt"># terminate when clean</span>
  </div>

  <p class="section-label">Reward shaping</p>
  <table class="rtable">
    <thead><tr><th>Event</th><th>Reward</th></tr></thead>
    <tbody>
      <tr><td>Issue resolved (per issue fixed)</td><td><span class="pill pos">+0.25 each</span></td></tr>
      <tr><td>Exact duplicates removed</td><td><span class="pill pos">+0.25</span></td></tr>
      <tr><td>Valid stop — all issues resolved</td><td><span class="pill pos">+0.10</span></td></tr>
      <tr><td>Minor state change</td><td><span class="pill neutral">+0.01</span></td></tr>
      <tr><td>Repeated / useless action</td><td><span class="pill neg">−0.10</span></td></tr>
      <tr><td>Consecutive repeat penalty</td><td><span class="pill neg">−0.05</span></td></tr>
      <tr><td>Invalid operation</td><td><span class="pill neg">−0.10</span></td></tr>
      <tr><td>Premature stop</td><td><span class="pill neg">−0.20</span></td></tr>
      <tr><td>Excessive row loss (&gt;50%)</td><td><span class="pill neg">−0.50</span></td></tr>
    </tbody>
  </table>

  <p class="section-label">Final score breakdown</p>
  <div class="score-grid">
    <div class="score-card"><div class="lbl">Null-free</div><div class="val">25%</div></div>
    <div class="score-card"><div class="lbl">No duplicates</div><div class="val">25%</div></div>
    <div class="score-card"><div class="lbl">Schema valid</div><div class="val">20%</div></div>
    <div class="score-card"><div class="lbl">Dates clean</div><div class="val">15%</div></div>
    <div class="score-card"><div class="lbl">Categories valid</div><div class="val">15%</div></div>
  </div>
  <p style="font-size:13px;color:#64748b;margin-bottom:2.5rem;">Score clamped to [0.0, 1.0]. Penalized by −0.50 if valid non-duplicate rows are lost during the episode.</p>

  <p class="section-label">Quick start</p>
  <div class="curl">
    <span class="cm"># 1. Start a hard episode</span><br>
    <span class="cmd">curl -X POST /reset</span> <span class="flag">-H</span> <span class="str">"Content-Type: application/json"</span> <span class="flag">-d</span> <span class="str">'{"difficulty":"hard"}'</span><br><br>
    <span class="cm"># 2. Apply a cleaning action</span><br>
    <span class="cmd">curl -X POST /step</span>  <span class="flag">-H</span> <span class="str">"Content-Type: application/json"</span> <span class="flag">-d</span> <span class="str">'{"operation":"remove_duplicates"}'</span><br><br>
    <span class="cm"># 3. Stop when all issues are resolved</span><br>
    <span class="cmd">curl -X POST /step</span>  <span class="flag">-d</span> <span class="str">'{"operation":"stop"}'</span>
  </div>

</div>
</body>
</html>
"""
@app.get("/health")
def health():
    return {"status": "ok"}
import uvicorn
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
