from fastapi import FastAPI, HTTPException
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
    return """
    <html>
    <head>
        <title>🧹 OpenEnv Data Cleaning</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.6;
                background: #0f172a;
                color: #f8fafc;
            }
            h1 { color: #38bdf8; }
            h2 { color: #22c55e; }
            code {
                background: #1e293b;
                padding: 4px 8px;
                border-radius: 6px;
            }
            pre {
                background: #1e293b;
                padding: 14px;
                border-radius: 10px;
                overflow-x: auto;
            }
            .box {
                background: #111827;
                padding: 20px;
                border-radius: 14px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h1>🧹 OpenEnv Data Cleaning Benchmark</h1>
        <p>
            A real-world multi-step benchmark for evaluating LLM agents on
            <b>tabular CRM data cleaning workflows</b>.
        </p>

        <div class="box">
            <h2>📡 API Endpoints</h2>
            <ul>
                <li><code>POST /reset</code> → load easy / medium / hard task</li>
                <li><code>POST /step</code> → apply cleaning action</li>
                <li><code>GET /state</code> → current environment metadata</li>
                <li><code>GET /health</code> → service health check</li>
            </ul>
        </div>

        <div class="box">
            <h2>🧪 Example</h2>
            <pre>curl -X POST /reset -d '{"difficulty":"hard"}'</pre>
        </div>

        <div class="box">
            <h2>🏆 Benchmark Features</h2>
            <ul>
                <li>3 difficulty levels</li>
                <li>dynamic reward shaping</li>
                <li>duplicate removal</li>
                <li>missing value repair</li>
                <li>date normalization</li>
                <li>real-world CRM cleaning</li>
            </ul>
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
