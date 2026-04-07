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
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenEnv Data Cleaning Benchmark | Scalar x Meta Hackathon</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                line-height: 1.7;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            
            header {
                text-align: center;
                margin-bottom: 60px;
                padding: 40px 20px;
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 20px;
                border: 1px solid rgba(56, 189, 248, 0.2);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            
            h1 {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #38bdf8 0%, #22c55e 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 20px;
                letter-spacing: -1px;
            }
            
            .tagline {
                font-size: 1.3rem;
                color: #94a3b8;
                font-weight: 300;
                margin-bottom: 30px;
            }
            
            .badges {
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
                margin-top: 25px;
            }
            
            .badge {
                padding: 8px 20px;
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 50px;
                font-size: 0.85rem;
                font-weight: 600;
                color: #38bdf8;
                letter-spacing: 0.5px;
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 30px;
                margin-bottom: 40px;
            }
            
            .card {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 35px;
                border-radius: 16px;
                border: 1px solid rgba(148, 163, 184, 0.1);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #38bdf8, #22c55e);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .card:hover {
                border-color: rgba(56, 189, 248, 0.3);
                transform: translateY(-5px);
                box-shadow: 0 20px 60px rgba(56, 189, 248, 0.15);
            }
            
            .card:hover::before {
                opacity: 1;
            }
            
            h2 {
                font-size: 1.8rem;
                color: #38bdf8;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .icon {
                font-size: 2rem;
            }
            
            .endpoint {
                background: #0f172a;
                padding: 15px;
                border-radius: 10px;
                margin: 12px 0;
                border-left: 3px solid #38bdf8;
                font-family: 'Monaco', 'Courier New', monospace;
                position: relative;
                transition: all 0.2s ease;
            }
            
            .endpoint:hover {
                background: #1e293b;
                border-left-color: #22c55e;
            }
            
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 700;
                margin-right: 10px;
                letter-spacing: 0.5px;
            }
            
            .post { background: #22c55e; color: #000; }
            .get { background: #38bdf8; color: #000; }
            
            .path {
                color: #e2e8f0;
                font-weight: 600;
            }
            
            .description {
                color: #94a3b8;
                font-size: 0.9rem;
                margin-top: 8px;
                font-style: italic;
            }
            
            pre {
                background: #0f172a;
                padding: 20px;
                border-radius: 12px;
                overflow-x: auto;
                border: 1px solid rgba(148, 163, 184, 0.1);
                position: relative;
                margin: 15px 0;
            }
            
            code {
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 0.9rem;
                color: #22c55e;
                line-height: 1.6;
            }
            
            .copy-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(56, 189, 248, 0.2);
                border: 1px solid #38bdf8;
                color: #38bdf8;
                padding: 6px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.8rem;
                transition: all 0.2s ease;
            }
            
            .copy-btn:hover {
                background: #38bdf8;
                color: #000;
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 25px;
            }
            
            .feature {
                padding: 20px;
                background: rgba(56, 189, 248, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(56, 189, 248, 0.1);
                transition: all 0.3s ease;
            }
            
            .feature:hover {
                background: rgba(56, 189, 248, 0.1);
                border-color: rgba(56, 189, 248, 0.3);
            }
            
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .feature-title {
                font-weight: 700;
                color: #38bdf8;
                margin-bottom: 8px;
                font-size: 1.1rem;
            }
            
            .feature-desc {
                color: #94a3b8;
                font-size: 0.9rem;
            }
            
            .stats {
                display: flex;
                justify-content: space-around;
                margin-top: 30px;
                padding: 30px;
                background: rgba(34, 197, 94, 0.05);
                border-radius: 16px;
                border: 1px solid rgba(34, 197, 94, 0.2);
            }
            
            .stat {
                text-align: center;
            }
            
            .stat-value {
                font-size: 3rem;
                font-weight: 800;
                color: #22c55e;
                line-height: 1;
            }
            
            .stat-label {
                color: #94a3b8;
                margin-top: 8px;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            footer {
                text-align: center;
                margin-top: 60px;
                padding: 30px;
                color: #64748b;
                border-top: 1px solid rgba(148, 163, 184, 0.1);
            }
            
            .cta {
                background: linear-gradient(135deg, #38bdf8 0%, #22c55e 100%);
                color: #000;
                padding: 15px 40px;
                border-radius: 50px;
                font-weight: 700;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(56, 189, 248, 0.3);
            }
            
            .cta:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(56, 189, 248, 0.4);
            }
            
            @media (max-width: 768px) {
                h1 { font-size: 2.5rem; }
                .tagline { font-size: 1.1rem; }
                .grid { grid-template-columns: 1fr; }
                .stat-value { font-size: 2rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🧹 OpenEnv Data Cleaning</h1>
                <p class="tagline">Production-Grade RL Benchmark for Tabular Data Cleaning</p>
                <div class="badges">
                    <span class="badge">🏆 SCALAR x META HACKATHON</span>
                    <span class="badge">⚡ FASTAPI POWERED</span>
                    <span class="badge">🤖 LLM-READY</span>
                </div>
            </header>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">3</div>
                    <div class="stat-label">Difficulty Levels</div>
                </div>
                <div class="stat">
                    <div class="stat-value">50+</div>
                    <div class="stat-label">Test Cases</div>
                </div>
                <div class="stat">
                    <div class="stat-value">9</div>
                    <div class="stat-label">Operations</div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2><span class="icon">📡</span> API Endpoints</h2>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <span class="path">/reset</span>
                        <div class="description">Initialize environment with task difficulty</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>
                        <span class="path">/step</span>
                        <div class="description">Execute cleaning action and receive reward</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <span class="path">/state</span>
                        <div class="description">Retrieve current environment metadata</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <span class="path">/health</span>
                        <div class="description">Service health check endpoint</div>
                    </div>
                </div>

                <div class="card">
                    <h2><span class="icon">🚀</span> Quick Start</h2>
                    <pre><button class="copy-btn" onclick="copyCode(this, 0)">📋 Copy</button><code id="code-0"># Reset environment
curl -X POST http://localhost:8000/reset \\
  -H "Content-Type: application/json" \\
  -d '{"difficulty": "hard"}'

# Execute cleaning action
curl -X POST http://localhost:8000/step \\
  -H "Content-Type: application/json" \\
  -d '{
    "operation": "fill_missing",
    "column": "age",
    "value": "median"
  }'</code></pre>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🎯</span> Benchmark Features</h2>
                <div class="features">
                    <div class="feature">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Smart Detection</div>
                        <div class="feature-desc">Automatic error identification in messy datasets</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🎨</div>
                        <div class="feature-title">Schema Normalization</div>
                        <div class="feature-desc">Standardize column names and data types</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📅</div>
                        <div class="feature-title">Date Formatting</div>
                        <div class="feature-desc">Unify mixed datetime formats</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🗑️</div>
                        <div class="feature-title">Deduplication</div>
                        <div class="feature-desc">Remove exact and fuzzy duplicates</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">💉</div>
                        <div class="feature-title">Missing Values</div>
                        <div class="feature-desc">Intelligent null imputation strategies</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Real CRM Data</div>
                        <div class="feature-desc">Production-like customer data chaos</div>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2><span class="icon">🏅</span> Difficulty Levels</h2>
                    <div style="margin-top: 20px;">
                        <div class="feature" style="border-left: 4px solid #22c55e;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.5rem;">🟢</span>
                                <div>
                                    <div class="feature-title">Easy</div>
                                    <div class="feature-desc">Single issue: missing values or duplicates</div>
                                </div>
                            </div>
                        </div>
                        <div class="feature" style="border-left: 4px solid #f59e0b; margin-top: 15px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.5rem;">🟡</span>
                                <div>
                                    <div class="feature-title">Medium</div>
                                    <div class="feature-desc">Multiple issues: dates + nulls + duplicates</div>
                                </div>
                            </div>
                        </div>
                        <div class="feature" style="border-left: 4px solid #ef4444; margin-top: 15px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.5rem;">🔴</span>
                                <div>
                                    <div class="feature-title">Hard</div>
                                    <div class="feature-desc">Real CRM chaos: schema inconsistencies + encodings</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2><span class="icon">⚙️</span> Available Operations</h2>
                    <div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">fill_missing</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">deduplicate</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">rename_column</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">standardize_date</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">convert_dtype</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">drop_column</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">remove_outliers</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">merge_columns</code>
                        <code style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; display: block;">split_column</code>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🎁</span> Reward System</h2>
                <div style="margin-top: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h3 style="color: #22c55e; margin-bottom: 15px;">✅ Positive Rewards</h3>
                            <div style="color: #94a3b8; line-height: 2;">
                                <div>🎯 <strong>+1 to +5</strong> per error fixed</div>
                                <div>⚡ <strong>+0.5</strong> valid transformation</div>
                                <div>🏆 <strong>+10 to +50</strong> completion bonus</div>
                            </div>
                        </div>
                        <div>
                            <h3 style="color: #ef4444; margin-bottom: 15px;">❌ Penalties</h3>
                            <div style="color: #94a3b8; line-height: 2;">
                                <div>⚠️ <strong>-2 to -10</strong> destructive actions</div>
                                <div>📉 <strong>-1</strong> per 1% data loss</div>
                                <div>🚫 <strong>-5</strong> schema violations</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer>
                <p style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px;">
                    Built for <strong style="color: #38bdf8;">Scalar x Meta Hackathon</strong>
                </p>
                <p style="color: #64748b;">
                    Powered by FastAPI • Designed for AI/ML Research • Open Source
                </p>
            </footer>
        </div>

        <script>
            function copyCode(button, codeId) {
                const code = document.getElementById('code-' + codeId).textContent;
                navigator.clipboard.writeText(code).then(() => {
                    button.textContent = '✅ Copied!';
                    setTimeout(() => {
                        button.textContent = '📋 Copy';
                    }, 2000);
                });
            }
        </script>
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
