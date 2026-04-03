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
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Data Cleaning OpenEnv",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state"
        }
    }
@app.get("/health")
def health():
    return {"status": "ok"}
import uvicorn
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
