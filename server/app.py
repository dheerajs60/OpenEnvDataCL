import os
import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from env.cleaner_env import DataCleanerEnv
from env.models import Action, Observation

# Use the official create_app factory to provide all required endpoints:
# /reset, /step, /state, /schema, /metadata, /health, /ws, /mcp
app = create_app(
    DataCleanerEnv,
    Action,
    Observation,
    env_name="data_cleaner_env",
    max_concurrent_envs=4,
)

def main():
    """Main entry point for starting the server."""
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
