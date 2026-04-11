import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from env.models import Action, Observation
from env.cleaner_env import DataCleanerEnv

_global_env = DataCleanerEnv()

app = create_app(
    lambda: _global_env,
    Action,
    Observation,
    env_name="data_cleaner_env",
    max_concurrent_envs=1,
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
