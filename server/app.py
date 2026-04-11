try:
    from openenv.core.env_server.http_server import create_app
    from ..models import Action, Observation
    from ..env.cleaner_env import DataCleanerEnv
except (ImportError, ValueError):
    try:
        from openenv.core.env_server.http_server import create_app
        from models import Action, Observation
        from env.cleaner_env import DataCleanerEnv
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from openenv.core.env_server.http_server import create_app
        from env.models import Action, Observation
        from env.cleaner_env import DataCleanerEnv

# Create the app with web interface and README integration
app = create_app(
    DataCleanerEnv,
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
