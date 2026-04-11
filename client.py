from openenv.core.client import EnvClient

try:
    from .models import Action, Observation
except ImportError:
    from models import Action, Observation

DataCleanerClient = EnvClient[Action, Observation, dict]
