# OpenEnv client entrypoint
import sys
from pathlib import Path

# Add project root to sys.path for internal imports
sys.path.append(str(Path(__file__).parent))

try:
    from inference import *
except ImportError:
    pass
