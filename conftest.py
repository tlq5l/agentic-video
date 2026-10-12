import sys
from pathlib import Path

# Add repo root to sys.path so tests can import src directly
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
