import sys
from pathlib import Path

# Add the project's "src" directory to the Python path so tests can import mvp.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))