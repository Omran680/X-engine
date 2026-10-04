import os
import sys
import tempfile

# Make the repo importable and keep tests hermetic: logs/models never touch the project tree.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
