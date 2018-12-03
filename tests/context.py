# Connecting parent dir to lower file
import os
import sys

# Entering path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src
