#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
"""Pull survey data from the production server to ./data/

Usage:
    uv run scripts/pull_data.py <user>          # explicit user
    USER=bkrijgsman uv run scripts/pull_data.py # via env var
"""

import os
import subprocess
import sys
from pathlib import Path

HOST = "acbcproduction.adaptivechoice.src.surf-hosted.nl"
REMOTE_PATH = "~/data/acbc-storage/acbc/"
LOCAL_PATH = Path("data")

if len(sys.argv) > 1:
    user = sys.argv[1]
elif "USER" in os.environ:
    user = os.environ["USER"]
else:
    print("Error: provide username as argument or set USER env var")
    print(f"  uv run scripts/pull_data.py <user>")
    sys.exit(1)

LOCAL_PATH.mkdir(exist_ok=True)

result = subprocess.run(
    ["rsync", "-avz", "--progress", f"{user}@{HOST}:{REMOTE_PATH}", str(LOCAL_PATH) + "/"],
    check=False,
)
sys.exit(result.returncode)
