"""PAC's optional assistant, installed only where the GPU can serve its model: checks this
machine's GPU and writes compose's .env, before `docker compose up -d --build`.

    python3 install_assistant.py            # asks, then checks the GPU
    python3 install_assistant.py --without  # PAC alone
    python3 install_assistant.py --remote http://gpu-host:8001/v1 --model Qwen/Qwen3-8B-FP8
"""

import sys
from pathlib import Path

# The standard library only: runs before PAC is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from masw.gpu import main

if __name__ == "__main__":
    sys.exit(main())
