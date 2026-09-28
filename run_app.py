"""Start the web app.

    python run_app.py              # this computer only
    python run_app.py --port 8600  # another port

Who can connect depends on one setting, APP_PASSWORD in .env:

  - not set: only this computer (http://localhost:8501). Nobody else on the
    network can even reach the page.
  - set:     everyone on the office network who knows the address AND the
    password. The launcher prints the address to give colleagues.

That rule is here, not in app.py, because the network address has to be
chosen before Streamlit starts -- and an app open to the network with no
password would let anyone on the network run analyses on your API key.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lan_address() -> str | None:
    """This computer's address on the local network, e.g. 192.168.1.23.

    Connecting a UDP socket sends nothing; it only makes the system pick the
    network interface it would use, and that interface's address is the one
    colleagues can reach.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the CSV Insight Agent web app.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed. Run:  pip install -r requirements.txt", file=sys.stderr)
        return 2
    try:
        from src.config import settings  # also reads .env
    except ValueError as exc:
        print(f"error in the settings (.env): {exc}", file=sys.stderr)
        return 2

    import main as cli

    problem = cli.credential_problem(settings)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    password = os.environ.get("APP_PASSWORD", "").strip()
    if password and len(password) < 8:
        print("APP_PASSWORD is shorter than 8 characters. Choose a longer one.", file=sys.stderr)
        return 2

    address = "0.0.0.0" if password else "127.0.0.1"
    print(f"CSV Insight Agent {cli.VERSION}  --  {settings.provider} / {settings.resolved_model()}")
    print(f"  on this computer:  http://localhost:{args.port}")
    if password:
        lan = lan_address()
        if lan:
            print(f"  for colleagues:    http://{lan}:{args.port}   (they need APP_PASSWORD)")
        print("  Windows may ask whether Python may use the network: allow it for")
        print("  PRIVATE networks only (office/home), not for public ones.")
    else:
        print("  Colleagues cannot connect: set APP_PASSWORD in .env to allow it.")
    print("  Stop the app with Ctrl+C in this window.\n")

    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.address", address,
        "--server.port", str(args.port),
        "--browser.gatherUsageStats", "false",
    ]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
