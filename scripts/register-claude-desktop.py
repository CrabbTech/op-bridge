#!/usr/bin/env python3
"""Register op-bridge with Claude Desktop in a way that survives the app.

Claude Desktop keeps claude_desktop_config.json in memory and writes that copy back on quit and on
preference changes, so an entry added while the app runs is lost. This script hands the work to a
detached worker that quits the app, waits for it to exit, merges the op-bridge entry into the file
(keeping everything else, with a backup beside it) and relaunches the app. Because the worker is
detached, the script can be run from a terminal inside Claude Desktop as well as from Terminal.app.

    python3 scripts/register-claude-desktop.py              # quit Claude, write, relaunch
    python3 scripts/register-claude-desktop.py --no-restart # only write; refuses while Claude runs

The worker logs to ~/Music/op-bridge/register-claude-desktop.log."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

CONFIG = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
APP_EXE = "/Applications/Claude.app/Contents/MacOS/Claude"
LOG = os.path.expanduser("~/Music/op-bridge/register-claude-desktop.log")
ENTRY = {"command": "/opt/homebrew/bin/uv", "args": ["--directory", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run", "op-bridge", "serve"]}


def claude_running() -> bool:
    """pgrep sees nothing from inside the app's own shell; ps does."""
    out = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True)
    return any(line.strip() == APP_EXE for line in out.stdout.splitlines())


def write_entry() -> dict:
    d = {}
    if os.path.exists(CONFIG):
        shutil.copyfile(CONFIG, CONFIG + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        with open(CONFIG) as f:
            d = json.load(f)
    d.setdefault("mcpServers", {})["op-bridge"] = ENTRY
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CONFIG), prefix=".claude_desktop_config.")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG)
    with open(CONFIG) as f:
        return json.load(f)["mcpServers"]["op-bridge"]


def worker() -> int:
    def log(msg: str) -> None:
        with open(LOG, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    log("worker started")
    if claude_running():
        subprocess.run(["osascript", "-e", 'quit app "Claude"'], check=False)
        for _ in range(120):
            if not claude_running():
                break
            time.sleep(0.5)
        else:
            log("Claude Desktop did not quit within 60 s; giving up (close it by hand and rerun)")
            return 1
        log("Claude Desktop quit")
        time.sleep(2.0)  # let its final config write land before ours
    log("written: " + json.dumps(write_entry()))
    subprocess.run(["open", "-a", "Claude"], check=False)
    log("Claude Desktop relaunched")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-restart", action="store_true", help="only write the entry; refuse while Claude Desktop is running")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker()
    if a.no_restart:
        if claude_running():
            print("Claude Desktop is running; quit it first, or drop --no-restart", file=sys.stderr)
            return 1
        print("written:", json.dumps(write_entry()))
        return 0
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    logf = open(LOG, "a")
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"], stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
                     start_new_session=True, close_fds=True)
    print("Claude Desktop will quit now, get the op-bridge entry, and relaunch in a few seconds." if claude_running()
          else "Claude Desktop is not running: writing the entry and launching it.")
    print(f"progress: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
