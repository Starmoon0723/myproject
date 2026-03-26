import logging
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONLINE_RUN_PATH = os.path.join(ROOT_DIR, "src_online", "models", "run.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    if not os.path.exists(ONLINE_RUN_PATH):
        raise FileNotFoundError(f"Online run script not found: {ONLINE_RUN_PATH}")

    cmd = [sys.executable, ONLINE_RUN_PATH] + sys.argv[1:]
    logger.info("Forwarding to online runner: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
