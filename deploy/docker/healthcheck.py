#!/usr/bin/env python3
"""Lightweight health check probe for agent containers.

Reads the health status file written by the agent runner.
Exit 0 = healthy, exit 1 = unhealthy.
"""
import json
import sys
import time
from pathlib import Path

HEALTH_FILE = Path("/tmp/agent_health.json")
STALENESS_THRESHOLD_S = 90.0


def main() -> int:
    if not HEALTH_FILE.exists():
        # Agent hasn't started yet — let startupProbe handle this
        return 1

    try:
        data = json.loads(HEALTH_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return 1

    if not data.get("alive", False):
        return 1

    last_heartbeat = data.get("last_heartbeat_epoch", 0)
    if last_heartbeat > 0:
        age = time.time() - last_heartbeat
        if age > STALENESS_THRESHOLD_S:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
