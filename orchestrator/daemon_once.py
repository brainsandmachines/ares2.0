"""Single reconcile + top-up pass (used by the cron watchdog)."""

from __future__ import annotations

import logging
import sys

from .daemon import run_once


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    run_once()


if __name__ == "__main__":
    main()
