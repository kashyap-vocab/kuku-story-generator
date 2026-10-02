"""Run the web app:  python -m serial_writer.web  [--host 0.0.0.0] [--port 8000]

One process on purpose: stories run in background threads inside it, and the
in-process engine is what knows which story is running."""

from __future__ import annotations

import argparse

import uvicorn

from ..config import Settings
from ..engine import Engine
from .app import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m serial_writer.web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    engine = Engine(Settings.from_env())
    try:
        uvicorn.run(create_app(engine), host=args.host, port=args.port, workers=1)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
