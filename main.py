#!/usr/bin/env python3
"""OpenProctor CLI — delegates to openproctor.cli.

Usage:
  python main.py video.mp4
  python main.py data/input/*.mp4
  python main.py data/input/
"""

from openproctor.cli import app

if __name__ == "__main__":
    app()
