"""
server/app.py — OpenEnv entry point for SRE-Gym.

openenv validate requires:
  - A main() function
  - if __name__ == '__main__' block
  - app entry point: server.app:main
"""
import uvicorn
from src.server import app


def main() -> None:
    """Start the SRE-Gym server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
