import os
import sys

if __package__ is None or __package__ == "":
    # Allow running via `python app/main.py` by adding project root to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.agent_loop import run_agent
else:
    from .agent_loop import run_agent

if __name__ == "__main__":
    answer = run_agent("How to optimize curtailment today?")
    print(answer)
