"""Compatibility entry point for the archived GPT-5 Nano evaluation."""

# ruff: noqa: F401,F403

from legacy.eval_gpt5nano_groundsg import *
from legacy.eval_gpt5nano_groundsg import _save_json
from legacy.eval_gpt5nano_groundsg import _selected_tasks
from legacy.eval_gpt5nano_groundsg import _task_report
from legacy.eval_gpt5nano_groundsg import _update_failure_csv


if __name__ == "__main__":
    import tyro

    tyro.cli(evaluate)
