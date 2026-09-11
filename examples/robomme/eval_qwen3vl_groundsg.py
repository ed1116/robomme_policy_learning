"""Compatibility entry point for the archived Qwen3-VL evaluation."""

# ruff: noqa: F401,F403

from legacy.eval_qwen3vl_groundsg import *


if __name__ == "__main__":
    import tyro

    tyro.cli(evaluate)
