# Object-memory planner

This package is the clean implementation area for the next VLM-only experiment.

The intended boundary is:

- the VLM reports scene changes and subgoal-completion evidence;
- deterministic code owns and renders the complete memory state;
- model-specific adapters live in `planners/`;
- archived GPT-5 Nano and Qwen3-VL attempts remain under `../legacy/`.

Do not import implementation code from `legacy/` here. Reuse should happen only
after an interface has proved useful in the new design.
