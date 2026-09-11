# Gemma 4 object-memory planner

This package implements the offline diagnostic architecture:

1. Gemma 4 26B A4B observes front-view frames at `t-12`, `t-8`, `t-4`, and `t`.
2. `MemoryReducer` assigns persistent identities and derives relations and
   events from the observation delta.
3. A separate Gemma 4 decision call judges completion and selects one atomic stage.

Initialization requires a visible-object inventory. Later perception calls may
return an empty delta; unchanged entities remain in deterministic memory. Gemma
returns only visible object-state observations, visibility changes, and
uncertainties. It does not return relations or events. The reducer converts
normalized `xyxy` boxes to 256-space `yxyx`, assigns persistent IDs, and
deterministically creates the allowed events (`appeared`, `moved`, `covered`,
`uncovered`, `pressed`, `picked`, and `placed`). Final memory separates the
persistent `entities` catalog (`id`, `type`) from dynamic `states`
(`bbox_yxyx`, color, visibility, motion, interaction flags). Entity IDs do not
encode color or position.

Perception uses separate system prompts:
`perception_rules_initial.txt` for the frame-0 inventory and
`perception_rules_delta.txt` for later change-only calls. The delta prompt
defines a 20-unit per-coordinate deadband in normalized `xyxy` space
(approximately 5 pixels at 256x256); an object whose four coordinates all
change by less than that is omitted unless a non-position state changed. A
`covers` relation is created only when a previously visible cube is reported
occluded, its previous bbox center lies inside exactly one currently observed
container bbox, and that container is an unambiguous one-to-one candidate.
Ambiguous or merely nearby containers do not create a relation. Reappearance
of the cube removes the relation and creates `uncovered`. Button and held-state
transitions similarly create `pressed`, `picked`, and `placed` events.

Oracle-video manifests retain both the recorded subgoal-text transition frames
and effective completion frames. The recorder changes its displayed subgoal one
frame after the action-completion image, so offline scoring uses
`completion_frame = text_transition_frame - 1`.

Decision stages are free-form language. Perception responses are strictly
validated against the observation-only Pydantic schema; correction retries are
not performed. Deterministic oracle canonicalization is used only after
inference for offline scoring.

The model is resolved offline from Hugging Face's cache at
`/home/ed1116/.cache/huggingface`. Gemma 4 requires Transformers 5.5 or newer, so
the launchers use the isolated `.venv-gemma4` environment rather than changing
the existing MS-Swift environment.

Run the five-episode serial evaluation with:

```bash
scripts/eval_object_memory_offline.sh
```

The FP16 checkpoint requires two 46–49 GB GPUs. For one episode on a GPU pair:

```bash
scripts/eval_object_memory_offline_gpu.sh 0,1 --args.only-episode=3 \
  --args.output-dir=runs/evaluation/object-memory-offline/ep3
```

Each call directory contains its four input images and four explicit stage
outputs: `perception_output.json`, `reducer_output.json`,
`decision_output.json`, and `grounded_subgoal_output.json`. The decision file is
the raw second-VLM result; the grounded-subgoal file is the deterministic
coordinate grounding derived from reducer memory. Raw model attempts and the
combined `call.json` trace are also retained. Archived planner attempts remain
under `../legacy/`; this package does not import them.
