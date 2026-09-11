# Session handoff — 2026-07-27

## Objective and current status

This work is a proof of concept for improving VLM temporal reasoning on
`ButtonUnmaskSwap`. The current diagnostic architecture separates perception,
memory maintenance, and decision making:

1. **Perception VLM call:** Gemma 4 receives four chronological 256×256
   front-view crops from frames `t-12`, `t-8`, `t-4`, and `t`.
2. **Deterministic reducer:** Applies the perception delta to persistent,
   object-centric memory.
3. **Decision VLM call:** Gemma 4 receives the task goal, active/completed
   subgoals, current events, and full reducer memory, then predicts
   `subgoal_completed` and the next subgoal.
4. **Deterministic grounding:** Looks up the selected entity in reducer memory
   and converts its bounding box into a 256-space `[y, x]` point.

There are two Gemma inference calls per 16-frame decision point. This is
intentionally an expensive diagnostic design, not yet the intended deployment
architecture. If it proves more reliable, the future direction is to
fine-tune/distill it into one compact VLM call.

## Important information boundaries

- The perception call receives only the cropped front-camera images, absolute
  frame numbers, the allowed object vocabulary, and a compact table of existing
  tracks. It does not receive the task goal or oracle subgoal.
- The decision call receives no images. Information from perception reaches it
  indirectly through reducer memory and current-call events.
- Across timesteps, information persists through reducer memory and the
  model-predicted active/completed subgoals. Raw VLM responses are not directly
  copied into the next VLM prompt.
- There is **no teacher forcing** for `subgoal_completed`, memory, or subgoals.
  Oracle subgoal timing is used only afterward for offline scoring.
- The physical video trajectory is an oracle-success rollout, but the Gemma
  calls are autoregressive.

## Prompt and policy changes

The following exact constraint is supplied to the perception prompt and its
runtime JSON payload:

> There are no objects other than these: ['cube', 'button', 'container']

The following exact policy is supplied to the decision prompt and runtime JSON
payload:

> If there are two or more buttons left to press, press the one with the biggest
> x-coordinate first.

Other prompt decisions:

- Perception should output a full visible inventory only at initialization and
  deltas afterward.
- `observation_id` is a response-local reference used for same-call events and
  relations. It is not persistent memory.
- Persistent entity IDs are opaque; color and position are state attributes,
  not identity or naming rules.
- A `covered`/`uncovered` event uses the cube as subject and container as
  object. A persistent `covers` relation uses container as subject and cube as
  object.
- `highlighted` was added as a state attribute.
- Possible subgoal forms are suggested in-context but are not a schema-enforced
  closed vocabulary.
- Confidence fields were removed.
- Pydantic VLM-output verification and schema-correction retries are disabled.
  There is one generation per VLM call; malformed structurally unusable output
  becomes a call error.

## Revised memory format

Reducer memory now follows the relevant conventions in
`/home/ed1116/Projects/robomme_planner/outputs_ep8/planner_output_t000096.json`:

- `entities`: persistent catalog containing only `{id, type}`.
- `states`: dynamic records joined by ID, containing `bbox_yxyx`, `present`,
  `visibility`, `motion`, `held`, `pressed`, `highlighted`, `color`, and
  `last_seen_frame`.
- Perception emits normalized `bbox_xyxy_norm1000`; the reducer converts it to
  256-space `bbox_yxyx`.
- Color does not gate automatic identity matching.
- Covered events are `cube -> container`; covers relations are
  `container -> cube`.

Remaining harmless differences from the ep8 example:

- `last_seen_frame` rather than `last_seen_step`.
- Current events/relations retain `_id`, evidence-frame, and source metadata.
- Current robot memory does not contain ep8's `gripper` field.
- Reducer memory includes `current_frame`.
- The reducer output is the memory object directly rather than being nested
  under `updated_memory`.

## Key files

- `examples/robomme/subgoal_prediction/object_memory/perception_rules.txt`:
  perception-only instructions and one JSON example.
- `examples/robomme/subgoal_prediction/object_memory/planner_rules.txt`:
  shared decision rules and the entities/states contract.
- `examples/robomme/subgoal_prediction/object_memory/decision_rules.txt`:
  completion and next-subgoal instructions.
- `examples/robomme/subgoal_prediction/object_memory/schemas.py`: VLM and
  reducer-memory schemas.
- `examples/robomme/subgoal_prediction/object_memory/reducer.py`:
  deterministic memory update and identity logic.
- `examples/robomme/subgoal_prediction/object_memory/planner.py`: two-call
  orchestration, JSON logging, and deterministic grounding.
- `examples/robomme/subgoal_prediction/object_memory/planners/gemma.py`:
  local Gemma 4 backend.
- `examples/robomme/eval_object_memory_offline.py`: oracle-video replay and
  scoring.
- `examples/robomme/subgoal_prediction/object_memory/oracle_video.py`:
  front-view extraction, four-frame windows, and oracle timing extraction.
- `scripts/eval_object_memory_offline_gpu.sh`: GPU evaluation launcher.

Each successful call directory contains:

- `perception_output.json`
- `reducer_output.json`
- `decision_output.json`
- `grounded_subgoal_output.json`
- raw attempt logs, four input PNGs, and combined `call.json`

## Verification completed

- Object-memory tests: **9 passed**.
- Ruff: passed.
- `git diff --check`: passed.

Five frame-0 smoke tests were run on oracle episodes 3–7:

| Episode | Visible inventory | Target | Grounded `[y,x]` | Initial result |
|---|---|---|---|---|
| ep3 | 2 buttons, 3 cubes | `button_b` | `[67,154]` | match |
| ep4 | 2 buttons, 3 cubes | `button_b` | `[62,153]` | match |
| ep5 | 2 buttons, 3 cubes | `button_b` | `[63,150]` | match |
| ep6 | 2 buttons, 3 cubes | `button_b` | `[63,154]` | match |
| ep7 | 2 buttons, 3 cubes | `button_b` | `[65,151]` | match |

Output:

`runs/evaluation/object-memory-offline/gemma-4-26b-a4b-it/smoke_ep3-7_ep8-memory`

These were only one-call initialization tests, not complete episode successes.
The fixed object vocabulary corrected ep5's earlier misclassification of
buttons as `base`.

## Partial full ep3 run

Command:

```bash
cd /home/ed1116/Projects/robomme_policy_learning
scripts/eval_object_memory_offline_gpu.sh 0,1 \
  --args.output-dir=runs/evaluation/object-memory-offline/gemma-4-26b-a4b-it/full_ep3 \
  --args.only-episode=3
```

The run was intentionally interrupted after six completed calls:
`t=0,16,32,48,64,80`. There were zero call errors, and no evaluation process is
currently running. The episode has 56 total calls. Because the episode did not
finish, `episode_report.json`, final scoring, and the requested annotated video
were not generated.

Partial output:

`runs/evaluation/object-memory-offline/gemma-4-26b-a4b-it/full_ep3/ep3`

Oracle stage transitions for ep3 are at frames:

`[1, 113, 289, 481, 513]`

### Failure trace

The error noticed at frame 80 (`container_e` and `container_f`) began earlier:

| Frame | What happened |
|---|---|
| `t=16` | Gemma ignored the delta instruction and regenerated all objects. Reducer identity was still correct. Button color became gray/yellow because the oracle grounding dot is visible. The `appeared` entries in reducer output are retained t=0 history, not new t=16 events; `current_call_events` was empty. |
| `t=32` | **First meaningful scene-memory divergence.** Cubes are visible at frame 28 and containers cover them by frame 32. Gemma reports old cube observations plus four new containers but emits no `covered` events or relations. The reducer therefore leaves cubes `visible` even though they are no longer visible at the end of the window. |
| `t=48` | **First persistent identity divergence.** Gemma permutes the four container IDs. The reducer treats supplied IDs as authoritative and records false movement for every container. |
| `t=64` | **First entity-count divergence.** Gemma reports each button twice: once as a button and again as `container_a`/`container_b`. The four physical containers are then assigned `container_c`–`container_f`, creating the erroneous `container_e` and `container_f`. |
| `t=80` | **First subgoal divergence.** Perception falsely emits a `pressed` event while the robot is above the first button. The decision call correctly follows that false event, changes to the second button, and marks completion. Oracle completion remains false until the call containing frame 113. |

### Root causes

1. Gemma treats the four-frame window as a combined inventory instead of
   distinguishing historical observations from end-of-window state.
2. The moved-only one-shot perception example does not adequately teach
   appearance/cover transitions.
3. The reducer blindly accepts an existing VLM-provided `entity_id` whenever
   its type matches, even after an implausible spatial jump.
4. There is no one-to-one global association check.
5. The same image region can be emitted simultaneously as two incompatible
   object types.
6. New IDs become permanent immediately; there is no provisional-track or
   duplicate-merging stage.
7. The decision call trusts a VLM `pressed` event without a persistent visual
   state transition.
8. The yellow oracle grounding marker contaminates perception.

### Localization-noise measurement

There is no filter that drops redundant observations. The existing
`MOVE_THRESHOLD_PIXELS = 10` only controls whether the reducer records a
`moved` event.

Measured on static ep3 objects from t=0–80:

- Center jitter: `0–2.5 px`
- Maximum individual box-edge jitter: `3 px`

Suggested future deadband: treat a detection as geometrically unchanged when
center displacement is under `4 px` and semantic state is unchanged. Keep the
existing `10 px` threshold for declaring `moved`. This does not fix the
identity, duplicate-class, or full-inventory failures above.

## Yellow-dot finding

The yellow dot in oracle videos is generated manually in:

`examples/robomme/utils.py`, inside `RolloutRecorder.record`

The recorder parses the grounded coordinate and calls:

```python
cv2.circle(concat_image, point[::-1], 5, (255, 255, 0), -1)
```

The marker is baked into the MP4. The saved evaluation directory contains only
the annotated videos and progress file, not the original clean rollout frames.
The reliable way to obtain a clean oracle rollout is to add a recorder option
that disables this point drawing and rerun the oracle episode with the same
seed. Inpainting the existing MP4 would not reproduce authentic pixels. No code
change for this has been made yet.

The VLM receives the cropped front-view portion of the annotated video, so it
currently sees the yellow dot. At t=80 the robot is above the still-visible dot,
which likely contributes to premature press detection.

## Event-history behavior

- Reducer memory retains only the latest 96 events.
- The decision prompt receives only the latest 32 events.
- Old events are truncated, not summarized into durable historical facts.
- `reducer_output.json` contains cumulative retained event history.
- `current_call_events` in `call.json` contains only events added during that
  timestep.

## Cost relative to the evaluated QwenVL GroundSG baseline

The exact baseline in
`runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/qwenvl` uses:

- one 256×256 image,
- one short task/history prompt,
- one short grounded-subgoal output capped at 128 tokens.

The current proof of concept uses:

- four images,
- two long prompts, with full reducer memory in the decision call,
- output caps of 1,600 perception tokens plus 400 decision tokens.

Approximate load is 4× visual input, roughly 15–30× textual input, and 3–30×
actual output depending on delta size. This is acceptable for diagnosis but is
not yet a strong final method. Novelty depends on first showing a meaningful
reliability gain and then reducing it to a compact/single-call implementation.

## Recommended next steps

Do not launch ten complete episodes yet. First:

1. Add an option to create a clean oracle video without the yellow grounding
   dot, then rerun ep3.
2. Strengthen temporal perception with a cover-transition example:
   cubes visible in earlier frames, containers visible at the newest frame,
   explicit `covered` events, and cube occlusion.
3. Make reducer data association authoritative. Treat VLM `entity_id` as a hint
   and perform one-to-one matching using prior boxes and trajectory.
4. Reject same-location incompatible duplicate detections.
5. Treat observations from older context frames as historical; do not mark an
   object visible at the current frame merely because it appeared earlier in
   the four-frame window.
6. Require an actual visual state transition for `pressed`; approach or robot
   overlap must remain insufficient.
7. Consider provisional new tracks and deterministic duplicate merging.
8. Rerun one full clean ep3 episode and inspect the first transition before
   expanding to ten episodes.
9. After a complete run, generate an H.264 overlay video containing oracle
   subgoal, Gemma subgoal/grounding, `subgoal_completed`, and divergence status.

No fixes from this failure analysis have been implemented yet.
