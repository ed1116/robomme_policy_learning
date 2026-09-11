# Session handoff — 2026-07-28

## Goal and current direction

The project is testing ways to improve temporal perception and object memory for
RoboMME `ButtonUnmaskSwap`, especially tracking container swaps and detecting
brief button presses.

The current conclusion is:

- Gemma 4 can reason over a persistent memory, but it is not a reliable source
  of exhaustive, temporally current object coordinates.
- Geometry, persistent instance tracking, and movement detection should move to
  a segmentation/tracking model.
- Gemma should retain semantic responsibilities that are difficult to derive
  from geometry alone, such as `pressed`, `held`, and `highlighted`.
- The deterministic reducer should derive strict geometric relations and
  events such as `appeared`, `moved`, `covered`, and `uncovered`.

The immediate next experiment is a SAM 3.1 (fallback SAM 3) segmentation smoke
test on the clean ep3 oracle-success video. The code and environment are
prepared, but the gated Hugging Face checkpoint has not been downloaded because
the current Hugging Face account has not been approved for either official
model repository.

## Repository safety

Main repository:

`/home/ed1116/Projects/robomme_policy_learning`

The worktree is intentionally dirty and contains substantial user work. Do not
reset, checkout, delete, or overwrite unrelated changes. In particular, many
object-memory and GroundSG files are modified or untracked.

Previous handoff:

`/home/ed1116/Projects/robomme_policy_learning/session_handoff_2026-07-27.md`

That document is useful history, but its perception/reducer description is now
partly outdated. Use this 2026-07-28 document as the current source of truth.

## Coordinate convention

This convention must be preserved everywhere:

- Gemma/segmenter observation coordinates:
  `bbox_xyxy_norm1000 = [x_min, y_min, x_max, y_max]`
- Reducer/final policy coordinates:
  `bbox_yxyx` in the 256×256 system
- Grounding point:
  `[y, x]` in the 256×256 system

Do not describe 256-space boxes as `xyxy`, and do not describe normalized
1000-space boxes as `yxyx`.

## Current object-memory pipeline

At every 16-frame decision point:

1. Gemma perception receives four chronological front-camera images at
   `t-12`, `t-8`, `t-4`, and `t`.
2. Gemma emits observations only.
3. `MemoryReducer` assigns persistent IDs and derives deterministic
   relations/events.
4. A separate Gemma decision call receives the task and reducer memory, then
   predicts completion and the next atomic subgoal.
5. Deterministic grounding converts the target entity's memory box to a
   256-space `[y, x]` point.

Gemma calls are independent model generations. Continuity exists only because
the code explicitly supplies reducer memory and subgoal history to later calls;
there is no ChatGPT-like persistent KV/session state.

### Current division of responsibility

Gemma perception currently emits:

- `observed_objects`
- `observed_visibility_changes`
- `uncertainties`
- visible attributes including `pressed`, `highlighted`, and `held`

Gemma no longer emits relations or events.

The deterministic reducer owns:

- persistent entity IDs
- coordinate conversion from `xyxy norm1000` to `yxyx 256`
- `appeared` and `moved`
- `covers` relations
- `covered` and `uncovered`
- transition events generated from Gemma states:
  `pressed`, `picked`, and `placed`

### Prompt split and no-change format

Perception prompts are split into:

- `examples/robomme/subgoal_prediction/object_memory/perception_rules_initial.txt`
- `examples/robomme/subgoal_prediction/object_memory/perception_rules_delta.txt`

The initialization prompt asks for a complete newest-frame inventory.

The delta prompt asks only for changes. Its schema-valid no-change response is:

```json
{
  "observed_objects": [],
  "observed_visibility_changes": [],
  "uncertainties": []
}
```

It must not emit literal `null`.

`observation_id` is response-local and unique across the entire response:
`observation_1`, `observation_2`, and so on. Numbering does not restart for
another field. It is not a persistent object identity.

### Movement rule

Both the delta prompt and reducer use:

`MOVE_DEADBAND_NORM1000 = 20`

This is approximately 5 pixels at 256×256. If all four absolute differences
between prior and current `xyxy norm1000` coordinates are below 20, the object
has not moved. The current implementation declares movement if any one box edge
changes by 20 or more.

The latter is still brittle: a single unstable edge may generate a false
movement even when the center is nearly stationary.

### Strict coverage rule

The reducer creates `container covers cube` only when:

- a previously visible cube is explicitly reported occluded,
- the cube is absent in the newest frame,
- the cube's previous center lies inside a currently observed container box,
- exactly one unambiguous container candidate satisfies the rule.

Nearness alone is insufficient. Reappearance removes the relation and creates
`uncovered`.

## Important current files

- `examples/robomme/subgoal_prediction/object_memory/README.md`
- `examples/robomme/subgoal_prediction/object_memory/planner.py`
- `examples/robomme/subgoal_prediction/object_memory/reducer.py`
- `examples/robomme/subgoal_prediction/object_memory/schemas.py`
- `examples/robomme/subgoal_prediction/object_memory/perception_rules_initial.txt`
- `examples/robomme/subgoal_prediction/object_memory/perception_rules_delta.txt`
- `examples/robomme/subgoal_prediction/object_memory/decision_rules.txt`
- `examples/robomme/eval_object_memory_offline.py`
- `scripts/eval_object_memory_offline_gpu.sh`
- `examples/robomme/subgoal_prediction/gemma_groundsg.py`
- `examples/robomme/subgoal_prediction/gemma_groundsg_worker.py`
- `scripts/eval_gemma_groundsg.sh`

Detailed frame-80/frame-96 analysis:

`/home/ed1116/frames_80_and_96.md`

## Clean oracle ep3 video

The clean, no-yellow-dot ep3 oracle-success video used throughout the latest
analysis is:

`runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/oracle/videos/ButtonUnmaskSwap_ep3_success_first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube_hard.mp4`

Video metadata:

- 570 frames
- 30 fps
- stored video resolution: 512×528
- the object-memory loader uses the front-view crop/resizing expected by the
  existing evaluation pipeline

## Evaluation history and findings

### Models tried

| Model/method | Pipeline | Result and useful behavior | Main failure |
|---|---|---|---|
| GPT-5.6 in `Projects/robomme_planner` | Full-video/image reasoning with explicit planner memory | Best qualitative swap reasoning; tracked the successful reference episode well | Not verified as a reproducible closed-loop local evaluation |
| GPT-5-Nano | GroundSG-style staged prompting | Partial task decomposition | Failed to reliably complete even the first subgoal |
| Qwen3-VL-4B | GroundSG and split object-memory prototypes | Some correct individual calls | Offline split run had 0/5 episode success and only 48/200 matching calls; online path also suffered schema failures |
| Gemma 4-26B-A4B | Split perception → reducer → decision, plus separate GroundSG reproduction | Most promising local model; initial buttons and some swap facts can be correct | Brief press states, exhaustive delta output, current-frame coordinates, and temporal attention remain unreliable |

Do not interpret the earlier qualitative percentages as benchmark success
rates. The verified complete object-memory ep3 result below is 0/1.

### `full_ep3_3`

Path:

`runs/evaluation/object-memory-offline/gemma-4-26b-a4b-it/full_ep3_3`

Verified report:

- success: `false`
- matching calls: `11/36`
- first divergence: frame `96`
- failure: premature first-button completion

At frame 96 Gemma falsely reported `button_b pressed=true` using evidence frame
92. The decision call trusted this and moved to the second button before the
oracle transition. At frame 112, when the actual first press was visible,
Gemma/decision failed to register completion. The run then remained stuck on
the second-button stage.

This is primarily a model perception failure. A single clear frame showing
`pressed=true` should have been sufficient; the four-frame window itself was
not the fundamental problem.

### `full_ep3_5`

Path:

`runs/evaluation/object-memory-offline/gemma-4-26b-a4b-it/full_ep3_5/ep3`

This run was interrupted after call `t0256`; it has 17 call directories and no
final episode report.

The deterministic relation/event redesign fixed the earlier free-form
`subject_ref: observation_1` relation problem and avoided asking Gemma to name
container identities in relation objects. It did not solve unreliable geometry
or missed temporal changes.

At `t0096`, raw Gemma output contained only:

- `button_b`
- `pressed=true`
- evidence frame `92`

No postprocessor removed container observations; Gemma itself omitted them.
The model had been instructed to include all bbox changes, and both moving
containers exceeded the threshold by a large margin. Therefore this was not a
borderline deadband issue or reducer filtering.

At `t0112`, Gemma emitted the same button with `pressed=false`, evidence frame
104. It still did not provide reliable exhaustive current-frame geometry.

### Container ID assignment at frame 32

At `t0032`, new containers had `entity_id=null`. The reducer deterministically
sorted new same-type objects by x-center and assigned:

- `observation_1 → container_a`
- `observation_2 → container_b`
- `observation_4 → container_c`
- `observation_3 → container_d`

The mistaken earlier claim that observation number directly equals container
letter caused apparent `container_c/container_d` jumps. That mapping was never
guaranteed.

### What actually moved from frame 32 to 80

Independent pixel/connected-component measurements showed:

- physical `container_a` moved
- physical `container_b` moved
- physical `container_c` stayed stationary
- physical `container_d` stayed stationary

Gemma's t80 box for `container_a` was stale/incorrect. It also made one edge of
`container_d` shift enough to trip the reducer's any-edge movement rule,
although the center changed by only about 2.8 pixels.

Thus the primary failure was inaccurate/inconsistent perception geometry,
amplified by a brittle deterministic movement predicate.

### Frame 80 to 96 omission

Independent approximate boxes:

- `container_a`: `[99,99,125,121] → [88,106,113,127]` in 256-space
  `yxyx`
- `container_b`: `[97,134,123,154] → [110,128,137,151]` in 256-space
  `yxyx`

Both changes were far beyond the 5-pixel/20-normalized-unit threshold. With
accurate boxes and correct persistent IDs, the reducer could have tracked both
movements and the button press in the same call. Gemma simply failed to
exhaustively compare/report all changed objects.

Accurate coordinates alone are not sufficient if Gemma supplies a wrong
existing `entity_id`, because the reducer currently treats a valid same-type
explicit ID as authoritative.

### Temporal-attention failure

The prompt says the newest image defines current state, but Gemma does not
consistently use the newest image for its reported coordinates/state:

- call t80, frames 68/72/76/80: evidence frame 80
- call t96, frames 84/88/92/96: evidence frame 92
- call t112, frames 100/104/108/112: evidence frame 104
- call t144: reported objects used evidence frame 132
- call t224: reported objects used evidence frame 212
- call t240: evidence frame 240

The current schema conflates:

- the frame defining current geometry/state, and
- an earlier frame that provides evidence of a transition.

A better schema should separate:

```json
{
  "geometry": {
    "entity_id": "button_b",
    "current_frame": 96,
    "bbox_xyxy_norm1000": [400, 300, 500, 400]
  },
  "transitions": [
    {
      "type": "pressed",
      "evidence_frame": 92
    }
  ]
}
```

Do not let an earlier transition evidence frame silently become the object's
current box frame.

## GroundSG reproduction with Gemma

A separate Gemma implementation was added to approximate the Gemini GroundSG
baseline by rebuilding the full multimodal conversation on every independent
Gemma call. It does not have true persistent KV/session caching.

Path:

`runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/gemma-groundsg`

Verified result:

- ep3 success: `false`
- total success rate: `0.0`
- run stopped after four turns

Conversation:

- turn 1 predicted a reasonable initial subgoal sequence
- at step 96 Gemma incorrectly described both buttons as already pressed
- it selected `pick up the container at <715, 465> that hides the green cube`
- it repeated this subgoal at steps 144 and 192

This method is slower than a genuinely continuous hosted session because the
complete text/image/video history must be reprocessed on every local call.
The Gemini method should not be assumed to use KV caching merely because its
logical conversation is continuous.

## Recommended architecture

Use a segmentation/tracking model as the authoritative geometry source:

1. Segment all visible instances of `cube`, `container`, `robot arm`, and
   `button`.
2. Maintain persistent instance tracks and per-frame masks/boxes.
3. Derive movement, appearance, disappearance candidates, overlap, and cube
   center-inside-container tests deterministically.
4. Ask Gemma only for semantic visible states and ambiguous transitions:
   `pressed`, `held`, `highlighted`, and potentially ambiguous occlusion.
5. Keep current geometry and semantic transition evidence as separate schema
   fields.

Using simulator ground-truth masks directly during evaluation would make the
method oracle/cheating. It is acceptable to use simulator masks as training
labels for a learned segmentation model, as long as evaluation uses only
pixels.

## SAM 3.1 / SAM 3 preparation

### Official model/code status

Official repositories:

- `https://huggingface.co/facebook/sam3.1`
- `https://huggingface.co/facebook/sam3`
- `https://github.com/facebookresearch/sam3`

SAM 3.1 exists and is the preferred model. It uses the official SAM3 code but
has no standalone Hugging Face Transformers integration. SAM 3.1 adds Object
Multiplex for shared-memory multi-object tracking.

Official code clone:

`/home/ed1116/Projects/sam3`

Clone commit:

`6dbb02bd38288df755dfa1378000a861e65b84f6`

### Environment

Isolated environment:

`/home/ed1116/Projects/sam3/.venv`

It was created with `--system-site-packages` from the existing Gemma/robomme
Python environment so it can reuse the installed CUDA PyTorch without
redownloading it.

Verified imports:

- SAM3 package `0.1.0`
- PyTorch `2.9.1+cu128`
- OpenCV `4.11.0`
- timm `1.0.28`
- `iopath`, `ftfy`, `pycocotools`, and `decord`

CUDA was previously verified outside the sandbox with:

- Quadro RTX 8000
- PyTorch CUDA available: `true`

At the last GPU check, GPUs 0 and 1 were almost completely free; GPUs 2–5 were
busy. Recheck before running because this is transient.

### Download blocker

Both download attempts returned:

`Error: Access denied. This repository requires approval.`

The following directories contain only partial README/LICENSE files, not model
weights:

- `/home/ed1116/models/facebook-sam3.1`
- `/home/ed1116/models/facebook-sam3`

Missing required files:

- `/home/ed1116/models/facebook-sam3.1/sam3.1_multiplex.pt`
- `/home/ed1116/models/facebook-sam3/sam3.pt`

The user must sign into the same Hugging Face account used by the CLI and accept
the Meta access terms at one or both model pages. Do not bypass the gated model
with an unofficial mirror.

After approval, prefer:

```bash
/home/ed1116/.venvs/tools/bin/hf download facebook/sam3.1 \
  --local-dir /home/ed1116/models/facebook-sam3.1
```

If SAM 3.1 remains unavailable, use:

```bash
/home/ed1116/.venvs/tools/bin/hf download facebook/sam3 \
  --local-dir /home/ed1116/models/facebook-sam3
```

### Smoke-test code

New scripts:

- `scripts/sam3_segment_video.py`
- `scripts/run_sam3_ep3_smoke.sh`

The Python script:

- supports either SAM 3.1 or SAM 3 from a local checkpoint
- uses four separate text-prompt sessions:
  `cube`, `container`, `robot arm`, and `button`
- propagates each prompt forward across the entire ep3 video
- overlays class-colored masks, boxes, track IDs, scores, and frame numbers
- writes H.264 when the bundled ffmpeg encoder succeeds, otherwise mp4v
- writes both machine-readable and readable text outputs
- records both `bbox_xyxy_norm1000` and `bbox_yxyx_256`

Expected output directory:

`runs/evaluation/sam3-segmentation/ep3-smoke`

Expected files:

- `segmented.mp4`
- `detections.jsonl`
- `detections.txt`
- `summary.json`

The launcher automatically prefers the SAM 3.1 checkpoint and falls back to
SAM 3.

Run after checkpoint approval/download:

```bash
cd /home/ed1116/Projects/robomme_policy_learning
scripts/run_sam3_ep3_smoke.sh 0
```

This is a four-pass full-video test and may take substantial time on an RTX
8000. It has passed shell syntax, Python compilation, and `--help` checks, but
has not yet loaded a model or run inference because no checkpoint is present.

An attempted final BF16 compatibility micro-test was interrupted by the user.
The official predictors use BF16 autocast; therefore, if the first real model
run fails on the Turing RTX 8000, inspect BF16/autocast compatibility first and
adapt the predictor to FP16 only if necessary.

## Exact next-session checklist

1. Ask whether the user has accepted access at
   `facebook/sam3.1` (or `facebook/sam3`) in the Hugging Face browser.
2. Retry the preferred checkpoint download.
3. Confirm the checkpoint file exists and has gigabyte-scale size; SAM 3.1 is
   approximately 3.5 GB.
4. Recheck GPU availability with `nvidia-smi`.
5. Run `scripts/run_sam3_ep3_smoke.sh 0` on a free GPU.
6. Monitor for:
   - checkpoint key/load errors,
   - BF16 support errors,
   - CUDA OOM,
   - four sequential prompt sessions completing all 570 frames.
7. Validate all four expected output files.
8. Visually inspect representative frames around:
   - frame 32, when containers cover cubes,
   - frames 80–112, container swap and first button press,
   - later pickup/uncover transitions.
9. Compare SAM tracks against the independent frame-32/80/96 measurements in
   `/home/ed1116/frames_80_and_96.md`.
10. Only after the smoke test is credible, integrate segmenter geometry into
    `MemoryReducer`; do not immediately replace the existing pipeline based on
    an unvalidated video.
