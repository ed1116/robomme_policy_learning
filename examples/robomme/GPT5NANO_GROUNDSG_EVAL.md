# GPT-5 Nano + GroundSG pi0.5 Evaluation

This evaluation replaces the fine-tuned QwenVL high-level predictor with the
prompt-engineered `gpt-5-nano` object-memory planner. The low-level policy is
the same GroundSG pi0.5 checkpoint used by the existing symbolic evaluation.

## Fixed configuration

- High-level model: `gpt-5-nano` (not configurable)
- Low-level checkpoint: `runs/ckpts/symbolic-grounded-subgoal/79999`
- Planner interval: 16 environment timesteps
- Planner input: four front-camera RGB frames sampled every four timesteps
- Evaluation: 2 episodes for each of 16 tasks
- Output: `runs/evaluation/ours_ablation/ckpt79999/seed7/gpt-5-nano`
- GPU: policy server and RoboMME client both use `CUDA_VISIBLE_DEVICES=0`

Reset-provided demonstration frames are processed before any low-level action.
The execution boundary is obtained dynamically from
`exec_start_idx = len(front_rgb_list)-1`, rather than guessed from a fixed
frame number. Task-specific contexts identify the nine demonstration tasks:

`VideoUnmask`, `VideoUnmaskSwap`, `VideoPlaceButton`, `VideoPlaceOrder`,
`VideoRepick`, `MoveCube`, `InsertPeg`, `PatternLock`, and `RouteStick`.

## Terminal 1: pi0.5 server

```bash
cd /home/ed1116/Projects/robomme_policy_learning
bash scripts/serve_ours_ablation.sh
```

## Terminal 2: GPT-5 Nano evaluation client

```bash
cd /home/ed1116/Projects/robomme_policy_learning
export OPENAI_API_KEY="..."
bash scripts/eval_ours_ablation.sh
```

Run selected tasks with the Python entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 /home/ed1116/micromamba/envs/robomme/bin/python3.11 \
  examples/robomme/eval_gpt5nano_groundsg.py \
  --args.port=8011 \
  --args.only-tasks=ButtonUnmaskSwap,VideoUnmaskSwap
```

## Outputs

- `progress.json`: resume-compatible episode success flags
- `episode_results.json`: detailed result and diagnostic trace per episode
- `log.json`: final task and overall success rates
- `failure_episodes.csv`: GPT-5 Nano post-hoc taxonomy audit
- `task_reports.jsonl`: report emitted after each task
- `task_summaries/<task>.json`: per-task report
- `planner_logs/<task>/ep<id>/`: exact request JSON, four input images,
  parsed output JSON, and cumulative planner trace for every call
- `videos/`: rollout videos

Oracle subgoals are recorded only in the diagnostic trace for post-hoc failure
analysis. They are never included in a planner request or low-level policy
input.
