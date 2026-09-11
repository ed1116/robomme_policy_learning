 ### Phase 1: Establish a causal offline benchmark

  Start with ButtonUnmaskSwap, but evaluate stored observation windows without
  executing the VLA.

  Create labels at every 16-step boundary for:

  - Which container initially covered each colored cube.
  - Container track identity after every swap.
  - Current boxes and visibility.
  - Relevant events.
  - Expected active-subgoal completion.
  - Expected next primitive and target object.

  This will tell you whether failures come from perception, tracking, completion,
  or planning.

  ### Phase 2: Reduce the model output

  Have the model output only an observation delta:

  {
    "observed_objects": [],
    "state_changes": [],
    "events": [],
    "uncertainties": []
  }

  Then let code:

  - Preserve entity IDs.
  - Merge states.
  - Retain occluded objects.
  - Deduplicate events.
  - Enforce timestamps and confidence.
  - Maintain task progress.
  - Compute grounding from the current observation.

  Add confidence, evidence_frame, and observation_source to inferred facts.
  Separate directly observed facts from persistent beliefs.

  ### Phase 3: Run component-oracle ablations

  The most informative experiment matrix is:

  1. Original GroundSG.
  2. Four frames, no object memory.
  3. Object memory, current frame only.
  4. Four frames plus object memory.
  5. Oracle detections/tracks plus learned planner.
  6. Learned detections/tracks plus oracle subgoal selection.
  7. Oracle completion verifier versus learned verifier.

  This quantifies which part closes the oracle gap. Report relative gap closure:

  [
  \frac{\text{Ours} - \text{GroundSG+Qwen}}{\text{GroundSG+Oracle} -
  \text{GroundSG+Qwen}}
  ]

  ### Phase 4: Fine-tune Qwen before RL

  Prompt engineering is useful for proving plausibility, but the RoboMME paper
  already shows a major domain gap for prompt-only foundation models. A base 4B
  model is unlikely to reliably regenerate a large structured belief state from
  scratch.

  Fine-tune Qwen on simulator-derived observation deltas, with a curriculum:

  1. Detection and attributes.
  2. Temporal association.
  3. Occlusion and swap updates.
  4. Events and relations.
  5. Completion judgment.
  6. Next-subgoal selection.

  Use difficult negative cases: identical containers, crossings, missed
  observations, false completion cues, and ambiguous boxes. Only consider RL after
  supervised outputs are structurally valid and offline tracking/planning metrics
  are strong.

  ### Phase 5: Expand beyond one task

  ButtonUnmaskSwap is the right anchor, but use representative tasks to establish
  generality:

  - StopCube: motion and temporal event counting.
  - PickHighlight: transient state.
  - VideoUnmaskSwap: identity tracking from demonstration.
  - BinFill: counting and object permanence.
  - ButtonUnmaskSwap: concurrent exogenous scene change.

  ## Bottom line

  Continue the project, but pivot from “ask a VLM to rewrite an object-centric
  world model” to “maintain a deterministic event-sourced belief state from
  learned visual deltas.” That will make the system more reliable, make GPT/Qwen
  comparisons meaningful, and give you the component-level evidence required for a
  strong paper.

  The most valuable next deliverable is not another 160-episode run. It is a
  labeled offline ButtonUnmaskSwap call-boundary benchmark plus the seven
  ablations above. That will tell you whether the research idea works before low-
  level control and serialization noise obscure the answer.
