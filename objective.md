** Title: Object-Centric Memory for Concurrent Event Tracking and Temporally Consistent Robotic Planning **

* Main Objective

The goal of this project is to refine the GroundSG approach of RoboMME.pdf to show similar results to the GroundSG + Oracle method. The oracle GroundSG shows 84 % success rate in RoboMME benchmark, whereas the current QwenVL + GroundSG method achieves success rate of less than 40 %.

* The current GroundSG method

In the /home/ed1116/RoboMME.pdf paper, the GroundSG is one of the baseline methods in RoboMME, where the model is hierarchical. VLM acts as high-level planner, and VLA handles low-level execution. 
The VLM receives the current image frame(front camera image only, which is third-person view), demo video(depending on the task), the full task instruction(for example: "first press both buttons on the table then pick up the container hiding the green cube finally pick up another container hiding the blue cube"), and the previously predicted subgoals as input.
Then, the VLM generates the next grounded subgoal as output, which is then passed to the VLA. Grounded means that the subgoals mostly include coordinates.
The VLA uses the current image frame and the next subgoal as input, and generates action chunks as output.

Here is an example prompt:

{
"messages":[
{
"role":"system",
"content":"You are a helpful assistant to help guide the robot to complete the task by
predicting a sequence of grounded language subgoals"
},
{
"role":"user",
"content":"<video>The task goal is: watch the video carefully, then repeatedly pick up
and put down the same block that was previously picked up for two times, finally press the
button to stop\nThe history of previous predicted grounded language subgoals are: 1. pick up
the correct cube at <bbox> for the first time; 2. put it down; 3. pick up the correct cube at
<bbox> for the second time; 4. put it down; 5. press the button at <bbox> to finish\n<image>
What’s the next grounded language subgoal based on current observation?"
},
{
"role":"assistant",
"content":"press the button at <bbox> to finish"
}
],
"objects":{
"ref":[],
"bbox":[[390, 706], [394, 706], [261, 518], [261, 518]]
},
"images":[
"<path_to_current_image.png>",
],
"videos":[
"<path_to_task_input_video.mp4>"
]
}

Here is an example output:
"pick up the red cube at <160,18>."

* Problems with the current method

The current approach has three problems:

1. **The VLM cannot encode motion into memory** because it makes decisions from only a single image at each inference step.

2. **It cannot remember environmental changes that occur concurrently with the robot’s actions**, such as target cubes being covered by containers and the containers subsequently swapping positions. This is because the memory primarily records the subgoals executed by the robot.

3. **Evaluation analysis reveals premature subgoal transitions**: the model sometimes provides the next subgoal before the current one has been completed, causing actions to be skipped or the overall task execution to fail.

But for now, let's just focus on Problem 1 and Problem 2.

* Proposed Solution to solve both Problem 1 and Problem 2

1. Multiple images, sampled over time, used as VLM input. For example, the current GroundSG calls VLM every 16 frames. Then, instead of using the current frame at time t, use frames at time t-4, t-8, and t-12 as well to enable the VLM to detect temporal changes, such as movement, appear/disappear, etc.

2. Object-Centric Memory (Main Contribution): Taking inspiration from ECS(Entity-Component-System) method from game engines, use a JSON format to create and update object-based memory. Since the current VLM only takes its previously predicted subgoals as input, we not only do that but also tell the VLM to receive the past object-centric memory as input and output the updated object-centric memory. This allows the VLM to detect changes in the scene, independent to its current subgoal or action.

Here is an example prompt: I unfortunately don't have an example prompt, but think of it as an addition of the following:
- Global Rules: Projects/robomme_planner/planner_rules.txt
- Current frame t, past frames t-4, t-8, t-12
- Task Instruction
- Previous Predicted Subgoals
- Past Object-Centric Memory

Here is an example output:
Projects/robomme_planner/output_ep2_2/planner_output_t000224.json

* Research Direction

1. See if the method is plausible - provide a video rollout of success episode, and check every call(every 16 frames) output and see if object-centric memory has correctly captured all entities, states, relations, events, and doesn't have hallucinations. Also check if the grounded subgoal is temporally correct, see if the coordinates also make sense.

2. Look for previous works and see if anything already exists.

3. Evaluation, but the high-level VLM isn't finetuned. See if this can be done by only using prompt engineering. 10 episodes per each task, total 160 episodes.

4. Fine-tuning: QwenVL3-4B-Instruct model finetuned to generate object-centric memory

5. Reinforcement Learning: Improve Robustness, and also try to solve Problem 3 via turn-taking, etc.

* Main focus: Solve the "ButtonUnmaskSwap" task, since it includes concurrent movements and is overall a challenging task.

* Current Progress

Research Direction 1 - Partial Success
- The object-centric memory successfully captures the location of the container of the target cube after swap, while performing a different subtask "press the button."

Research Direction 2 - Work in Progress
- Currently didn't find a work that exactly coincides with mine, so novelty is there. Looked at CodeGraphVLP and Goal2Skill. Goal2Skill -> need to look in detail. I still need to check a lot more papers.

Research Direction 3 - Currently a failure.
- Check Projects/robomme_policy_learning/examples folder. Both GPT-5-Nano and Qwen3-VL-4B-Instruct models are not generating object-centric memory to my desire or is a weak model to execute such task.