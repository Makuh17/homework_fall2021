This repository contains my solutions to the assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

Based on my interest in RL, I have chosen to self study this course as it appears to be the most comprehensive course on 
Deep RL available.

___
##Code overview
### HW1
###[scripts/run_hw1.py](hw1/cs285/scripts/run_hw1.py)
The interface script allowing to change parameters by passing arguments. Creates an instance of `RL_Trainer and runs the 
training loop based on the parameters given. No actual RL-functionality in this file.

###[infrastructure/rl_trainer.py](hw1/cs285/infrastructure/rl_trainer.py)
Defines the class `RL_Trainer` which upon initialisation sets up the chosen environment and agent, etc. The class has the
method `run_training_loop` which executes training. Several iterations of the following can be executed:
1. Collect trajectories (either using policy of from expert data)
2. Relabel (reassign actions based on expert policy) if running DAgger
3. Add data to replay buffer
4. Train agent based on buffer

If DAgger is not used, only a single (outer) iteration is performed and all data comes from expert dataset rather than 
relabeled based on current policy.

`train_agent()` samples randomly based on the agent's sampling function, which uses the `replay_buffer`'s sampling function.
Then uses the agent's `train()` function which for behaviour cloning is just updating the actor-network to mirror the 
expert actions. I.e, generate predicted actions, use the provided loss function to generate gradients and update based on that.

###[agents/bc_agent.py](hw1/cs285/agents/bc_agent.py)
Simply a class containing functions related to the agent. `train()`, `add_to_replay_buffer()`, `update()`

###[policies/MLP_policy.py](hw1/cs285/policies/MLP_policy.py)
Initialises policy networks and defines relevant methods. The actual network is defined in `pytorch_utils.py`

###[infrastructure/replay_buffer.py](hw1/cs285/infrastructure/replay_buffer.py)

###[infrastructure/utils.py](hw1/cs285/infrastructure/utils.py)
Mostly sampling and rollout-related functions.

###[infrastructure/pytorch_util.py](hw1/cs285/infrastructure/pytorch_util.py)
