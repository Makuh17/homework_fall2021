#Assignment 1
##Code overview
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

---

## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2021/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs285/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs285/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](cs285/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v2.pkl
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

