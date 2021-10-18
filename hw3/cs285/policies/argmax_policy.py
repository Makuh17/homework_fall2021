import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # TODO return the action that maxinmizes the Q-value
        ## at the current observation as the output

        # This can be done by monte carlo sampling if the action space is continuous.
        # otherwise it is fairly trivial. Notice that seemingly, this whole setup only works for discrete action spaces.
        # for this reason, the continuous case is not currently handled.
        qa_val = self.critic.qa_values(observation)
        action = np.argmax(qa_val, axis=1)
        return action.squeeze()
