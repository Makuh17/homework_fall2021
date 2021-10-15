import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure import utils
from scipy.linalg import toeplitz

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data, and , DONE
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using, DONE
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        q_values = []
        if not self.reward_to_go:
            for rewards in rewards_list:
                q_values = np.append(q_values, self._discounted_return(rewards))

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for rewards in rewards_list:
                q_values = np.append(q_values, self._discounted_cumsum(rewards))

        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            values =utils.unnormalize(utils.normalize(values_unnormalized, values_unnormalized.mean(), values_unnormalized.std()), q_values.mean(), q_values.std())
            # values_std = values.std()
            # values_mean = values.mean()
            #
            # q_values_std = q_values.std()
            # q_values_mean = q_values.mean()

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                tmp = q_values - values[:-1]
                for i in reversed(range(batch_size)):
                    # calculatr delta_t
                    if terminals[i] == 1:
                        delta_t = rews[i] - values[i]
                        advantages[i] = delta_t
                    else:
                        delta_t = rews[i] + self.gamma*values[i+1] - values[i]
                        advantages[i] = delta_t + self.gamma * self.gae_lambda * advantages[i + 1]
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula


                # remove dummy advantage
                advantages = advantages[:-1]



            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                advantages = q_values-values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero, Done
            ## and a standard deviation of one
            advantages = utils.normalize(advantages, advantages.mean(), advantages.std())

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        # It appears that all elements of the list have to be the same
        discounted_rewards = np.array([self.gamma**t * r for t, r in enumerate(rewards)])
        list_of_discounted_returns = [np.sum(discounted_rewards)]*len(rewards)
        # TODO: create list_of_discounted_returns, DONE

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        # {}
        # TODO: create `list_of_discounted_returns`, DONE
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        #import time

        # convolution solution, quick tests seem to indicate this is the fastest, but closely followed by matmul
        length = len(rewards)
        gammas = [self.gamma**t for t in range(length)]
        #start = time.time()
        conv = np.convolve(gammas, np.flip(rewards))
        list_of_discounted_cumsums = np.flip(conv[:length])
        #end = time.time()
        #conv_time = end-start

        # # for loop solution
        # start = time.time()
        # list_of_discounted_cumsums = []
        # for t in range(length):
        #     rew = np.array(rewards[t:])
        #     if t == 0:
        #         gam = np.array(gammas[0:])
        #     else:
        #         gam = np.array(gammas[0:-t])
        #     list_of_discounted_cumsums.append(np.sum(rew * gam))
        # end = time.time()
        # for_time = end-start
        #
        # start = time.time()
        # # matrix mult solution
        # gamma_mat = np.triu(toeplitz(gammas))
        # list_of_discounted_cumsums = np.dot(gamma_mat, rewards)
        # end = time.time()
        # mat_time = end-start

        return list_of_discounted_cumsums
