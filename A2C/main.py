# ! / usr / bin / env python
# coding: utf-8

# ## 6.5 A2C (Advanced Actor-Critic) with PyTorch
# Package import
import numpy as np
import matplotlib.pyplot as plt
# get_ipython (). run_line_magic ('matplotlib','inline')
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import copy

# Constant setting
ENV = 'CartPole-v0'  # Issue name to use
GAMMA = 0.99  # time discount rate
MAX_STEPS = 200  # 1 Number of trial steps
NUM_EPISODES = 1000  # Maximum number of attempts

NUM_PROCESSES = 32  # Simultaneous execution environment
NUM_ADVANCED_STEP = 5  # Set how many steps to proceed to calculate the reward sum

# Constant setting for calculation of A2C loss function
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5


# Memory class definition
class RolloutStorage(object):
    '''Advantage Memory class for learning'''

    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps + 1, num_processes, 4)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # Store discount reward sum
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # index to insert

    def insert(self, current_obs, action, reward, mask):
        '''Store transition in the next index'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # Update index

    def after_update(self):
        '''When the number of advanced steps is completed, store the latest one in index0'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Calculate the sum of discount rewards for each step in the Advantage step'''

        # Caution: Calculated in the opposite direction from the 5th step
        # Caution: Advantage 1 is the 5th step. The fourth step is Advantage 2.・ ・ ・
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


# A2C Deep Neural Network Construction
class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Since the action is decided, the output is the number of types of action
        self.actor = nn.Linear(n_mid, n_out)
        self.critic = nn.Linear(n_mid, 1)  # State value, so 1 output

    def forward(self, x):
        '''Define the forward calculation of the network'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # Calculation of state value
        actor_output = self.actor(h2)  # Behavior calculation

        return critic_output, actor_output

    def act(self, x):
        '''Probabilistically find action from state x'''
        value, actor_output = self(x)
        # Calculate softmax in the direction of action type with # dim = 1
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(
            num_samples=1)  # dim = 1 to calculate the probability in the direction of the action type
        return action

    def get_value(self, x):
        '''Calculate the state value from the state x'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''Calculate the state value, log probability and entropy of actual action actions from state x'''
        value, actor_output = self(x)

        # dim = 1 to calculate in the direction of action type
        log_probs = F.log_softmax(actor_output, dim=1)
        # Find log_probs of actual actions
        action_log_probs = log_probs.gather(1, actions)

        # dim = 1 to calculate in the direction of action type
        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


# Define the class that is the brain of the agent and share it with all agents
class Brain(object):
    def __init__(self, actor_critic):
        # actor_critic is a deep neural network of class Net
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def update(self, rollouts):
        '''Update using all 5 steps calculated in Advantage'''
        obs_shape = rollouts.observations.size(
        )[2:]  # torch.Size ([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[: -1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # Note: The size of each variable
        # rollouts.observations [: -1] .view (-1, 4) torch.Size ([80, 4])
        # rollouts.actions.view (-1, 1) torch.Size ([80, 1])
        # values torch.Size ([80, 1])
        # action_log_probs torch.Size ([80, 1])
        # entropy torch.Size ([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size ([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Calculation of advantage (action value-state value)
        advantages = rollouts.returns[: -1] - values  # torch.Size ([5, 16, 1])

        # Calculate loss of Critic
        value_loss = advantages.pow(2).mean()

        # Calculate the gain of Actor and later multiply it by minus to make it loss
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detach and treat advantages as constants

        # Sum of error functions
        total_loss = (value_loss * value_loss_coef -
                      action_gain - -entropy * entropy_coef)

        # Updated join parameters
        self.actor_critic.train()  # In training mode
        self.optimizer.zero_grad()  # Reset gradient
        total_loss.backward()  # Calculate backpropagation
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        # Make the gradient size up to 0.5 so that the coupling parameters do not change too much at once

        self.optimizer.step()  # Update join parameters

# There is no agent class this time

# The class of the environment to execute


class Environment:
    def run(self):
        '''Main run'''

        # Generate env for the number of simultaneous execution environments
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        # Generate a brain brain shared by all agents
        n_in = envs[0].observation_space.shape[0]  # state is 4
        n_out = envs[0].action_space.n  # Action is 2
        n_mid = 32
        # Generating deep neural networks
        actor_critic = Net(n_in, n_mid, n_out)
        global_brain = Brain(actor_critic)

        # Generation of storage variables
        obs_shape = n_in
        current_obs = torch.zeros(
            NUM_PROCESSES, obs_shape)  # torch.Size ([16, 4])
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rollouts object
        # Keep rewards for current trials
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])
        # Keep rewards for last attempt
        final_rewards = torch.zeros([NUM_PROCESSES, 1])
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy array
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy array
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy array
        # Record the number of steps in each environment
        each_step = np.zeros(NUM_PROCESSES)
        episode = 0  # Number of trials of environment 0

        # Start of initial state
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size ([16, 4])
        current_obs = obs  # Store the latest obs

        # Advanced Learning object Saves the current state as the first state of rollouts
        rollouts.observations[0].copy_(current_obs)

        # Execution loop
        for j in range(NUM_EPISODES * NUM_PROCESSES):  # whole for loop
            # advanced Calculated for each step to be learned
            for step in range(NUM_ADVANCED_STEP):

                # Seek action
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1) → (16,) → tensor to NumPy
                actions = action.squeeze(1).numpy()

                # 1 step execution
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])

                    # Episode end evaluation and state_next set
                    if done_np[
                            i]:  # Done becomes true when 200 steps have passed or when tilted more than a certain angle

                        # Output only when environment is 0
                        if i == 0:
                            print('% d Episode: Finished after% d steps' % (
                                episode, each_step[i] + 1))
                            episode += 1

                        # Reward setting
                        if each_step[i] < 195:
                            # Give reward -1 as a penalty if you get lost in the middle
                            reward_np[i] = -1.0
                        else:
                            # Give reward 1 at the end while standing
                            reward_np[i] = 1.0

                        each_step[i] = 0  # reset the number of steps
                        # Reset execution environment
                        obs_np[i] = envs[i].reset()

                    else:
                        reward_np[i] = 0.0  # Normally reward 0
                        each_step[i] += 1

                # Convert reward to tensor and add to total trial reward
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # For each execution environment, set mask to 0 if done, and set mask to 1 if ongoing
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # Update the total reward for the last attempt
                # If it is ongoing, multiply it by 1 and leave it as it is. If it is done, multiply it by 0 to reset it.
                final_rewards *= masks
                # Add 0 while continuing, add episode_rewards when done
                final_rewards += (1 - masks) * episode_rewards

                # Update total trial reward
                # The ongoing mask is 1, so leave it as it is, and if done, set it to 0.
                episode_rewards *= masks

                # Set the current state to 0 when done
                current_obs *= masks

                # Updated current_obs
                obs = torch.from_numpy(obs_np).float()  # torch.Size ([16, 4])
                current_obs = obs  # store the latest obs

                # Insert step transition into memory object now
                rollouts.insert(current_obs, action.data, reward, masks)

            # advanced for loop end

            # Calculate the expected state value from the state of the advanced final step

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # The size of rollouts.observations is torch.Size ([6, 16, 4])

            # Calculate the sum of discount rewards for all steps and update the rollouts variable returns
            rollouts.compute_returns(next_value)

            # Network and rollout updates
            global_brain.update(rollouts)
            rollouts.after_update()

            # Success if all NUM_PROCESSES continue to pass 200 steps
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('continuous success')
                break


cartpole_env = Environment()
cartpole_env.run()
