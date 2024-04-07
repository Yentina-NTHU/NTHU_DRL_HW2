import cv2
import numpy as np
from collections import deque
import torch
from torch import nn

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class Model(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # DuelingDQN: Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        # DuelingDQN: Separate streams for V and A
        # Outputs a single value (V)
        self.value_stream = nn.Linear(512, 1)
        # Outputs advantage for each action (A)
        self.advantage_stream = nn.Linear(512, n_actions)

        if freeze:
            self._freeze()
    
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _freeze(self):
        for p in self.conv_layers.parameters():
            p.requires_grad = False
        for p in self.feature_layer.parameters():
            p.requires_grad = False
        for p in self.value_stream.parameters():
            p.requires_grad = False
        for p in self.advantage_stream.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.feature_layer(x)
        # DuelingDQN: Split the output of the feature layer into value and advantage streams
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # DuelingDQN: Combine value and advantages to get Q-values as per dueling architecture
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values



class Agent(object):
    def __init__(self):
        self.input_dims = (4, 84, 84)
        self.num_actions = 12

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.network = nn.DataParallel(Model(self.input_dims, self.num_actions, freeze=True)).to(self.device)
        self.network.load_state_dict(torch.load('112062570_hw2_data', map_location=torch.device('cpu')))

        # Preprocessing Params
        self.num_stack = 4
        self.frame_skip = 4
        self.frame_buffer = deque(maxlen=(self.num_stack*self.frame_skip)-1)  # store last num_stack-1 frames

    def act(self, observation):
        observation = self.preprocess_observation(observation)
        observations = self.get_stacked_observation(observation)
        if np.random.random() < 0.1:
            return np.random.choice(self.num_actions)
        else:
            observation_tensor = torch.tensor(np.array(observations), dtype=torch.float32).to(self.device).unsqueeze(0)
            actions = self.network(observation_tensor)
            return torch.argmax(actions).item()
        

    def preprocess_observation(self, observation, shape=(84, 84)):
        # Resize
        observation = cv2.resize(observation, shape, interpolation=cv2.INTER_AREA)
        # Grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Normalize
        observation = observation / 255.0
        
        return observation  # or stacked_observation if stacking frames

    def get_stacked_observation(self, new_obs):
        self.frame_buffer.append(new_obs)
        if len(self.frame_buffer) < (self.num_stack*self.frame_skip)-1:
            # If buffer isn't full yet, duplicate the first frame
            for _ in range((self.num_stack*self.frame_skip) - 1 - len(self.frame_buffer)):
                self.frame_buffer.appendleft(new_obs)
        stacked_observation = np.stack(list(self.frame_buffer)[3::4] + [new_obs], axis=2)
        stacked_observation = stacked_observation.transpose((2, 0, 1))
        return stacked_observation

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    observation = env.reset()

    agent = Agent()
    total_rewards = []

    NUM_EPISODES = 1        
    for _ in range(NUM_EPISODES):
        observation = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.act(observation) 
            observation, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
        print("Episode Reward:", total_reward)
        total_rewards.append(total_reward)

    env.close()
    print("Average Reward:", np.mean(total_rewards))

if __name__ == "__main__":
    main()

