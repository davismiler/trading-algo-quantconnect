import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from access_api import APIAccess


# Define a simple environment
class SimpleEnv:
    def __init__(self):
        # The intital state represents the paramters of the trading algorithm when intialised
        self.api_object = APIAccess(298141,"f5f148c306562e6b458321e389255f6445b0ad7f7377ed96f71d4d87d7aeb44d",18808389,"XAUUSD")
        self.inital_params = self.api_object.get_parameters()
        self.state = np.array(self.inital_params)
        self.prev_backtest_id = self.api_object.backtest()

    def reset(self):
        # Resets the state to the state of the environment when it was was intitialised
        self.state = self.inital_params
        self.prev_backtest_id = self.api_object.backtest()
        return self.state

    def step(self, action):

        prev_score = self.api_object.compute_score_from_results(self.prev_backtest_id)

        # Adds or Subtracts one from one of the values in the state
        index = action // 2
        change = 1 if action % 2 == 0 else -1
        self.state[index] += change

        # Updates the model through the API based on the action
        self.api_object.update_parameters(self.state[0], self.state[1], self.state[2])
        new_backtest_id = self.api_object.backtest()
        new_score = self.api_object.compute_score_from_results(new_backtest_id)

        self.prev_backtest_id = new_backtest_id
        
        # Reward is positive when the score increased from the previous step
        reward = new_score - prev_score
        
        next_state = self.state

        # Continue indefinitely
        done = False 
        return next_state, reward, done, {}


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()

        # Creates a fully connected layer with input_dim inputs and 64 outputs
        self.fc1 = nn.Linear(input_dim, 64)

        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    
    # Defines how data flows through the network: x -> 64 -> 64 -> output_dim
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_agent():
    env = SimpleEnv()
    state_dim = 3
    input_dim = state_dim
    output_dim = 2 * state_dim
    model = DQNetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_episodes = 50000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start

    scores = []
    average_scores = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        # 100 steps per episode
        for t in range(100):  

            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2 * state_dim)
            else:
                # Selects the action with the highest predicted value (sum of dicounted future rewards)
                with torch.no_grad():
                    action_values = model(state)
                action = torch.argmax(action_values).item()

            # Performs the action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward

            # Adds the immediate reward to the discounted predicted maximum value
            target = reward + gamma * torch.max(model(next_state)).item()

            # Prediction for all actions in the current state
            output = model(state) 
            target_f = output.clone()

            # Updates chosen action with our target
            target_f[action] = target
            loss = criterion(output, target_f)

            # Updates model parmaters using backpropagation to minimise the loss function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if done:
                break

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])  # Moving average of last 100 episodes
        average_scores.append(avg_score)

        # Implements epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}")

    return scores, average_scores

# Train the agent
scores, average_scores = train_agent()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(scores, label='Score', alpha=0.3)
plt.plot(average_scores, label='Average Score', color='red')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.legend()
plt.show()

print(f"Final average score: {average_scores[-1]:.2f}")