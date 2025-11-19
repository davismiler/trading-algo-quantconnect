import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

from access_api import APIAccess

# Load environment variables
load_dotenv()


# Define a simple environment
class SimpleEnv:
    def __init__(self):
        # Load credentials from environment variables
        user_id = os.getenv('QC_USER_ID')
        api_token = os.getenv('QC_API_TOKEN')
        project_id = os.getenv('QC_PROJECT_ID')
        pair_name = os.getenv('QC_PAIR_NAME', 'XAUUSD')
        
        # Validate credentials
        if not all([user_id, api_token, project_id]):
            raise ValueError(
                "Missing required environment variables. Please set QC_USER_ID, "
                "QC_API_TOKEN, and QC_PROJECT_ID in your .env file"
            )
        
        try:
            user_id = int(user_id)
            project_id = int(project_id)
        except ValueError as e:
            raise ValueError(f"Invalid user_id or project_id format: {e}")
        
        # The initial state represents the parameters of the trading algorithm when initialised
        self.api_object = APIAccess(user_id, api_token, project_id, pair_name)
        
        try:
            self.initial_params = self.api_object.get_parameters()
            if self.initial_params is None:
                raise ValueError("Failed to retrieve initial parameters from API")
            self.state = np.array(self.initial_params)
            self.prev_backtest_id = self.api_object.backtest()
            if self.prev_backtest_id is None:
                raise ValueError("Failed to create initial backtest")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment: {e}")

    def reset(self):
        # Resets the state to the state of the environment when it was initialised
        self.state = np.array(self.initial_params)
        try:
            self.prev_backtest_id = self.api_object.backtest()
            if self.prev_backtest_id is None:
                raise ValueError("Failed to create backtest during reset")
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment: {e}")
        return self.state

    def step(self, action):
        try:
            prev_score = self.api_object.compute_score_from_results(self.prev_backtest_id)
            if prev_score is None:
                raise ValueError("Failed to compute previous score")
        except Exception as e:
            raise RuntimeError(f"Failed to get previous score: {e}")

        # Adds or Subtracts one from one of the values in the state
        index = action // 2
        change = 1 if action % 2 == 0 else -1
        self.state[index] += change

        # Updates the model through the API based on the action
        try:
            self.api_object.update_parameters(int(self.state[0]), int(self.state[1]), int(self.state[2]))
            new_backtest_id = self.api_object.backtest()
            if new_backtest_id is None:
                raise ValueError("Failed to create new backtest")
            new_score = self.api_object.compute_score_from_results(new_backtest_id)
            if new_score is None:
                raise ValueError("Failed to compute new score")
        except Exception as e:
            raise RuntimeError(f"Failed to execute step: {e}")

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
    try:
        env = SimpleEnv()
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return None, None
    
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
        try:
            state = env.reset()
            state = torch.FloatTensor(state)
            total_reward = 0

            # 100 steps per episode
            for t in range(100):  

                if np.random.rand() < epsilon:
                    action = np.random.randint(0, 2 * state_dim)
                else:
                    # Selects the action with the highest predicted value (sum of discounted future rewards)
                    with torch.no_grad():
                        action_values = model(state)
                    action = torch.argmax(action_values).item()

                # Performs the action
                try:
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

                    # Updates model parameters using backpropagation to minimise the loss function
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    state = next_state

                    if done:
                        break
                except Exception as e:
                    print(f"Error in step {t} of episode {episode + 1}: {e}")
                    break

            scores.append(total_reward)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)  # Moving average of last 100 episodes
            average_scores.append(avg_score)

            # Implements epsilon decay
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}")
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            continue

    return scores, average_scores


if __name__ == "__main__":
    # Train the agent
    scores, average_scores = train_agent()
    
    if scores is not None and average_scores is not None:
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
    else:
        print("Training failed. Please check your configuration and try again.")