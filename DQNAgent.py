import numpy as np
import torch
from torch.optim import Adam

from DuelingQNetwork import DuelingQNetwork
from PrioritizedReplayMemory import PrioritizedReplayMemory


class DQNAgent:
    def __init__(self,game, state_size=400, action_size=3, memory_size=50000, alpha=0.6, beta_start=0.4,
                 beta_increment=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = (game.grid_size * game.grid_size,)
        self.action_size = action_size
        self.memory = PrioritizedReplayMemory(capacity=memory_size, alpha=alpha, beta_start=beta_start,
                                              beta_increment=beta_increment)
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.5
        self.learning_rate = 0.001
        self.model = DuelingQNetwork((state_size,), action_size).to(self.device)
        self.target_model = DuelingQNetwork((state_size,), action_size).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch, indices, weights = self.memory.sample(batch_size)

        states = torch.stack([item[0][0] for item in minibatch], dim=0)
        actions = torch.tensor([item[1] for item in minibatch], dtype=torch.long)
        rewards = torch.tensor([item[2] for item in minibatch], dtype=torch.float32)
        next_states = torch.stack([item[3][0] for item in minibatch], dim=0)
        dones = torch.tensor([item[4] for item in minibatch], dtype=torch.float32)

        online_actions = self.model(next_states).detach().argmax(dim=1)
        target_q_values = self.target_model(next_states).detach()

        target_values = rewards + (1 - dones) * self.gamma * target_q_values.gather(1, online_actions.unsqueeze(
            1)).squeeze()

        current_q_values = self.model(states)

        updated_q_values = current_q_values.clone()
        updated_q_values[range(batch_size), actions] = target_values

        loss = ((updated_q_values - current_q_values).pow(2) * torch.tensor(weights, dtype=torch.float32).unsqueeze(
            1).expand_as(updated_q_values - current_q_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_priorities(indices, (updated_q_values - current_q_values).detach().abs().sum(dim=1).numpy())

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + 1e-4
        self.memory.update_priorities(indices, priorities)


    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        torch.save(self.target_model.state_dict(), name)

