import torch
import torch.nn as nn
from torch.distributions import Categorical


def preprocess(obs_array, agent_id):
    """Flip piece signs for player_1 so both players see their own pieces as positive, then scale to [-1, 1]."""
    board = torch.FloatTensor(obs_array.copy())
    if agent_id == "player_1":
        board = -board
    board = board / 2.0
    return board


# policy network: maps board state to action logits
class Actor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=324):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs_tensor, action_mask):
        logits = self.forward(obs_tensor)
        # mask illegal actions with large negative values
        mask_t = torch.FloatTensor(action_mask)
        masked_logits = logits + (1 - mask_t) * (-1e9)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def get_log_prob(self, obs_tensor, action, action_mask):
        logits = self.forward(obs_tensor)
        mask_t = torch.FloatTensor(action_mask)
        masked_logits = logits + (1 - mask_t) * (-1e9)
        dist = Categorical(logits=masked_logits)
        action_t = torch.tensor(action, dtype=torch.long)
        return dist.log_prob(action_t), dist.entropy()


# value network: estimates V(s)
class Critic(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActorCriticAgent:
    def __init__(self, lr=1e-3, gamma=0.99, beta=0.01):
        self.actor = Actor()
        self.critic = Critic()
        self.gamma = gamma
        self.beta = beta
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def update(self, obs, action, action_mask, reward, next_obs, done, I):
        obs_t = preprocess(obs, "player_0")
        next_obs_t = preprocess(next_obs, "player_0")

        V_s = self.critic(obs_t)

        if done:
            V_sp = 0.0
        else:
            V_sp = self.critic(next_obs_t).detach()

        # td error
        delta = reward + self.gamma * V_sp - V_s

        critic_loss = delta ** 2

        log_prob, entropy = self.actor.get_log_prob(obs_t, action, action_mask)

        # policy gradient scaled by discount factor I
        actor_loss = -log_prob * delta.detach() * I

        # entropy bonus to encourage exploration
        entropy_loss = -self.beta * entropy

        total_loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
