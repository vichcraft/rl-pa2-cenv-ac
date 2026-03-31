import copy
import random

import torch
import numpy as np

from mycheckersenv import env as make_env
from myagent import ActorCriticAgent, preprocess


# stores historical policy snapshots for self-play
class OpponentPool:
    def __init__(self, max_size=10):
        self.pool = []
        self.max_size = max_size

    def add(self, agent):
        state_dict = copy.deepcopy(agent.actor.state_dict())
        self.pool.append(state_dict)
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample_opponent(self, agent):
        opponent_actor = copy.deepcopy(agent.actor)
        if self.pool:
            sd = random.choice(self.pool)
            opponent_actor.load_state_dict(sd)
        opponent_actor.eval()
        return opponent_actor


# play one game and train player_0 online after each of its turns
def run_episode(env, agent, opponent_actor, gamma=0.99):
    env.reset()
    I = 1.0
    p0_prev = None
    total_reward = 0.0

    for current_agent in env.agent_iter():
        obs_dict, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if done:
            if current_agent == "player_0" and p0_prev is not None:
                agent.update(*p0_prev, reward, obs_dict["observation"], done=True, I=I)
                total_reward += reward
            env.step(None)
            continue

        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]

        if current_agent == "player_0":
            # update on the previous transition before acting
            if p0_prev is not None:
                agent.update(*p0_prev, reward, obs, done=False, I=I)
                total_reward += reward
                I *= gamma

            action, _, _ = agent.actor.get_action(preprocess(obs, "player_0"), mask)
            p0_prev = (obs, action, mask)
            env.step(action)
        else:
            # opponent plays greedily from a past policy snapshot
            with torch.no_grad():
                obs_t = preprocess(obs, "player_1")
                logits = opponent_actor(obs_t)
                mask_t = torch.FloatTensor(mask)
                logits = logits + (1 - mask_t) * (-1e9)
                action = torch.argmax(logits).item()
            env.step(action)

    return total_reward


def evaluate(agent, num_games=100):
    """Win rate against a random opponent."""
    wins = 0
    for _ in range(num_games):
        e = make_env()
        e.reset()
        for current_agent in e.agent_iter():
            obs_dict, reward, termination, truncation, info = e.last()
            done = termination or truncation
            if done:
                if current_agent == "player_0" and reward > 0:
                    wins += 1
                e.step(None)
                continue

            obs = obs_dict["observation"]
            mask = obs_dict["action_mask"]

            if current_agent == "player_0":
                with torch.no_grad():
                    obs_t = preprocess(obs, "player_0")
                    logits = agent.actor(obs_t)
                    mask_t = torch.FloatTensor(mask)
                    logits = logits + (1 - mask_t) * (-1e9)
                    action = torch.argmax(logits).item()
            else:
                legal = np.where(mask == 1)[0]
                action = np.random.choice(legal)
            e.step(action)
    return wins / num_games


def train(num_episodes=50000, gamma=0.99, checkpoint_every=500, eval_every=1000):
    agent = ActorCriticAgent(lr=1e-3, gamma=gamma, beta=0.01)
    pool = OpponentPool(max_size=10)
    pool.add(agent)

    reward_history = []
    e = make_env()

    for ep in range(1, num_episodes + 1):
        opponent_actor = pool.sample_opponent(agent)
        ep_reward = run_episode(e, agent, opponent_actor, gamma=gamma)
        reward_history.append(ep_reward)

        # periodically save current policy to opponent pool
        if ep % checkpoint_every == 0:
            pool.add(agent)
            avg_reward = np.mean(reward_history[-checkpoint_every:])
            print(f"Episode {ep} | Avg reward (last {checkpoint_every}): {avg_reward:.4f}")

        if ep % eval_every == 0:
            win_rate = evaluate(agent, num_games=100)
            print(f"Episode {ep} | Win rate vs random: {win_rate:.2%}")

    torch.save(agent.actor.state_dict(), "actor_final.pt")
    print("Training complete.")
    return agent


if __name__ == "__main__":
    train()
