import numpy as np
import torch

from myagent import ActorCriticAgent, preprocess
from mycheckersenv import NUM_SQUARES, decode_action, idx_to_rc
from mycheckersenv import env as make_env


def print_board(board):
    symbols = {0: ".", 1: "b", 2: "B", -1: "r", -2: "R"}
    # build full 6x6 grid with light squares shown as "_"
    grid = [["_" for _ in range(6)] for _ in range(6)]
    for i in range(NUM_SQUARES):
        row, col = idx_to_rc(i)
        grid[row][col] = symbols[board[i]]
    print("  " + " ".join(str(c) for c in range(6)))
    for r, row_data in enumerate(grid):
        print(f"{r} " + " ".join(row_data))


def format_square(idx):
    row, col = idx_to_rc(idx)
    return f"{idx}(r{row},c{col})"


def main():
    agent = ActorCriticAgent()
    weights_path = "actor_final.pt"
    try:
        agent.actor.load_state_dict(torch.load(weights_path, weights_only=True))
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found, using untrained weights.")
    agent.actor.eval()

    e = make_env()
    e.reset()

    print("=== Initial Board ===")
    print_board(e.unwrapped.board)
    print()

    total_steps = 0
    cumulative_reward = 0.0

    for current_agent in e.agent_iter():
        obs_dict, reward, termination, truncation, info = e.last()
        done = termination or truncation

        if current_agent == "player_0":
            cumulative_reward += reward

        if done:
            if reward != 0:
                print(f"--- {current_agent} receives terminal reward: {reward} ---")
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

        from_idx, to_idx = decode_action(action)
        e.step(action)
        total_steps += 1

        # print move and board after step
        print(
            f"--- Step {total_steps}: {current_agent} moved {format_square(from_idx)} -> {format_square(to_idx)} | reward: {reward} ---"
        )
        print_board(e.unwrapped.board)
        print()

    # determine winner
    if cumulative_reward > 0:
        winner = "player_0 (trained agent) WINS"
    elif cumulative_reward < 0:
        winner = "player_1 (random opponent) WINS"
    else:
        winner = "DRAW (truncated)"

    print("=" * 40)
    print(f"Total steps:       {total_steps}")
    print(f"Cumulative reward: {cumulative_reward}")
    print(f"Result:            {winner}")
    print("=" * 40)


if __name__ == "__main__":
    main()
