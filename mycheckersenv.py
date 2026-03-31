import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers


def rc_to_idx(row, col):
    """Convert (row, col) to flat dark-square index (0-17), or -1 if light square."""
    if (row + col) % 2 == 0:
        return -1
    offset = (col - 1) // 2 if row % 2 == 0 else col // 2
    return row * 3 + offset


def idx_to_rc(idx):
    """Convert flat dark-square index to (row, col)."""
    row = idx // 3
    pos = idx % 3
    col = (pos * 2 + 1) if row % 2 == 0 else (pos * 2)
    return row, col


def build_neighbors():
    """Precompute diagonal neighbors for each dark square.
    Returns dict: {square_idx: {direction: (adjacent_idx, jump_idx)}}
    jump_idx is -1 if the jump landing is off-board.
    """
    neighbors = {}
    for i in range(18):
        row, col = idx_to_rc(i)
        nbrs = {}
        for d, (dr, dc) in [
            ("ul", (-1, -1)),
            ("ur", (-1, 1)),
            ("dl", (1, -1)),
            ("dr", (1, 1)),
        ]:
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2 * dr, col + 2 * dc
            if 0 <= r1 < 6 and 0 <= c1 < 6:
                adj = rc_to_idx(r1, c1)
                jump = rc_to_idx(r2, c2) if 0 <= r2 < 6 and 0 <= c2 < 6 else -1
                if adj != -1:
                    nbrs[d] = (adj, jump)
        neighbors[i] = nbrs
    return neighbors


NEIGHBORS = build_neighbors()
NUM_SQUARES = 18
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 324


def encode_action(from_idx, to_idx):
    return from_idx * NUM_SQUARES + to_idx


def decode_action(action):
    return action // NUM_SQUARES, action % NUM_SQUARES


def get_capture_moves_from(board, agent, src):
    """Return capture (from, to) tuples originating from a single square."""
    piece = board[src]
    my_man = 1 if agent == "player_0" else -1
    my_king = 2 if agent == "player_0" else -2
    if piece not in (my_man, my_king):
        return []

    opp_values = {-1, -2} if agent == "player_0" else {1, 2}
    forward_dirs = ["ul", "ur"] if agent == "player_0" else ["dl", "dr"]
    all_dirs = ["ul", "ur", "dl", "dr"]
    dirs = all_dirs if abs(piece) == 2 else forward_dirs

    captures = []
    for d in dirs:
        if d not in NEIGHBORS[src]:
            continue
        adj, jump_sq = NEIGHBORS[src][d]
        if board[adj] in opp_values and jump_sq != -1 and board[jump_sq] == 0:
            captures.append((src, jump_sq))
    return captures


def get_legal_moves(board, agent, jumper=None):
    """Return list of (from, to) tuples. Captures are mandatory."""
    if jumper is not None:
        return get_capture_moves_from(board, agent, jumper)

    my_man = 1 if agent == "player_0" else -1
    my_king = 2 if agent == "player_0" else -2
    opp_values = {-1, -2} if agent == "player_0" else {1, 2}
    forward_dirs = ["ul", "ur"] if agent == "player_0" else ["dl", "dr"]
    all_dirs = ["ul", "ur", "dl", "dr"]

    captures = []
    moves = []

    for src in range(NUM_SQUARES):
        piece = board[src]
        if piece not in (my_man, my_king):
            continue
        dirs = all_dirs if abs(piece) == 2 else forward_dirs

        for d in dirs:
            if d not in NEIGHBORS[src]:
                continue
            adj, jump_sq = NEIGHBORS[src][d]
            if board[adj] == 0:
                moves.append((src, adj))
            elif board[adj] in opp_values and jump_sq != -1 and board[jump_sq] == 0:
                captures.append((src, jump_sq))

    return captures if captures else moves


def compute_action_mask(board, agent, jumper=None):
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for f, t in get_legal_moves(board, agent, jumper):
        mask[encode_action(f, t)] = 1
    return mask


def maybe_promote(board, to_idx, agent):
    """Promote a man to king if it reached the opponent's back row. Returns True if promoted."""
    row, _ = idx_to_rc(to_idx)
    if agent == "player_0" and row == 0 and board[to_idx] == 1:
        board[to_idx] = 2
        return True
    if agent == "player_1" and row == 5 and board[to_idx] == -1:
        board[to_idx] = -2
        return True
    return False


def env(**kwargs):
    e = raw_env(**kwargs)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class raw_env(AECEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "checkers_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_0", "player_1"]
        self.render_mode = render_mode
        self._current_jumper = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-2, high=2, shape=(NUM_SQUARES,), dtype=np.int8
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._current_jumper = None
        self.num_steps = 0

        self.board = np.zeros(NUM_SQUARES, dtype=np.int8)
        self.board[0:6] = -1  # player_1 men (rows 0-1)
        self.board[12:18] = 1  # player_0 men (rows 4-5)

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        jumper = self._current_jumper if agent == self.agent_selection else None
        return {
            "observation": self.board.copy(),
            "action_mask": compute_action_mask(self.board, agent, jumper),
        }

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        from_idx, to_idx = decode_action(action)

        # validate action before mutating the board
        legal_moves = get_legal_moves(self.board, agent, self._current_jumper)
        if (from_idx, to_idx) not in legal_moves:
            raise ValueError(
                f"Illegal action {action} (from={from_idx}, to={to_idx}) "
                f"for {agent}. Legal moves: {legal_moves}"
            )

        # move piece
        self.board[to_idx] = self.board[from_idx]
        self.board[from_idx] = 0

        # check if this was a capture (find jumped-over piece)
        was_capture = False
        for d, (adj, jump_sq) in NEIGHBORS[from_idx].items():
            if jump_sq == to_idx:
                self.board[adj] = 0
                was_capture = True
                self.rewards[agent] = 0.1
                break

        # king promotion
        promoted = maybe_promote(self.board, to_idx, agent)

        # multi-jump handling
        if was_capture and not promoted:
            further = get_capture_moves_from(self.board, agent, to_idx)
            if further:
                self._current_jumper = to_idx
                self._accumulate_rewards()
                if self.render_mode == "human":
                    self.render()
                return

        # turn is over; advance to next agent
        self._current_jumper = None
        self.num_steps += 1
        self.agent_selection = self._agent_selector.next()

        # win condition: next agent has no legal moves
        next_agent = self.agent_selection
        if not get_legal_moves(self.board, next_agent):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = self.rewards.get(agent, 0) + 1.0
            self.rewards[next_agent] = self.rewards.get(next_agent, 0) - 1.0

        # truncate long games
        self.truncations = {a: self.num_steps >= 200 for a in self.agents}

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            return
        symbols = {0: ".", 1: "b", 2: "B", -1: "r", -2: "R"}
        grid = [["." for _ in range(6)] for _ in range(6)]
        for i in range(NUM_SQUARES):
            row, col = idx_to_rc(i)
            grid[row][col] = symbols[self.board[i]]
        lines = ["  " + " ".join(str(c) for c in range(6))]
        for r, row_data in enumerate(grid):
            lines.append(f"{r} " + " ".join(row_data))
        output = "\n".join(lines) + "\n"
        if self.render_mode == "human":
            print(output)
        elif self.render_mode == "ansi":
            return output

    def close(self):
        pass


if __name__ == "__main__":
    from pettingzoo.test import api_test

    e = env(render_mode=None)
    api_test(e, num_cycles=100, verbose_progress=True)
