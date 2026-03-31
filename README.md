# 6x6 Checkers

This environment implements a 6x6 checkers game as a PettingZoo AEC (Agent Environment Cycle) environment.

|                        |                                    |
| ---------------------- | ---------------------------------- |
| **Import**             | `from mycheckersenv import env`    |
| **Actions**            | Discrete                           |
| **Parallel API**       | No                                 |
| **Manual Control**     | No                                 |
| **Agents**             | `agents= ['player_0', 'player_1']` |
| **Agents**             | 2                                  |
| **Action Shape**       | (1,)                               |
| **Action Values**      | Discrete(324)                      |
| **Observation Shape**  | (18,)                              |
| **Observation Values** | [-2, 2]                            |

6x6 Checkers is a 2-player turn-based game played on the dark squares of a 6x6 board. Each player starts with 6 men on their two home rows. Players move diagonally forward, and captures are mandatory. A man that reaches the opponent's back row is promoted to a king, which can move diagonally in all four directions. Multi-jump captures are supported: if a piece can continue capturing after a jump, it must do so before the turn passes.

## Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation is a flat array of 18 integers representing the 18 dark squares of the 6x6 board. Only dark squares are playable in checkers -on a 6x6 board there are 3 dark squares per row, giving 18 total. Each value encodes the piece at that square:

| Value | Meaning       |
| ----- | ------------- |
| `1`   | player_0 man  |
| `2`   | player_0 king |
| `-1`  | player_1 man  |
| `-2`  | player_1 king |
| `0`   | empty square  |

Squares are indexed row by row from the top-left. For example, the initial board layout maps to:

```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
Row:    0  0  0  1  1  1  2  2  2  3  3  3  4  4  4  5  5  5
Value: -1 -1 -1 -1 -1 -1  0  0  0  0  0  0  1  1  1  1  1  1
```

Indices 0-5 hold player_1's starting men (rows 0-1), indices 6-11 are empty (rows 2-3), and indices 12-17 hold player_0's starting men (rows 4-5). The observation is shared, both players see the same board with the same sign conventions.

### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector of length 324 where each index represents whether the corresponding action is legal. When captures are available, only capture moves are marked as legal (mandatory capture rule). During a multi-jump sequence, only continuation captures from the jumping piece are legal.

## Action Space

The action space is Discrete(324), representing all possible (from, to) pairs of dark-square indices. An action is encoded as:

```
action = from_idx * 18 + to_idx
```

where `from_idx` is the source square and `to_idx` is the destination square, both in [0, 17]. To decode:

```
from_idx = action // 18
to_idx   = action % 18
```

For example, action `225` means moving from square `12` (row 4, col 1) to square `9` (row 3, col 0). Not all 324 actions are valid at any given time -the action mask indicates which moves are currently legal. Simple moves go to an adjacent diagonal square, while captures jump over an opponent's piece to a square two diagonals away.

## Rewards

| Event                           | Reward |
| ------------------------------- | ------ |
| Capturing an opponent's piece   | +0.1   |
| Winning (opponent has no moves) | +1.0   |
| Losing                          | -1.0   |
| Simple move / draw              | 0      |

Capture and win rewards are additive: a capture that also wins the game yields +1.1.

## Termination and Truncation

The game terminates when a player has no legal moves on their turn. That player loses (-1.0 reward) and the opponent wins (+1.0 reward).

The game is truncated (forced draw) if 200 full turns have elapsed without a winner. Both players receive 0 reward on truncation.

## Usage

### AEC

```python
from mycheckersenv import env

e = env(render_mode="ansi")
e.reset(seed=42)

for agent in e.agent_iter():
    observation, reward, termination, truncation, info = e.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = e.action_space(agent).sample(mask)

    e.step(action)
e.close()
```

## API

_class_ `mycheckersenv.env(**kwargs)`

_class_ `mycheckersenv.raw_env(render_mode: str | None = None)`

### `raw_env.action_space(agent)`

Takes in agent and returns the action space for that agent.

MUST return the same value for the same agent name.

---

### `raw_env.close()`

Closes any resources that should be released.

---

### `raw_env.observation_space(agent)`

Takes in agent and returns the observation space for that agent.

MUST return the same value for the same agent name.

---

### `raw_env.observe(agent)`

Returns the observation an agent currently can make.

`last()` calls this function.

---

### `raw_env.render()`

Renders the environment as specified by `self.render_mode`.

Render mode can be `'human'` to print the board to the console, or `'ansi'` to return the board as a string.

---

### `raw_env.reset(seed=None, options=None)`

Resets the environment to a starting state.

---

### `raw_env.step(action)`

Accepts and executes the action of the current `agent_selection` in the environment.

Automatically switches control to the next agent. Raises `ValueError` if the action is illegal.
