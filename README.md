# Snake Reinforcement Learning

A Python learning project that trains an agent to play Snake using a small PyTorch Q-network. Watch the game in Pygame while a chart tracks each episode's score and the running average.

## How it works

```text
Observe the game -> choose an action -> receive a reward
        ^                                     |
        |________ update the Q-network _______|
```

- **State:** 11 features describing nearby danger, travel direction, and food position.
- **Actions:** continue straight, turn right, or turn left.
- **Network:** 11 inputs, a 256-unit hidden layer with ReLU, and 3 action values.
- **Rewards:** +10 for food, -10 for collision or the frame limit, and 0 otherwise.
- **Learning:** Adam and mean squared error, with a learning rate of 0.001 and discount factor of 0.9.
- **Replay:** remembers up to 100,000 transitions and samples up to 1,000 after an episode. It also trains on each individual transition.

The random-action threshold decreases with the number of completed games. Training continues until you stop it.

## Run locally

Requirements: Git, Python **3.13 or later**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a desktop session capable of displaying Pygame and Matplotlib windows.

```sh
git clone https://github.com/denyelcnamgyal42-hash/snake_game_reinforcement_learning.git
cd snake_game_reinforcement_learning
uv sync --locked
uv run agent.py
```

Run these commands from the repository root: the game loads the bundled `arial.ttf` using a relative path. The dependency install includes PyTorch and can take some time.

You should see the Snake window, score/average-score plots, and terminal logs reporting the game number, score, and record. Close the Snake window or press Ctrl+C in the terminal to stop.

## Training controls and saved models

| Setting | Location | Default |
| --- | --- | --- |
| Learning rate | `agent.py`: `LR` | `0.001` |
| Replay capacity | `agent.py`: `MAX_MEMORY` | `100000` |
| Replay batch size | `agent.py`: `BATCH_SIZE` | `1000` |
| Game speed | `game.py`: `SPEED` | `100` frames/second |
| Per-frame debug logs | `agent.py`: `SnakeGameAI(debug=True)` | Enabled |

Set `debug=False` to reduce terminal output. The `SPEED` constant in `game.py` controls the displayed game; the similarly named constant in `agent.py` is not used by the game loop.

When an episode exceeds the current run's record, the network weights are saved to `model/model.pth`. **Back up that file before training if you want to preserve the supplied checkpoint.** Starting `agent.py` creates a fresh network; it does not automatically load the saved model or resume previous training.

## Project map

| File | Purpose |
| --- | --- |
| [agent.py](agent.py) | State encoding, action selection, memory, and training loop |
| [game.py](game.py) | Pygame environment, movement, collision detection, and rewards |
| [model.py](model.py) | Q-network, optimizer, and checkpoint saving |
| [helper.py](helper.py) | Score and running-average plots |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | Python dependencies and locked environment |

## Limitations and next experiments

This is an educational implementation, not a benchmark result. Runs are not seeded, so scores vary. It uses the same network for current and next-state value estimates and does not implement a separate target network. There is no evaluation-only command or automatic checkpoint loading.

Useful next experiments include adding reproducible seeds, separating training from evaluation, implementing a target network, and comparing reward or exploration schedules.

If the font cannot be found, check your working directory. If no game or chart window appears, use a desktop environment with GUI support; plotting behavior also depends on the Matplotlib backend.
