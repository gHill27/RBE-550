# Wildfire Simulation

A multi-agent simulation of a firetruck versus a wumpus in a procedurally generated grid world. The firetruck uses a **Probabilistic Roadmap (PRM)** planner with **Reeds-Shepp curves** to navigate and extinguish fires, while the wumpus uses **grid-based A\*** to hunt for obstacles to burn. Three independent rounds are run as a tournament, with results compared via charts and a winner declared.

---

## 📸 Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     WILDFIRE SIMULATION                     │
│                                                             │
│   🚒 Firetruck  →  PRM + Reeds-Shepp planner                │
│                     navigates to fires, extinguishes them   │
│                                                             │
│   👾 Wumpus     →  Grid A* planner                          │
│                     hunts intact obstacles and burns them   │
│                                                             │
│   🗺  Map        →  Procedural tetromino obstacle grid      │
│                     fire spreads to neighbours over tim     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗 Project Structure

```
wildfire-sim/
│
├── Map_Generator.py        # World state: obstacle grid, fire spread, status machine
├── firetruck.py            # Nonholonomic PRM planner (Reeds-Shepp curves)
├── wumpus.py               # Grid A* agent — navigates to and burns obstacles
├── simulation_engine.py    # Orchestrator: tick loop, scoring, state machines
├── pathVisualizer.py       # Real-time matplotlib visualizer + end screen
├── tournament.py           # Runs 3 independent simulations, compares results
├── main.py                 # Runs all relevant code for 4 round tourney
│
└── tests/
    └── test_wildfire.py    # ~130 unit + integration tests (pytest)
```

---

## ⚙️ How It Works

### Map
- A square grid (`grid_num × grid_num` cells, each `cell_size` metres wide) is generated with tetromino-shaped obstacle clusters placed randomly at `fill_percent` density.
- Safe buffer zones around the firetruck and wumpus start positions are guaranteed to be clear.
- Each obstacle cell has one of four statuses: `INTACT → BURNING → EXTINGUISHED | BURNED`.
- Fire spreads to nearby cells after **10 seconds** of burning and the cell burns out to `BURNED` after **30 seconds**.

### Firetruck Agent
- Uses a **Probabilistic Roadmap (PRM)** built once at startup (`prm_nodes` samples in free configuration space).
- Edges are connected using **Reeds-Shepp curves** via the `rsplan` library, with Shapely-based collision checking at every waypoint.
- Planning runs in a **background daemon thread** so the main simulation loop is never blocked.
- A three-state machine governs behaviour: `idle → driving → suppressing → idle`.
- Fire targets are triaged by Euclidean distance and remaining burn time — viable fires (reachable before burnout) are prioritised over fallback fires.
- Extinguishes fires using a **proximity timer** (cells in range for `proximity_duration` seconds) plus a **BFS flood-fill** over connected burning cells up to `flood_fill_radius` steps.
- **Scoring:** +2 points per cell extinguished.

### Wumpus Agent
- Uses **grid A\*** with euclidean distance heuristic and 4-directional movement.
- Navigates toward the closest `INTACT` obstacle (stopping adjacent to it rather than inside it).
- Burns adjacent cells each tick — any `INTACT` neighbour becomes `BURNING`.
- Replans immediately when fires are newly lit or its current path is exhausted.
- **Scoring:** +1 point per fire started, +1 point per cell that burns out naturally.

### Tournament
- Three fully independent engines are constructed (new map, new PRM each run).
- Per-run and aggregate scores are printed to a table.
- A two-subplot matplotlib chart compares CPU planning time and points per run.
- An ASCII trophy is printed in the terminal for the overall winner.

---

## 🚀 Getting Started

### Requirements

```
Python 3.10+
```

### Install Dependencies

```bash
pip install matplotlib numpy scipy shapely rsplan
```

> `rsplan` provides Reeds-Shepp curve planning. Install it via pip or from source if not on PyPI.

### Run a Tournament (3 rounds)

```bash
python tournament.py
```

### Run a Single Simulation

```python
from simulation_engine import SimulationEngine

engine = SimulationEngine(
    grid_num       = 50,
    cell_size      = 5.0,
    fill_percent   = 0.10,
    firetruck_start= (25.0, 25.0, 0.0),
    wumpus_start   = (220.0, 220.0),
    prm_nodes      = 300,
    plot           = True,
    sim_duration   = 3600.0,
)
result = engine.run(run_index=1)
print(result)
```

### Run Tests

```bash
# From the project root
pytest tests/test_wildfire.py -v
```

---

## 🔧 Configuration

All parameters are set in `tournament.py`'s `DEFAULT_ENGINE_KWARGS` or passed directly to `SimulationEngine`.

| Parameter | Default | Description |
|---|---|---|
| `grid_num` | `50` | Grid cells per side |
| `cell_size` | `5.0` | Metres per grid cell |
| `fill_percent` | `0.10` | Fraction of cells filled with obstacles |
| `firetruck_start` | `(25, 25, 0)` | Firetruck start pose `(x, y, θ°)` |
| `wumpus_start` | `(220, 220)` | Wumpus start position `(x, y)` in metres |
| `prm_nodes` | `300` | Number of PRM roadmap samples |
| `sim_duration` | `3600.0` | Hard stop in simulation seconds |
| `tick_real_time` | `0.0` | Wall-clock sleep per tick (0 = max speed) |
| `display_every_n_ticks` | `5` | Render every N ticks (higher = faster) |
| `plot` | `False` | Enable real-time visualizer |
| `wumpus_catch_radius` | `6.0` | Distance (m) for wumpus-catch termination |
| `flood_fill_radius` | `4` | BFS depth for connected-obstacle extinguish |
| `burn_lifetime` | `30.0` | Seconds a cell burns before self-extinguishing |
| `extinguish_margin` | `5.0` | Extra seconds of burn budget required to target a fire |

---

## 📊 Scoring

| Event | Agent | Points |
|---|---|---|
| Cell extinguished by firetruck | 🚒 Firetruck | +2 |
| Fire started by wumpus | 👾 Wumpus | +1 |
| Cell burns out naturally (BURNED) | 👾 Wumpus | +1 |

The agent with the highest total points across all three rounds wins the tournament.

---

## 🖥 Visualizer

When `plot=True`, a real-time matplotlib window renders:

- **Obstacle colours** — green (intact), orange (burning, pulsing), blue (extinguished), charcoal (burned)
- **Firetruck** — amber rotated rectangle with heading arrow and cabin highlight
- **Wumpus** — red 5-pointed star
- **Paths** — dashed cyan (firetruck planned path), dotted pink (wumpus A* path)
- **HUD panel** — live sim clock, per-status obstacle counts, active fire count, agent poses
- **End screen** — triggered on wumpus-catch, overlays result text for 1 second then auto-closes

---

## 🧪 Test Coverage

Tests live in `tests/test_wildfire.py` and cover ~130 cases across:

- `Map_Generator` — status state machine, fire spread, map generation, goal logic
- `Wumpus` — A\* correctness, bounds checking, burn logic, closest-obstacle search
- `Firetruck` — PRM build, car geometry, collision space, A\* internals, SE(2) distance
- `SimulationEngine` — scoring, state machine, proximity extinguish, flood-fill, thread safety
- `Tournament` — summarise output, chart helpers, default config keys
- Integration — full short runs, invariant checks (`active_fires ⊆ obstacle_set`)

```bash
pytest tests/test_wildfire.py -v --tb=short
```

---

## 🤖 AI Usage Disclosure

This project was developed with **Claude (Anthropic) — claude-sonnet-4-6** as an implementation and debugging partner. Each source file contains a detailed disclosure block specifying which components were AI-assisted.

---

## 📄 License

MIT License — see `LICENSE` for details.
