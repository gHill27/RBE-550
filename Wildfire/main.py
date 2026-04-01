"""
run_sim.py
==========
Entry point for the Wildfire simulation.
All configuration lives here — everything else is handled by SimulationEngine.

Run with:
    python run_sim.py
"""

from pathSimulator import SimulationEngine

engine = SimulationEngine(
    grid_num        = 50,           # grid cells per side
    cell_size       = 5.0,          # metres per cell
    fill_percent    = 0.1,         # fraction of cells that are obstacles
    firetruck_start = (25.0, 25.0, 0.0),   # (x_m, y_m, theta_deg)
    wumpus_start    = (220.0, 220.0),       # (x_m, y_m)
    prm_nodes       = 500,          # PRM roadmap size
    tick_real_time  = 0.005,         # wall-clock seconds per tick
    plot            = False,
    sim_duration    = 3600.0,       # 1 hour of sim time
)
engine.run()