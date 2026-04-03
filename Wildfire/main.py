"""
run_sim.py
==========
Entry point for the Wildfire simulation.
All configuration lives here — everything else is handled by SimulationEngine.

Run with:
    python run_sim.py
"""

from pathSimulator import SimulationEngine
from tourney import run_tournament, summarise,plot_results



args = dict(
    grid_num              = 50,
    cell_size             = 5.0,
    fill_percent          = 0.10,
    firetruck_start       = (25.0, 25.0, 0.0),
    wumpus_start          = (220.0, 220.0),
    prm_nodes             = 500,
    tick_real_time        = 0.000,
    display_every_n_ticks = 5,
    plot                  = True,      # set True to watch the sim
    plot_prm              = False,
    sim_duration          = 3600.0,
    wumpus_catch_radius   = 8.0,
    flood_fill_radius     = 4,
    extinguish_margin     = 5.0,
    burn_lifetime         = 30.0,
)

if __name__ == "__main__":
    results = run_tournament(n_runs=3, **args)
    summarise(results)
    plot_results(results)