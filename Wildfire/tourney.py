"""
tournament.py
=============
Run three independent SimulationEngine games, compare results, display
charts, and crown the winner with an ASCII trophy.


╔══════════════════════════════════════════════════════════════════════════╗
║                          AI USAGE DISCLOSURE                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Tool      : Claude (Anthropic) — claude-sonnet-4-6                      ║
║  Role      : Primary author                                              ║
║  Scope     : Substantially AI-generated                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Contributions                                                           ║
║  ─ Designed and wrote run_tournament() to construct a fresh engine for   ║
║    each run so maps and PRM roadmaps are independently randomised.       ║
║  ─ Authored summarise() score table with per-run winner column.          ║
║  ─ Designed the two-subplot matplotlib figure: grouped bar charts for    ║
║    CPU time and points with value labels, total annotations, and a       ║
║    shared legend.  Saves output to tournament_results.png.               ║
║  ─ Wrote the ASCII trophy system with three cases (firetruck win,        ║
║    wumpus win, draw) and a ★-bordered terminal output block.             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Human contributions                                                     ║
║  ─ Concept: 3-run tournament with side-by-side score comparison.         ║
║  ─ Scoring rules: firetruck +2/extinguish, wumpus +1/fire +1/burned.     ║
║  ─ Request for ASCII trophy and winner announcement in terminal output.  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import time
from typing import List

import matplotlib
matplotlib.use("TkAgg")          # change to "Agg" if running headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pathSimulator import SimulationEngine, RunResult

# ── Shared engine kwargs ────────────────────────────────────────────────────
# Adjust these to match your project's typical configuration.
DEFAULT_ENGINE_KWARGS = dict(
    grid_num              = 50,
    cell_size             = 5.0,
    fill_percent          = 0.10,
    firetruck_start       = (25.0, 25.0, 0.0),
    wumpus_start          = (220.0, 220.0),
    prm_nodes             = 300,
    tick_real_time        = 0.0,
    display_every_n_ticks = 5,
    plot                  = False,      # set True to watch the sim
    plot_prm              = False,
    sim_duration          = 3600.0,
    wumpus_catch_radius   = 6.0,
    flood_fill_radius     = 4,
    extinguish_margin     = 5.0,
    burn_lifetime         = 30.0,
)


# ===========================================================================
# Core tournament runner
# ===========================================================================

def run_tournament(n_runs: int = 3, **engine_kwargs) -> List[RunResult]:
    """
    Create and run `n_runs` independent simulations.

    Each run builds a fresh engine (new map, new PRM) so results are
    genuinely independent.  Returns a list of RunResult objects.
    """
    kwargs = {**DEFAULT_ENGINE_KWARGS, **engine_kwargs}
    results: List[RunResult] = []

    for i in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"  TOURNAMENT — Run {i} of {n_runs}")
        print(f"{'='*60}")
        wall_start = time.perf_counter()
        engine = SimulationEngine(**kwargs)
        result = engine.run(run_index=i)
        result_with_wall = result   # RunResult is a dataclass; we can annotate
        print(f"[Tournament] Run {i} wall-clock: "
              f"{time.perf_counter() - wall_start:.1f}s")
        results.append(result)

    return results


# ===========================================================================
# Scoring summary
# ===========================================================================

def summarise(results: List[RunResult]) -> None:
    """Print a per-run score table and declare the overall tournament winner."""
    total_ft = sum(r.firetruck_points for r in results)
    total_wu = sum(r.wumpus_points    for r in results)

    print("\n" + "═"*62)
    print("  TOURNAMENT SCORE TABLE")
    print("═"*62)
    print(f"  {'Run':<6} {'FT pts':>8} {'WU pts':>8}  {'Run winner':<14}")
    print("  " + "-"*58)
    for r in results:
        print(f"  {r.run_index:<6} {r.firetruck_points:>8} {r.wumpus_points:>8}  {r.winner:<14}")
    print("  " + "="*58)
    print(f"  {'TOTAL':<6} {total_ft:>8} {total_wu:>8}")
    print("═"*62 + "\n")

    if total_ft > total_wu:
        winner, loser = "Firetruck", "Wumpus"
        margin = total_ft - total_wu
    elif total_wu > total_ft:
        winner, loser = "Wumpus", "Firetruck"
        margin = total_wu - total_ft
    else:
        winner, loser = "Nobody (Draw)", ""
        margin = 0

    _print_trophy(winner, total_ft, total_wu, margin)


# ===========================================================================
# ASCII trophy
# ===========================================================================

_TROPHY = r"""
         ___________
        '._==_==_=_.'
        .-\:      /-.
       | (|:.     |) |
        '-|:.     |-'
          \::.    /
           '::. .'
             ) (
           _.' '._
          `"""""""`
"""

_FIREBALL = r"""
          )
         ) \
        / ) (
        \(_)/
"""

_WUMPUS_WIN = r"""
    /\  /\
   (  \/  )
    \    /
    /    \
   (  /\  )
    \/  \/
"""


def _print_trophy(winner: str, ft_pts: int, wu_pts: int, margin: int) -> None:
    border = "★" * 62
    print(border)

    if winner == "Nobody (Draw)":
        print("  🏳  IT'S A DRAW! Both agents scored equally.")
        print(f"  Firetruck: {ft_pts} pts   Wumpus: {wu_pts} pts")
        print(border)
        return

    icon = _TROPHY if winner == "Firetruck" else _WUMPUS_WIN
    lines = icon.strip("\n").splitlines()
    pad   = " " * 20
    for line in lines:
        print(pad + line)

    print()
    print(f"{'🏆  TOURNAMENT WINNER':^62}")
    print(f"{'':^62}")
    print(f"{winner:^62}")
    print(f"{'':^62}")
    print(f"  Firetruck total : {ft_pts} pts")
    print(f"  Wumpus total    : {wu_pts} pts")
    print(f"  Winning margin  : {margin} pts")
    print()
    print(border)


# ===========================================================================
# Charts
# ===========================================================================

_FT_COLOR  = "#e05c33"    # firetruck orange-red
_WU_COLOR  = "#4a90d9"    # wumpus blue
_BAR_WIDTH = 0.32


def plot_results(results: List[RunResult]) -> None:
    """
    Two-subplot figure:
      Left  — CPU time (s) per run, grouped bars (firetruck vs wumpus)
      Right — Points per run, grouped bars (firetruck vs wumpus)
    """
    n     = len(results)
    runs  = [f"Run {r.run_index}" for r in results]
    x     = np.arange(n)

    ft_cpu = [r.firetruck_cpu_time for r in results]
    wu_cpu = [r.wumpus_cpu_time    for r in results]
    ft_pts = [r.firetruck_points   for r in results]
    wu_pts = [r.wumpus_points      for r in results]

    fig, (ax_cpu, ax_pts) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"wspace": 0.35},
    )
    fig.suptitle("Wildfire Tournament — 3-Run Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── CPU time ─────────────────────────────────────────────────────────
    _grouped_bars(ax_cpu, x, ft_cpu, wu_cpu, _BAR_WIDTH,
                  title="Planning CPU Time",
                  ylabel="Wall-clock seconds",
                  run_labels=runs)
    ax_cpu.set_title("Planning CPU Time per Run", fontweight="bold")

    # Annotate total CPU times above the chart
    ax_cpu.text(
        0.5, 1.06,
        f"Totals — FT: {sum(ft_cpu):.2f}s  |  WU: {sum(wu_cpu):.2f}s",
        ha="center", va="bottom", transform=ax_cpu.transAxes,
        fontsize=9, color="gray",
    )

    # ── Points ────────────────────────────────────────────────────────────
    _grouped_bars(ax_pts, x, ft_pts, wu_pts, _BAR_WIDTH,
                  title="Points per Run",
                  ylabel="Points",
                  run_labels=runs)
    ax_pts.set_title("Points Scored per Run", fontweight="bold")
    ax_pts.text(
        0.5, 1.06,
        f"Totals — FT: {sum(ft_pts)} pts  |  WU: {sum(wu_pts)} pts",
        ha="center", va="bottom", transform=ax_pts.transAxes,
        fontsize=9, color="gray",
    )

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=_FT_COLOR, label="Firetruck"),
        mpatches.Patch(color=_WU_COLOR, label="Wumpus"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig("tournament_results.png", dpi=150, bbox_inches="tight")
    print("[Tournament] Chart saved to tournament_results.png")
    plt.show()


def _grouped_bars(ax, x, ft_vals, wu_vals, width, title, ylabel, run_labels):
    """Draw paired bars for firetruck and wumpus on a given axes."""
    bars_ft = ax.bar(x - width/2, ft_vals, width,
                     color=_FT_COLOR, label="Firetruck", zorder=3)
    bars_wu = ax.bar(x + width/2, wu_vals, width,
                     color=_WU_COLOR, label="Wumpus",    zorder=3)

    # Value labels centred on each bar
    for bar in (*bars_ft, *bars_wu):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(ft_vals + wu_vals) * 0.01,
            f"{h:.1f}" if isinstance(h, float) and h != int(h) else f"{int(h)}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    results = run_tournament(n_runs=3)
    summarise(results)
    plot_results(results)