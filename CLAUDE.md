# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
./.venv/bin/python pixel_cell_m1.py
```

This produces `pixel_cell_m1.gif` (default output). The `run()` function accepts parameters to customize the simulation:

```python
run(seed=3, H=160, W=160, T=800, capture_every=6, out_gif="pixel_cell_m1.gif")
```

## Environment

Dependencies are managed via `.venv/`. No `requirements.txt` exists — installed packages include `numpy`, `imageio`, `matplotlib`, and `pillow`.

## Architecture

Single-file project (`pixel_cell_m1.py`). It implements a biologically-inspired cellular automaton on a 2D grid with four coupled fields:

- **A** (Activity): fast-changing neural state, tanh-nonlinear, energy-gated
- **M** (Memory): slow exponential integrator of A
- **E** (Energy): diffuses spatially, refills over time, depleted by activity and learning
- **C** (Connectivity): synaptic weights to 4-neighbors; updated via Hebbian plasticity gated by a "curiosity" signal (local prediction error reduction)

Each timestep: energy diffuses → neighbors sensed → A updated → M updated → curiosity computed → C updated via plasticity → energy costs deducted → RGB frame captured.

**Rendering**: A→Red, M→Green, blend of C and E→Blue. Frames saved as GIF at 60ms/frame.

**Neighbor operations**: `lap()` computes discrete Laplacian via `np.roll`; `neigh_mean()` averages 4-connected neighbors.

**Embryo seed**: simulation starts with a circular high-activity patch at the grid center.
