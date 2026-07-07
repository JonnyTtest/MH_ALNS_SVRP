# Prize-Collecting Skill-VRP — ALNS Metaheuristic

Solution for the Metaheuristics project (31-MM34, SoSe 26): an Adaptive Large
Neighborhood Search for the prize-collecting Skill Vehicle Routing Problem
with time windows and courier shifts.

## Usage (as required by the project)

```
python3 skillvrp.py <path to instance> <timeout (s)>
```

Example: `python3 skillvrp.py instance.txt 30` — writes the solution to
stdout in the checker format. Requires Python >= 3.9, no third-party
packages.

## Files

| file                 | purpose                                                        |
|----------------------|----------------------------------------------------------------|
| `skillvrp.py`        | entry point: CLI, configuration, greedy start + ALNS, output   |
| `instance_reader.py` | instance parsing and derived data (distances, feasible-vehicle lists, bounds) |
| `initial_solution.py`| multi-start greedy insertion construction heuristic            |
| `alns.py`            | ALNS framework: route caches (forward time slack), destroy/repair operators, adaptive weights, SA acceptance |
| `solution.py`        | solution representation and checker-format output              |
| `benchmark.py`       | dev tool: run all instances, validate with checker, report %UB |
| `report/`            | report source material (approach, experiments)                 |

## Method summary

* **Initial solution**: multi-start greedy insertion (4 customer orderings),
  best kept.
* **Destroy** (size ~ U(0.05, 0.28) of served, capped at 70): random,
  worst-density (static), worst-detour (solution-dependent), Shaw-style
  related removal, and a problem-specific skill-scarcity removal.
* **Repair** (feasible insertions only): greedy best-first, regret-2, and
  single-pass sequential cheapest insertion (profit / random order).
* **Acceleration**: O(1) insertion feasibility via cached forward time
  slacks; per-(customer, vehicle) move cache invalidated per modified route;
  early-exit suffix evaluation as fallback.
* **Acceptance**: simulated annealing with time-based cooling (T0 scaled to
  move deltas), return-to-best intensification, randomized restarts.
* **Adaptivity**: operator weights updated every 100 iterations from scores,
  normalized by operator runtime.
