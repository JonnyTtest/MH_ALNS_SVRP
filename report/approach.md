# 1. Approach Description

## 1.1 Overview

We solve the Prize-Collecting Skill-VRP with an **Adaptive Large Neighborhood
Search (ALNS)** in the style of Ropke & Pisinger (2006): starting from a greedy
initial solution, the search repeatedly *destroys* part of the incumbent
(removes served customers) and *repairs* it (re-inserts removed and further
unserved customers), accepting candidates with a simulated-annealing (SA)
criterion. Operator selection is adaptive: destroy and repair operators are
drawn with probabilities proportional to weights that are updated from their
recent success.

Pipeline (`skillvrp.py`):

1. Parse instance, precompute distance matrix, per-customer feasible-vehicle
   lists (skills + singleton time feasibility) and profit-density bounds.
2. Multi-start greedy insertion (4 orderings: hybrid, profit, density,
   fewest-vehicles) -> best feasible start solution.
3. ALNS until the time budget (minus a small output reserve) is exhausted.
4. Print the best feasible solution; the empty plan is the last-resort
   fallback, so output is always feasible.

## 1.2 Initial solution

Customers are sorted by one of four orderings and inserted one at a time at
the feasible position with minimum travel detour over all skill-compatible
couriers. The four orderings are all evaluated (cheap relative to the budget)
and the most profitable start is kept. Customers that no courier can serve
even as a singleton tour (skills or time windows) are excluded once and for
all.

## 1.3 Destroy operators

All destroy sizes q are sampled per call from a fraction range
U(0.04, 0.18) of currently served customers, capped at 50 — mixing small
"polishing" and large "restructuring" moves.

* **Random removal** — uniform sample of served customers (diversification).
* **Worst-density removal** — removes customers with the lowest lower-bound
  profit density (profit per minimal time to serve), with noise.
* **Skill-scarcity removal** — problem-specific: removes low-profit customers
  occupying couriers whose skill set is in high demand among *unserved*
  customers, freeing contested courier capacity that distance-based operators
  cannot "see".

## 1.4 Repair operators

* **Greedy best insertion** — repeatedly inserts the candidate with the
  highest delta score (profit; ties by minimal travel detour).
* **Regret-2 insertion** — prioritizes candidates whose second-best insertion
  is much worse than their best (customers about to lose their last good slot).
* **Sequential cheapest insertion** (profit / random order, plus a *noised*
  variant that perturbs the position ranking by +-10% of the maximum distance,
  Ropke & Pisinger 2006) — one cheap pass, trading per-repair optimality for
  iteration throughput.

Candidates are the removed customers plus 100 unserved extras (half top
profit density, half uniformly sampled). All operators only generate
*feasible* insertions.

## 1.5 Efficient move evaluation

Three accelerations keep iteration cost low enough for 1 000-customer
instances in Python:

1. **Route caches with forward time slack** (Savelsbergh 1992): every route
   stores arrival/service-start/departure times and the maximum delay each
   position tolerates. An insertion's feasibility is then checked in O(1)
   instead of re-simulating the route suffix.
2. **Per-(customer, vehicle) move cache**: inserting a customer changes only
   one route, so cached best moves on all other vehicles remain exact and only
   the modified vehicle's entries are recomputed (Ropke & Pisinger 2006).
3. **Early-exit delta evaluation** (fallback path for infeasible routes):
   the route suffix is only re-simulated until its timeline re-synchronizes
   with the cached one.

## 1.6 Acceptance and adaptivity

SA acceptance on the solution score with a **time-based cooling schedule**:
the temperature decays exponentially from T0 to T_min over the *wall-clock*
budget, making the schedule independent of iteration throughput (which varies
by two orders of magnitude between n=50 and n=1000). T0 is scaled to the
magnitude of a single move (2 x mean customer profit), not to the total
solution profit. If no candidate is accepted for 500 iterations, the search
restarts from a randomized greedy solution and reheats.

Operator weights are updated every 100 iterations from scores (new global
best: 25, better than current: 10, accepted: 3, rejected: 0), smoothed with
reaction factor 0.2 and normalized by the operator's average runtime, so slow
operators must earn their cost.

## 1.7 Local-search polish

Whenever a new global best is found (at most twice per second), every route is
compressed with **Or-opt** (segments of length 1-3 relocated within the route,
first improvement, feasibility-checked) and the freed shift capacity is
immediately refilled by a greedy insertion pass over the best unserved
candidates. Or-opt changes no profit by itself; its value is the freed
capacity, which regularly admits additional customers. The search then
continues from the compressed solution.
