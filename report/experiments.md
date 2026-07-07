# 3. Experimental Investigation of Components

All experiments: single machine, CPython 3.9, one seed unless noted, time
budget 25-30 s per run (the project's target regime). Quality is reported as
profit or as %UB, where UB is the (loose) singleton-feasibility upper bound:
the profit sum of all customers servable as a one-customer tour.

## 3.1 Move evaluation acceleration (throughput ablation)

Iterations completed in a 30 s run of the full ALNS, n1000 instance
(skillvrp_n1000_v25_s8_k2.0_20):

| evaluation scheme                                    | iterations | best profit |
|------------------------------------------------------|-----------:|------------:|
| naive suffix re-simulation per candidate insertion   |          4 |      30 193 |
| + early-exit delta evaluation + move cache           |        517 |      33 526 |
| + O(1) forward-time-slack feasibility (final)        |      ~ 700 |      35 000+ |

The naive scheme cannot even leave the greedy start on large instances (the
first ALNS iterations consume the whole budget). The forward-time-slack fast
path is the decisive factor (~175x iteration throughput); quality follows
throughput directly.

## 3.2 Destroy-size distribution

Fixed vs. randomized destroy fraction (25 s, extras = 100):

| instance | fixed 0.15 | fixed 0.06 | U(0.04, 0.18) |
|----------|-----------:|-----------:|--------------:|
| n400     |     12 475 |     12 824 |        12 913 |
| n1000    |     35 355 |     34 838 |        35 032 |

No fixed fraction wins on both sizes, and no single randomized range does
either (a wide U(0.05, 0.28) later helped large but hurt small instances).
The final design makes the destroy size part of the **operator identity**:
each geometric destroy operator exists in a small (U(0.03, 0.12), polish)
and a large (U(0.12, 0.28), restructure) variant, and the adaptive weights
learn the right size mix per instance. This was robust across seeds:

| instance | uniform range | size-split variants (2 seeds) | baseline LNS |
|----------|--------------:|------------------------------:|-------------:|
| n100     |         2 910 |                 3 020 / 3 011 |        3 004 |
| n600     |        20 750 |               21 098 / 20 915 |       21 177 |
| n1000    |        36 170 |               36 216 / 35 929 |       35 781 |

## 3.3 Destroy operator portfolio

Adding the spatially-correlated RelatedRemoval (Shaw) and the
solution-dependent WorstDetourRemoval to the portfolio (random / static
worst-density / skill-scarcity), 30 s:

| instance | without | with    | baseline LNS |
|----------|--------:|--------:|-------------:|
| n100     |   2 823 |   2 876 |        3 004 |
| n250     |   9 450 |   9 787 |        9 874 |
| n800     |  24 829 |  25 713 |       25 491 |

Both operators improve every instance tested; correlated removal matters most
on mid/large instances, where removing a scattered sample never frees a
coherent region of a route.

## 3.4 Acceptance: temperature scale and schedule

Two design decisions dominate:

1. **T0 must be scaled to move deltas, not solution value.** Setting T0
   relative to the total start profit (Ropke-Pisinger style "w% of solution")
   makes the SA a random walk on large instances: deltas of destroy/repair
   cycles are sums of a few customer profits (10..100) regardless of n, while
   the total profit grows by 20x from n50 to n1000. T0 = 1.5 x mean customer
   profit works uniformly.
2. **Cooling must be time-based, not iteration-based.** Iteration throughput
   varies by ~2 orders of magnitude between n50 (~20 000 it/30 s) and n1000
   (~700 it/30 s); a fixed geometric rate per iteration is either frozen on
   small or never cools on large instances. We decay T from T0 to T_min
   exponentially in elapsed wall-clock fraction.

A T0-factor sweep (1x/2x mean profit, 25 s) changes results by only ~1%,
i.e., the *scale* is what matters, not the exact factor.

## 3.5 Repair operator portfolio

The best-first repairs (greedy / regret-2) re-select the globally best
candidate after every single insertion — high quality per repair, but
expensive. Adding a *sequential* cheapest-insertion repair (one pass over the
candidates in profit or random order, each inserted immediately at its
cheapest feasible position) roughly triples the achievable iteration count
and lets the adaptive weights trade care against throughput per instance
(25 s, destroy U(0.05, 0.28), max 70):

| instance | best-first only | + sequential | share sequential_profit |
|----------|----------------:|-------------:|------------------------:|
| n100     |           2 962 |        2 990 |                     40% |
| n250     |           9 694 |        9 976 |                     46% |
| n850     |           28 810 |       28 667 |                     43% |
| n1000    |          35 610 |       35 808 |                     46% |

The weight shares confirm the adaptation: the cheap sequential repair is
drawn most often on every instance, while regret-2 keeps a minor share.

A larger candidate pool was also tested and *rejected*: repairing from the
full unserved pool (instead of removed + 100 sampled extras) collapses the
iteration count (n850: 759 -> 214 iterations) and loses ~1 000 profit —
throughput beats per-repair completeness at this time budget.

## 3.6 Intensification

Return-to-best after 2 000 non-improving iterations gives small but mostly
positive changes (e.g., n100: 2 876 -> 2 997 in the best configuration) and
never hurt by more than noise; kept.
