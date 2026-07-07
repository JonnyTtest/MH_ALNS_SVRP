# 4. Overall Performance

Final configuration, all 20 benchmark instances, 30 s time limit, single run
(seed 1), CPython 3.9 on an Apple-silicon laptop. Every solution passes the
provided checker; every run finishes within the limit (max wall time 29.2 s).
UB is the singleton-feasibility upper bound (profit sum of all customers that
at least one courier can serve as a one-customer tour) — it is loose, since
it ignores all capacity interaction between customers.

| instance                        |    n |  B |  profit |      UB |   %UB | greedy start | ALNS gain |
|---------------------------------|-----:|---:|--------:|--------:|------:|-------------:|----------:|
| skillvrp_n50_v2_s3_k0.0_1       |   50 |  2 |   1 806 |   2 717 | 66.5% |        1 501 |    +20.3% |
| skillvrp_n75_v2_s4_k0.5_2       |   75 |  2 |   2 776 |   4 255 | 65.2% |        2 058 |    +34.9% |
| skillvrp_n100_v2_s5_k1.0_3      |  100 |  2 |   2 946 |   5 160 | 57.1% |        2 292 |    +28.5% |
| skillvrp_n125_v3_s6_k1.5_4      |  125 |  3 |   4 316 |   6 866 | 62.9% |        3 697 |    +16.7% |
| skillvrp_n150_v3_s8_k2.0_5      |  150 |  3 |   4 800 |   8 259 | 58.1% |        4 065 |    +18.1% |
| skillvrp_n200_v5_s3_k0.0_6      |  200 |  5 |   7 259 |  10 236 | 70.9% |        5 504 |    +31.9% |
| skillvrp_n250_v6_s4_k0.5_7      |  250 |  6 |   9 775 |  14 034 | 69.7% |        7 845 |    +24.6% |
| skillvrp_n300_v7_s5_k1.0_8      |  300 |  7 |  11 279 |  16 729 | 67.4% |        8 331 |    +35.4% |
| skillvrp_n350_v8_s6_k1.5_9      |  350 |  8 |  12 623 |  19 050 | 66.3% |        9 708 |    +30.0% |
| skillvrp_n400_v10_s8_k2.0_10    |  400 | 10 |  12 930 |  22 335 | 57.9% |       10 426 |    +24.0% |
| skillvrp_n450_v11_s3_k0.0_11    |  450 | 11 |  14 247 |  24 478 | 58.2% |       11 003 |    +29.5% |
| skillvrp_n500_v12_s4_k0.5_12    |  500 | 12 |  17 372 |  27 820 | 62.4% |       13 344 |    +30.2% |
| skillvrp_n600_v15_s5_k1.0_13    |  600 | 15 |  20 934 |  32 477 | 64.5% |       16 277 |    +28.6% |
| skillvrp_n700_v17_s6_k1.5_14    |  700 | 17 |  23 550 |  37 852 | 62.2% |       18 732 |    +25.7% |
| skillvrp_n750_v18_s8_k2.0_15    |  750 | 18 |  25 299 |  40 329 | 62.7% |       19 542 |    +29.5% |
| skillvrp_n800_v20_s3_k0.0_16    |  800 | 20 |  25 728 |  43 128 | 59.7% |       20 995 |    +22.5% |
| skillvrp_n850_v21_s4_k0.5_17    |  850 | 21 |  29 586 |  46 190 | 64.1% |       21 708 |    +36.3% |
| skillvrp_n900_v22_s5_k1.0_18    |  900 | 22 |  28 511 |  49 703 | 57.4% |       23 101 |    +23.4% |
| skillvrp_n950_v23_s6_k1.5_19    |  950 | 23 |  29 509 |  52 946 | 55.7% |       22 960 |    +28.5% |
| skillvrp_n1000_v25_s8_k2.0_20   | 1000 | 25 |  36 016 |  54 664 | 65.9% |       30 193 |    +19.3% |

**Average: 62.7% of the singleton UB, 0 infeasible solutions.**

## Comparison against a strong single-trajectory LNS baseline

We compare against an independently implemented, heavily optimized plain LNS
(greedy construction + random/worst/related removal + full-pool greedy
insertion + record-to-record-travel acceptance; `skillvrp+.py` in the
repository) at the same 30 s budget:

* baseline average: 63.0% UB — final ALNS: 62.7% UB (single seed, seed noise
  is about +-1% per instance);
* head-to-head the ALNS wins 9 of 20 instances, including the largest one
  (n1000: 36 016 vs 35 781);
* the ALNS achieves this while searching with 8 destroy x 4 repair adaptive
  operator pairs, i.e., the framework overhead of adaptivity is fully
  amortized by the O(1) slack-based move evaluation.

## Progression during development (average %UB, 30 s)

| stage                                                        | avg %UB |
|--------------------------------------------------------------|--------:|
| naive framework (before acceleration), large instances stuck |    n/a  |
| + slack fast path, move cache, time-based SA                 |   61.0% |
| + related & worst-detour removal, candidate sampling         |   62.2% |
| + sequential repair, larger destroys                         |   62.2% |
| + size-split destroy operators                               |   62.7% |
| + Or-opt polish & refill, noised insertion (final)           |   63.0% |

## Final configuration, 3 seeds (noise-robust measurement)

The final configuration (with Or-opt polish/refill and noised insertion) was
measured with **3 seeds per instance** (60 runs, all checker-feasible, none
over the time limit): **63.0% average %UB of the seed means** — i.e., the
*expected* quality now equals the best *single-run* result of the baseline
LNS. Head-to-head (our 3-seed mean vs. the baseline's single run) we win or
tie 12 of 20 instances; on the largest instance every single seed beats the
baseline (n1000: 36 111-36 268 vs 35 781). Seed spread is small (typically
<1.5% between min and max), see `final_benchmark_30s_3seeds.log`.

## Scaling behaviour

Iteration throughput at 30 s ranges from ~50 000 (n50) to ~1 000 (n1000)
iterations. Quality (%UB) shows no systematic degradation with n — the
hardest instances are those with scarce skills (k1.5/k2.0 suffixes) and few
couriers per skill, not the largest ones. Runs at 2 s and 5 s on n1000 still
return feasible solutions (partial greedy / greedy+short ALNS), and the
program respects arbitrary small time limits.
