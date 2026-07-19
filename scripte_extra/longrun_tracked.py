"""
Longrun mit ALNS-Tracking: 10 Minuten pro Instanz, Elbow-Charts als Output.

Laedt das beste Grid-Search-Setup aus gridsearch_top3_setups.json,
laeuft 600s pro Instanz mit run_alns_tracked und erstellt:
  - longrun_results/<instanz>_best_solution.csv   (Konvergenz-Zeitreihe)
  - longrun_results/<instanz>_operator_weights.csv
  - longrun_results/elbow_charts.png              (Plot fuer den Bericht)

Verwendung:
    python3 longrun_tracked.py

Laufzeit: 3 Instanzen x 10 min = ca. 30 Minuten.
(Alle 20 Instanzen: INSTANCES = None setzen, dann ~3.5 h)
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from alns import SAParams, ALNSParams, PenaltyParams
from alns_tracking import run_alns_tracked, make_tracking_params
from instance_reader import read_instance
from initial_solution import greedy_initial_solution
from make_operators_fixed import make_operators

# -----------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------

DATASET_DIR   = Path("../data_extracted/dataset")
TOP3_PATH     = Path("gridsearch_top3_setups.json")
OUT_DIR       = Path("longrun_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME       = 600.0    # 10 Minuten pro Instanz
SEED          = 1

# Alle Instanzen aus dataset/ (None = automatisch alle *.txt laden)
INSTANCES = None

BASELINE_SEC  = 30.0     # gestrichelte Linie im Plot

# -----------------------------------------------------------------------
# Setup laden
# -----------------------------------------------------------------------

with open(TOP3_PATH, encoding="utf-8") as f:
    best_setup = json.load(f)[0]
for key in ("base_fraction", "related_fraction"):
    if isinstance(best_setup.get(key), list):
        best_setup[key] = tuple(best_setup[key])

print(f"Setup: id={best_setup['setup_id']}  "
      f"temp_factor={best_setup['temp_factor']}  "
      f"profile={best_setup['destroy_profile']}  "
      f"avg_improvement={best_setup['avg_improvement_pct']:.1f}%")

SA = SAParams(
    initial_temperature=100.0,
    min_temperature=0.001,
    cooling_rate=0.9995,
    reheat_factor=0.75,
)

BASE_PARAMS = ALNSParams(
    random_seed=SEED,
    segment_length=int(best_setup.get("segment_length", 100)),
    no_accept_limit=int(best_setup.get("no_accept_limit", 700)),
    reaction_factor=float(best_setup.get("reaction_factor", 0.20)),
    min_operator_weight=float(best_setup.get("min_operator_weight", 0.05)),
    score_global_best=float(best_setup.get("score_global_best", 25.0)),
    score_better_current=float(best_setup.get("score_better_current", 10.0)),
    score_accepted=float(best_setup.get("score_accepted", 3.0)),
    score_rejected=float(best_setup.get("score_rejected", 0.0)),
    time_cost_alpha=float(best_setup.get("time_cost_alpha", 0.5)),
    time_scale_seconds=float(best_setup.get("time_scale_seconds", 0.01)),
    verbose=False,
    polish_interval_seconds=float(best_setup.get("polish_interval_seconds", 0.0)),
)

PENALTIES = PenaltyParams(
    time_window_penalty=float(best_setup.get("time_window_penalty", 20.0)),
    shift_penalty=float(best_setup.get("shift_penalty", 20.0)),
    skill_penalty=float(best_setup.get("skill_penalty", 10000.0)),
)

# min_operator_weight sehr niedrig + laengere Segmente -> Gewichtsdynamik
# wird im Plot sichtbar
TRACK_PARAMS = make_tracking_params(
    BASE_PARAMS,
    min_operator_weight=1e-3,
    segment_length=200,
)

# -----------------------------------------------------------------------
# Instanzliste aufbauen
# -----------------------------------------------------------------------

if INSTANCES is None:
    inst_files = sorted(DATASET_DIR.glob("skillvrp_*.txt"))
else:
    inst_files = [DATASET_DIR / name for name in INSTANCES]

# -----------------------------------------------------------------------
# Laeufe
# -----------------------------------------------------------------------

run_data = []   # (label, best_df) fuer den Plot

for inst_path in inst_files:
    label = inst_path.stem
    n_tag = label.split("_")[1]   # z. B. "n100"
    print(f"\n{'='*60}")
    print(f"Instanz: {label}  ({RUNTIME:.0f}s)")
    print(f"{'='*60}")

    inst  = read_instance(str(inst_path))
    start = greedy_initial_solution(inst, strategy="multi_start", verbose=False)
    print(f"Greedy: {start.objective}")

    # Starttemperatur skaliert mit dem Durchschnittsprofit der Startloesung
    avg_profit = start.objective / max(1, len(inst_files))
    sa_scaled  = replace(SA, initial_temperature=max(
        50.0, best_setup.get("temp_factor", 0.8) * start.objective * 0.005
    ))

    d_ops, r_ops = make_operators(inst, best_setup)

    result, tracker = run_alns_tracked(
        runtime=RUNTIME,
        inst=inst,
        start_solution=start,
        sa=sa_scaled,
        params=TRACK_PARAMS,
        penalties=PENALTIES,
        destroy_operators=d_ops,
        repair_operators=r_ops,
        csv_dir=str(OUT_DIR),
        csv_prefix=f"{label}_",
    )

    profit = result.best_evaluation.profit
    print(f"Ergebnis: {profit}  ({result.iterations} Iter, "
          f"feasible={result.best_evaluation.feasible})")

    # best_solution.csv einlesen fuer den Plot
    import csv as _csv
    best_csv = OUT_DIR / f"{label}_best_solution.csv"
    times, profits = [0.0], [start.objective]
    with open(best_csv) as f:
        for row in _csv.DictReader(f):
            times.append(float(row["elapsed_seconds"]))
            profits.append(int(row["objective_value"]))
    times.append(RUNTIME)
    profits.append(profit)

    run_data.append((n_tag, times, profits))

# -----------------------------------------------------------------------
# Elbow Charts
# -----------------------------------------------------------------------

n_plots = len(run_data)
n_cols  = min(4, n_plots)
n_rows  = (n_plots + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4 * n_rows),
                         sharey=False)
axes = list(axes.flatten()) if n_plots > 1 else [axes]
# leere Subplots ausblenden
for ax in axes[n_plots:]:
    ax.set_visible(False)
axes = axes[:n_plots]

import itertools
_palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7",
            "#e34948", "#008300", "#e87ba4", "#eb6834"]
COLORS = list(itertools.islice(itertools.cycle(_palette), n_plots))

for ax, (label, times, profits), color in zip(axes, run_data, COLORS):
    # Treppenfunktion: Profit bleibt konstant bis neue Bestloesung gefunden
    ax.step(times, profits, where="post",
            color=color, linewidth=2.2, label="best profit")
    ax.plot(times, profits, "o", color=color, markersize=4, zorder=5)

    # Gestrichelte Linie bei 30s (Baseline-Zeitbudget)
    ax.axvline(x=BASELINE_SEC, color="#888", linestyle="--",
               linewidth=1.4, label=f"{BASELINE_SEC:.0f}s baseline")

    # Profit-Wert an der 30s-Linie annotieren
    profit_at_30 = profits[0]
    for t, p in zip(times, profits):
        if t <= BASELINE_SEC:
            profit_at_30 = p
    ax.annotate(f"{profit_at_30}",
                xy=(BASELINE_SEC, profit_at_30),
                xytext=(BASELINE_SEC + RUNTIME * 0.03, profit_at_30),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Zeit (Sekunden)")
    ax.set_ylabel("Bester Profit" if ax == axes[0] else "")
    ax.legend(fontsize=8)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(left=0, right=RUNTIME)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}s"
    ))

fig.suptitle(
    f"Konvergenz: Bester Profit ueber die Zeit (10-Minuten-Lauf)\n"
    f"Gestrichelt = 30s-Baseline | Setup: {best_setup['destroy_profile']}, "
    f"temp={best_setup['temp_factor']}",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plot_path = OUT_DIR / "elbow_charts.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot gespeichert: {plot_path}")
print("Fertig.")
