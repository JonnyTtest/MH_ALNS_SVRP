"""
Korrigierte make_operators-Funktion fuer Grid Search und Tracking.

Alle 4 Bugs aus dem alten Grid-Search-Code sind gefixt:
  - HistoryBasedRemoval:         'noise'      -> 'selection_bias'
  - NoisyGreedyInsertionRepair:  'noise_amp'  -> 'noise'
  - ScarceSkillFirstRepair:      'max_insertions' entfernt (nicht vorhanden)
  - LastRemovedFirstInsertedRepair: 'max_insertions' entfernt (nicht vorhanden)

Verwendung im Notebook:
  1. Fuer Grid Search: make_operators(inst, setup)
  2. Fuer Tracking/Benchmark: make_operators(inst)
"""

from alns import (
    # Destroy
    RandomRemoval,
    WorstDensityRemoval,
    SkillScarcityRemoval,
    RelatedRemoval,
    WorstDetourRemoval,
    WorstDetourRemovalV2,
    ShawRelatedRemoval,
    RouteRemoval,
    TimeWindowSegmentRemoval,
    SequenceRemoval,
    ClusterRemoval,
    LargestSavingRemoval,
    TemporalShawRemoval,
    HistoryBasedRemoval,
    # Repair
    GreedyBestInsertionRepair,
    Regret2InsertionRepair,
    SequentialCheapestInsertionRepair,
    NoisyGreedyInsertionRepair,
    ScarceSkillFirstRepair,
    LastRemovedFirstInsertedRepair,
    RegretKInsertionRepair,
    RandomPositionInsertionRepair,
    ShawInsertionRepair,
)


def make_operators(inst, setup: dict = None):
    """
    Erstellt ein frisches, zustandsloses Operator-Set fuer einen ALNS-Lauf.

    Parameters
    ----------
    inst   : Instance-Objekt
    setup  : dict aus make_setup_grid() fuer Grid Search.
             Wenn None, werden vernuenftige Default-Werte verwendet.
    """
    # ------------------------------------------------------------------ #
    # Parameter aus Setup oder Defaults
    # ------------------------------------------------------------------ #
    n = inst.num_customers
    max_remove = max(5, int(n * (setup["max_remove_ratio"] if setup else 0.25)))

    bf  = setup["base_fraction"]     if setup else 0.15   # base destroy fraction
    rf  = setup["related_fraction"]  if setup else 0.20   # related/shaw fraction
    eul = setup["extra_unserved_limit"] if setup else 100  # repair candidate pool
    mds = setup.get("min_delta_score", 0.0) if setup else 0.0

    # ------------------------------------------------------------------ #
    # Destroy-Operatoren  (15 Stueck, alle korrekt instanziiert)
    # ------------------------------------------------------------------ #
    destroy_operators = [

        # --- Basis-Operatoren ---
        RandomRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove,
            initial_weight=1.0,
        ),
        WorstDensityRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove,
            noise=0.05, initial_weight=1.0,
        ),
        SkillScarcityRemoval(                          # euer eigener Operator
            fraction=bf, min_remove=1, max_remove=max_remove,
            noise=0.05, initial_weight=1.0,
        ),
        RelatedRemoval(
            fraction=rf, min_remove=1, max_remove=max_remove,
            bias=4.0, initial_weight=1.0,
        ),

        # --- Detour-basiert ---
        WorstDetourRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove,
            bias=4.0, initial_weight=1.0,
        ),
        WorstDetourRemovalV2(
            fraction=bf, min_remove=1, max_remove=max_remove,
            noise=0.05, selection_bias=3.0, initial_weight=1.0,
        ),

        # --- Shaw / verwandte Cluster ---
        ShawRelatedRemoval(
            fraction=rf, min_remove=1, max_remove=max_remove,
            p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25,
            initial_weight=1.0,
        ),
        # Time-based removal = Shaw nur mit Zeitterm (Demir et al. 2012)
        ShawRelatedRemoval(
            fraction=rf, min_remove=1, max_remove=max_remove,
            p_determinism=4.0, w_distance=0.0, w_time=1.0, w_skill=0.0,
            initial_weight=1.0, neighbor_limit=100,
        ),

        # --- Routen- / Fenster-basiert ---
        RouteRemoval(
            max_routes=2, selection_bias=3.0, initial_weight=1.0,
        ),
        TimeWindowSegmentRemoval(
            window_fraction=0.15, max_remove=max_remove, initial_weight=1.0,
        ),

        # --- Literatur-basierte Erweiterungen ---
        SequenceRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove, initial_weight=1.0,
        ),
        ClusterRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove, initial_weight=1.0,
        ),
        LargestSavingRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove,
            selection_bias=3.0, initial_weight=1.0,
        ),
        TemporalShawRemoval(
            fraction=rf, min_remove=1, max_remove=max_remove,
            p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25,
            initial_weight=1.0,
        ),
        # BUG FIX: 'selection_bias' statt 'noise'
        HistoryBasedRemoval(
            fraction=bf, min_remove=1, max_remove=max_remove,
            selection_bias=3.0, initial_weight=1.0,
        ),
    ]

    # ------------------------------------------------------------------ #
    # Repair-Operatoren  (9 Stueck, alle korrekt instanziiert)
    # ------------------------------------------------------------------ #
    repair_operators = [

        GreedyBestInsertionRepair(
            extra_unserved_limit=eul, max_insertions=None,
            min_delta_score=mds, initial_weight=1.0,
        ),
        Regret2InsertionRepair(
            extra_unserved_limit=eul, max_insertions=None,
            min_delta_score=mds, initial_weight=1.0,
        ),
        SequentialCheapestInsertionRepair(
            extra_unserved_limit=eul, order="profit",
            initial_weight=1.0,
        ),
        # BUG FIX: 'noise' statt 'noise_amp'
        NoisyGreedyInsertionRepair(
            extra_unserved_limit=eul, max_insertions=None,
            min_delta_score=mds, noise=0.10, initial_weight=1.0,
        ),
        # BUG FIX: kein 'max_insertions' Parameter
        ScarceSkillFirstRepair(
            extra_unserved_limit=eul,
            min_delta_score=mds, initial_weight=1.0,
        ),
            # BUG FIX: kein 'max_insertions' Parameter
        LastRemovedFirstInsertedRepair(
            extra_unserved_limit=eul,
            min_delta_score=mds, initial_weight=1.0,
        ),
        RegretKInsertionRepair(
            extra_unserved_limit=eul, max_insertions=None,
            min_delta_score=mds, k=3, initial_weight=1.0,
        ),
        RandomPositionInsertionRepair(
            extra_unserved_limit=eul, initial_weight=1.0,
        ),
        ShawInsertionRepair(
            extra_unserved_limit=eul, min_delta_score=mds,
            p_determinism=4.0, w_distance=0.5, w_time=0.25, w_skill=0.25,
            initial_weight=1.0,
        ),
    ]

    return destroy_operators, repair_operators
