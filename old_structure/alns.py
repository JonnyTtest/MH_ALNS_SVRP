from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
import time

from solution import Solution
from initial_solution import find_best_insertion, insert_customer as greedy_insert_customer



# Parameterclasses


@dataclass(slots=True)
class SAParams:
    initial_temperature: float
    min_temperature: float
    cooling_rate: float
    reheat_factor: float


@dataclass(slots=True)
class PenaltyParams:
    time_window_penalty: float
    shift_penalty: float
    skill_penalty: float


@dataclass(slots=True)
class ALNSParams:
    random_seed: int
    segment_length: int
    no_accept_limit: int

    reaction_factor: float
    min_operator_weight: float

    score_global_best: float
    score_better_current: float
    score_accepted: float
    score_rejected: float

    time_cost_alpha: float
    time_scale_seconds: float

    verbose: bool

    # Local-search polish: on a new global best (at most once per this many
    # seconds), Or-opt every route and refill freed capacity (0 = disabled).
    polish_interval_seconds: float = 0.0



# Cached evaluation

@dataclass(slots=True)
class RouteCache:
    arrival: list[int]
    service_start: list[int]
    departure: list[int]
    prefix_penalty: list[float]
    end_time: int
    penalty: float
    # Savelsbergh forward time slack per position; only computed (non-empty)
    # when the route is feasible (penalty == 0). slack[i] = max delay of the
    # service start at position i that keeps positions i..end and the return
    # to the depot feasible. Enables O(1) insertion feasibility checks.
    slack: list[int]


@dataclass(slots=True)
class SolutionEvaluation:
    feasible: bool
    profit: int
    penalty: float
    score: float
    served_customers: int


def build_route_cache(inst, vehicle: int, route: list[int], penalties: PenaltyParams) -> RouteCache:
    arrival_list = []
    service_start_list = []
    departure_list = []
    prefix_penalty = [0.0]
    time_now = inst.vehicle_start[vehicle]
    previous = 0
    penalty = 0.0
    vehicle_skills = inst.vehicle_skills[vehicle]
    distance = inst.distance


    for customer in route:
        if not inst.required_skills[customer].issubset(vehicle_skills):
            penalty += penalties.skill_penalty

        arrival = time_now + distance[previous][customer]

        if arrival > inst.due[customer]:
            penalty += penalties.time_window_penalty * (arrival - inst.due[customer])

        service_start = max(arrival, inst.ready[customer])
        departure = service_start + inst.service[customer]

        arrival_list.append(arrival)
        service_start_list.append(service_start)
        departure_list.append(departure)
        prefix_penalty.append(penalty)

        time_now = departure
        previous = customer

    end_time = time_now + distance[previous][0]

    if end_time > inst.vehicle_end[vehicle]:
        penalty += penalties.shift_penalty * (end_time - inst.vehicle_end[vehicle])

    slack: list[int] = []

    if penalty == 0.0 and route:
        m = len(route)
        slack = [0] * m
        slack_end = inst.vehicle_end[vehicle] - end_time
        slack[m - 1] = min(inst.due[route[m - 1]] - service_start_list[m - 1], slack_end)

        for i in range(m - 2, -1, -1):
            wait_next = service_start_list[i + 1] - arrival_list[i + 1]
            slack[i] = min(
                inst.due[route[i]] - service_start_list[i],
                wait_next + slack[i + 1],
            )

    return RouteCache(
        arrival=arrival_list,
        service_start=service_start_list,
        departure=departure_list,
        prefix_penalty=prefix_penalty,
        end_time=end_time,
        penalty=penalty,
        slack=slack,
    )


def initialize_solution_state(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
) -> SolutionEvaluation:

    route_cache = []
    total_penalty = 0.0

    for vehicle, route in enumerate(solution.routes):
        cache = build_route_cache(inst, vehicle, route, penalties)
        route_cache.append(cache)
        total_penalty += cache.penalty

    solution.route_cache = route_cache
    solution.total_penalty = total_penalty
    solution.score = solution.objective - total_penalty
    solution.feasible = total_penalty == 0.0

    return evaluation_from_solution(inst, solution)


def evaluation_from_solution(inst, solution: Solution) -> SolutionEvaluation:
    return SolutionEvaluation(
        feasible=solution.feasible,
        profit=solution.objective,
        penalty=solution.total_penalty,
        score=solution.score,
        served_customers=inst.num_customers - len(solution.unserved),
    )


def copy_solution_with_state(solution: Solution) -> Solution:
    new_solution = solution.copy()
    if hasattr(solution, "route_cache"):
        new_solution.route_cache = list(solution.route_cache)
        new_solution.total_penalty = solution.total_penalty
        new_solution.score = solution.score
        new_solution.feasible = solution.feasible

    return new_solution


def update_route_cache(
    inst,
    solution: Solution,
    vehicle: int,
    penalties: PenaltyParams,
) -> None:
    old_penalty = solution.route_cache[vehicle].penalty
    new_cache = build_route_cache(inst, vehicle, solution.routes[vehicle], penalties)
    solution.route_cache[vehicle] = new_cache
    solution.total_penalty += new_cache.penalty - old_penalty
    solution.score = solution.objective - solution.total_penalty
    solution.feasible = solution.total_penalty == 0.0


def full_evaluate_solution(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
) -> SolutionEvaluation:

    penalty = 0.0
    seen = set()

    for vehicle, route in enumerate(solution.routes):
        cache = build_route_cache(inst, vehicle, route, penalties)
        penalty += cache.penalty

        for customer in route:
            if customer < 1 or customer > inst.num_customers:
                penalty += penalties.skill_penalty
            elif customer in seen:
                penalty += penalties.skill_penalty
            else:
                seen.add(customer)

    score = solution.objective - penalty


    return SolutionEvaluation(
        feasible=penalty == 0.0,
        profit=solution.objective,
        penalty=penalty,
        score=score,
        served_customers=len(seen),
    )



# Solution operations

def remove_customer(solution: Solution, inst, customer: int) -> tuple[int, int]:
    vehicle = solution.customer_to_vehicle[customer]
    position = solution.routes[vehicle].index(customer)
    solution.routes[vehicle].pop(position)
    solution.unserved.add(customer)
    del solution.customer_to_vehicle[customer]
    solution.objective -= inst.profit[customer]
    return vehicle, position


def apply_insertion(
    solution: Solution,
    inst,
    penalties: PenaltyParams,
    move: "InsertionMove",
) -> None:
    solution.routes[move.vehicle].insert(move.position, move.customer)
    solution.unserved.remove(move.customer)
    solution.customer_to_vehicle[move.customer] = move.vehicle
    solution.objective += inst.profit[move.customer]
    update_route_cache(inst, solution, move.vehicle, penalties)



def served_customers(solution: Solution) -> list[int]:
    return [c for route in solution.routes for c in route]


# Random restart solution ()

def random_start_solution(
    inst,
    rng: random.Random,
    penalties: PenaltyParams,
    deadline: float | None = None,
) -> Solution:
    solution = Solution.empty(inst)

    customers = [
        c for c in inst.customers
        if inst.solo_feasible_vehicles[c]
    ]

    rng.shuffle(customers)

    for index, customer in enumerate(customers):
        if (
            deadline is not None
            and (index & 31) == 0
            and time.perf_counter() > deadline
        ):
            break

        insertion = find_best_insertion(inst, solution.routes, customer)

        if insertion is not None:
            greedy_insert_customer(solution, inst, insertion)

    initialize_solution_state(inst, solution, penalties)
    return solution



# Destroy / repair data

@dataclass(slots=True)
class DestroyResult:
    partial_solution: Solution
    removed_customers: list[int]
    origin_vehicle: dict[int, int]
    origin_position: dict[int, int]
    affected_vehicles: set[int]
    tags: set[str]


@dataclass(slots=True)
class InsertionMove:
    customer: int
    vehicle: int
    position: int
    delta_score: float
    travel_delta: int
    new_route_penalty: float



# Operator base classes


class ALNSOperator:
    def __init__(self, name: str, initial_weight: float):
        self.name = name
        self.weight = initial_weight
        self.total_uses = 0
        self.total_time = 0.0
        self.segment_uses = 0
        self.segment_score = 0.0
        self.segment_time = 0.0


    def observe(self, score: float, elapsed: float) -> None:
        self.total_uses += 1
        self.total_time += elapsed
        self.segment_uses += 1
        self.segment_score += score
        self.segment_time += elapsed

    def update_weight(self, params: ALNSParams) -> None:
        if self.segment_uses == 0:
            return

        avg_score = self.segment_score / self.segment_uses
        avg_time = self.segment_time / self.segment_uses

        time_factor = 1.0

        if params.time_cost_alpha > 0.0:
            time_factor += (
                avg_time / max(1e-12, params.time_scale_seconds)
            ) ** params.time_cost_alpha

        utility = avg_score / time_factor

        self.weight = (
            (1.0 - params.reaction_factor) * self.weight
            + params.reaction_factor * utility
        )

        self.weight = max(params.min_operator_weight, self.weight)
        self.segment_uses = 0
        self.segment_score = 0.0
        self.segment_time = 0.0


class DestroyOperator(ALNSOperator):
    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        raise NotImplementedError


class RepairOperator(ALNSOperator):
    requires: set[str]

    def compatible_with(self, destroy_result: DestroyResult) -> bool:
        return self.requires.issubset(destroy_result.tags)

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        raise NotImplementedError



# Destroy helpers


def number_to_remove(
    n_served: int,
    fraction,
    min_remove: int,
    max_remove: int,
    rng: random.Random | None = None,
) -> int:
    """`fraction` is either a fixed float or a (lo, hi) range sampled per call,
    which diversifies the destroy size (small repairs polish, large ones
    restructure)."""
    if isinstance(fraction, tuple):
        lo, hi = fraction
        f = rng.uniform(lo, hi) if rng is not None else (lo + hi) / 2.0
    else:
        f = fraction

    q = round(f * n_served)
    q = max(min_remove, q)
    q = min(max_remove, q)
    q = min(n_served, q)
    return q


def build_destroy_result(
    inst,
    solution: Solution,
    removed: list[int],
    tag: str,
    penalties: PenaltyParams,
) -> DestroyResult:
    new_solution = copy_solution_with_state(solution)
    origin_vehicle = {}
    origin_position = {}
    affected_vehicles = set()

    for customer in removed:
        vehicle, position = remove_customer(new_solution, inst, customer)
        origin_vehicle[customer] = vehicle
        origin_position[customer] = position
        affected_vehicles.add(vehicle)

    for vehicle in affected_vehicles:
        update_route_cache(inst, new_solution, vehicle, penalties)

    return DestroyResult(
        partial_solution=new_solution,
        removed_customers=removed,
        origin_vehicle=origin_vehicle,
        origin_position=origin_position,
        affected_vehicles=affected_vehicles,
        tags={"customer_pool", "origin_info", "affected_routes", tag},
    )



###########################################

# Destroy operators

class RandomRemoval(DestroyOperator):
    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        initial_weight: float,
    ):
        super().__init__("random_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        # Fallback: should not normally trigger, an empty solution has nothing to destroy.
        if not served:
            return build_destroy_result(inst, solution, [], "random_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        removed = rng.sample(served, q)

        return build_destroy_result(inst, solution, removed, "random_removal", penalties)






class WorstDensityRemoval(DestroyOperator):
    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        noise: float,
        initial_weight: float,
    ):
        super().__init__("worst_density_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.noise = noise

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "worst_density_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        ranked = sorted(
            served,
            key=lambda c: inst.profit_density_lb[c] + self.noise * rng.random(),
        )

        removed = ranked[:q]

        return build_destroy_result(inst, solution, removed, "worst_density_removal", penalties)


class SkillScarcityRemoval(DestroyOperator):
    """Removes served customers that are occupying a courier whose skill set
    is in high demand among the currently-unserved customers, weighted by
    how little profit they themselves contribute.

    RandomRemoval and WorstDensityRemoval only ever reason about travel
    distance/profit density; neither looks at skills at all, even though
    skill compatibility is the defining constraint of this problem variant.
    The idea here: if a courier with a rare skill set is currently tied up
    serving a low-profit customer while several unserved customers would
    also need exactly that skill set, that courier's time is a contested
    resource. Removing the low-profit occupant gives the repair step a
    chance to reassign that capacity to a more profitable customer who
    needs the same scarce skill -- a swap that distance-based operators
    cannot "see" because it has nothing to do with route geometry.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        noise: float,
        initial_weight: float,
    ):
        super().__init__("skill_scarcity_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.noise = noise

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "skill_scarcity_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        # demand_per_vehicle[b] = how many currently-unserved customers could
        # courier b serve, skill-wise. High demand -> b's time is contested.
        demand_per_vehicle = [0] * inst.num_vehicles
        for c in solution.unserved:
            for vehicle in inst.skill_feasible_vehicles[c]:
                demand_per_vehicle[vehicle] += 1

        def scarcity_score(c: int) -> float:
            vehicle = solution.customer_to_vehicle[c]
            demand = demand_per_vehicle[vehicle]
            return inst.profit[c] / (1.0 + demand)

        # Low score first: low profit on a high-demand courier is removed first.
        ranked = sorted(
            served,
            key=lambda c: scarcity_score(c) + self.noise * rng.random(),
        )

        removed = ranked[:q]

        return build_destroy_result(inst, solution, removed, "skill_scarcity_removal", penalties)



class RelatedRemoval(DestroyOperator):
    """Shaw-style related removal: picks a random served seed customer and
    removes it together with its nearest served neighbors (biased sampling).
    Spatially correlated removals give the repair step a real chance to
    re-route a whole neighborhood instead of scattered singletons."""

    def __init__(self, fraction, min_remove, max_remove, bias, initial_weight):
        super().__init__("related_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.bias = bias  # >1: strongly prefer the closest neighbors

    def apply(self, inst, solution, rng, penalties):
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "related_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        seed = rng.choice(served)
        served_set = set(served)
        removed = [seed]
        served_set.discard(seed)

        # nearest_customers[seed] is precomputed by increasing distance.
        neighbors = [c for c in inst.nearest_customers[seed] if c in served_set]

        while len(removed) < q and neighbors:
            index = int((rng.random() ** self.bias) * len(neighbors))
            removed.append(neighbors.pop(index))

        return build_destroy_result(inst, solution, removed, "related_removal", penalties)


class WorstDetourRemoval(DestroyOperator):
    """Solution-dependent worst removal: removes customers with the poorest
    profit per travel detour in their *current* route position (biased
    sampling, Ropke & Pisinger 2006). Complements the static density-based
    ranking, which never changes between iterations."""

    def __init__(self, fraction, min_remove, max_remove, bias, initial_weight):
        super().__init__("worst_detour_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.bias = bias

    def apply(self, inst, solution, rng, penalties):
        distance = inst.distance
        scored = []

        for route in solution.routes:
            last = len(route) - 1
            for i, customer in enumerate(route):
                prev_node = route[i - 1] if i > 0 else 0
                next_node = route[i + 1] if i < last else 0
                detour = (
                    distance[prev_node][customer]
                    + distance[customer][next_node]
                    - distance[prev_node][next_node]
                )
                # rounded distances can violate the triangle inequality
                ratio = inst.profit[customer] / (max(detour, 0) + 1.0)
                scored.append((ratio, customer))

        if not scored:
            return build_destroy_result(inst, solution, [], "worst_detour_removal", penalties)

        q = number_to_remove(
            n_served=len(scored),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        scored.sort()  # ascending ratio: worst first
        pool = [customer for _, customer in scored]
        removed = []

        while len(removed) < q and pool:
            index = int((rng.random() ** self.bias) * len(pool))
            removed.append(pool.pop(index))

        return build_destroy_result(inst, solution, removed, "worst_detour_removal", penalties)


class WorstDetourRemovalV2(DestroyOperator):
    """alns.py variant of the worst-detour removal, kept alongside the
    alns(1) WorstDetourRemoval above.

    True worst removal: ranks served customers by profit per unit of travel
    time actually saved by removing them from the CURRENT solution.

    Details:
    * detour can be slightly negative because rounded Euclidean distances
      may violate the triangle inequality -> clamp the denominator.
    * noise is multiplicative (scale-free), unlike additive noise whose
      effect would depend on the instance's profit scale.
    * selection draws with bias rand()^selection_bias from the worst end of
      the ranking instead of a hard cutoff, so repeated calls do not remove
      identical sets.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        noise: float,
        initial_weight: float,
        selection_bias: float = 3.0,
    ):
        super().__init__("worst_detour_removal_v2", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.noise = noise
        self.selection_bias = selection_bias

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "worst_detour_removal_v2", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        scored = []

        for route in solution.routes:
            n_route = len(route)

            for position, customer in enumerate(route):
                previous = 0 if position == 0 else route[position - 1]
                next_node = 0 if position == n_route - 1 else route[position + 1]

                detour = (
                    inst.distance[previous][customer]
                    + inst.distance[customer][next_node]
                    - inst.distance[previous][next_node]
                )

                ratio = inst.profit[customer] / (max(detour, 0) + 1.0)

                noise_factor = 1.0 + self.noise * (2.0 * rng.random() - 1.0)
                scored.append((ratio * noise_factor, customer))

        # Ascending: lowest profit-per-saved-time first == worst first.
        scored.sort(key=lambda item: item[0])
        pool = [customer for _, customer in scored]

        removed = []

        while len(removed) < q and pool:
            index = int((rng.random() ** self.selection_bias) * len(pool))
            index = min(index, len(pool) - 1)
            removed.append(pool.pop(index))

        return build_destroy_result(inst, solution, removed, "worst_detour_removal_v2", penalties)


# --- New destroy operators merged from alns.py ---
# All three are self-contained: every helper they use is an instance method,
# so no free functions need to precede them.


class ShawRelatedRemoval(DestroyOperator):
    """Shaw / related removal: removes a cluster of mutually similar customers.

    Relatedness combines three normalized terms: spatial distance, time-window
    proximity and skill-set dissimilarity (1 - Jaccard). Removing related
    customers creates real recombination opportunities for the repair step,
    because similar customers can be exchanged between routes/vehicles.
    p_determinism > 1 biases the selection toward the most related customer
    (classic Shaw randomization: index = floor(rand()^p * len(list))).

    Richer variant of RelatedRemoval (which only uses spatial distance): this
    one also weighs time windows and skill overlap.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        p_determinism: float,
        w_distance: float,
        w_time: float,
        w_skill: float,
        initial_weight: float,
        neighbor_limit: int = 100,
    ):
        super().__init__("shaw_related_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.p_determinism = p_determinism
        self.w_distance = w_distance
        self.w_time = w_time
        self.w_skill = w_skill
        self.neighbor_limit = neighbor_limit

        # Lazily computed normalizers (instance is fixed per run).
        self._max_distance: float | None = None
        self._time_horizon: float | None = None

    def _ensure_normalizers(self, inst) -> None:
        if self._max_distance is None:
            self._max_distance = max(
                (max(row) for row in inst.distance),
                default=1,
            ) or 1
            self._time_horizon = max(inst.due) or 1

    def _relatedness(self, inst, i: int, j: int) -> float:
        distance_term = inst.distance[i][j] / self._max_distance
        time_term = abs(inst.ready[i] - inst.ready[j]) / self._time_horizon

        skills_i = inst.required_skills[i]
        skills_j = inst.required_skills[j]
        union = skills_i | skills_j
        jaccard = (len(skills_i & skills_j) / len(union)) if union else 1.0
        skill_term = 1.0 - jaccard

        return (
            self.w_distance * distance_term
            + self.w_time * time_term
            + self.w_skill * skill_term
        )

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "shaw_related_removal", penalties)

        self._ensure_normalizers(inst)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        seed = rng.choice(served)
        removed = [seed]
        removed_set = {seed}
        served_set = set(served)

        while len(removed) < q:
            reference = rng.choice(removed)

            # Restrict to the precomputed nearest neighbors of the reference
            # instead of ranking the full served list: relatedness is
            # dominated by the distance term, so customers far away are
            # never interesting picks anyway. Turns each pick from
            # O(served * log(served)) into O(scan + limit * log(limit)).
            pool = []

            for other in inst.nearest_customers[reference]:
                if other in served_set and other not in removed_set:
                    pool.append(other)

                    if len(pool) >= self.neighbor_limit:
                        break

            if not pool:
                break

            pool.sort(key=lambda c: self._relatedness(inst, reference, c))

            index = int((rng.random() ** self.p_determinism) * len(pool))
            index = min(index, len(pool) - 1)

            chosen = pool[index]
            removed.append(chosen)
            removed_set.add(chosen)

        return build_destroy_result(inst, solution, removed, "shaw_related_removal", penalties)


class RouteRemoval(DestroyOperator):
    """Empties one or more complete courier tours, biased toward tours with
    the worst profit per unit of used shift time.

    Particularly valuable for the skill variant: a highly qualified courier
    (large skill set) whose tour is filled with low-skill customers is wasted
    potential. Emptying the whole tour lets the repair step refill it with
    the demanding customers nobody else can serve. Removal fractions of
    single customers can never reach this reallocation in one move.
    """

    def __init__(
        self,
        max_routes: int,
        selection_bias: float,
        initial_weight: float,
    ):
        super().__init__("route_removal", initial_weight)
        self.max_routes = max_routes
        self.selection_bias = selection_bias

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        non_empty = [v for v in inst.vehicles if solution.routes[v]]

        if not non_empty:
            return build_destroy_result(inst, solution, [], "route_removal", penalties)

        def profit_per_time(vehicle: int) -> float:
            route = solution.routes[vehicle]
            profit = sum(inst.profit[c] for c in route)
            used_time = (
                solution.route_cache[vehicle].end_time
                - inst.vehicle_start[vehicle]
            )
            return profit / max(1, used_time)

        # Worst tours first; biased random pick so it is not deterministic.
        pool = sorted(non_empty, key=profit_per_time)

        n_routes = rng.randint(1, min(self.max_routes, len(pool)))

        chosen = []

        for _ in range(n_routes):
            index = int((rng.random() ** self.selection_bias) * len(pool))
            index = min(index, len(pool) - 1)
            chosen.append(pool.pop(index))

        removed = [c for vehicle in chosen for c in solution.routes[vehicle]]

        return build_destroy_result(inst, solution, removed, "route_removal", penalties)


class TimeWindowSegmentRemoval(DestroyOperator):
    """Removes customers whose service start falls into a random time
    interval, across all routes.

    Time windows tend to create temporal congestion: many customers compete
    for the same part of the day. Emptying one time slice everywhere lets
    the repair step re-decide which customers deserve that contested slice.
    Uses the cached service_start values, so it is O(#served).
    """

    def __init__(
        self,
        window_fraction: float,
        max_remove: int,
        initial_weight: float,
    ):
        super().__init__("time_segment_removal", initial_weight)
        self.window_fraction = window_fraction
        self.max_remove = max_remove

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        entries = []

        for vehicle, route in enumerate(solution.routes):
            cache = solution.route_cache[vehicle]

            for position, customer in enumerate(route):
                entries.append((customer, cache.service_start[position]))

        if not entries:
            return build_destroy_result(inst, solution, [], "time_segment_removal", penalties)

        horizon = max(inst.due) or 1
        width = max(1, int(self.window_fraction * horizon))

        _, anchor_time = entries[rng.randrange(len(entries))]
        window_start = anchor_time - width // 2
        window_end = window_start + width

        removed = [
            customer
            for customer, service_start in entries
            if window_start <= service_start <= window_end
        ]

        if len(removed) > self.max_remove:
            removed = rng.sample(removed, self.max_remove)

        return build_destroy_result(inst, solution, removed, "time_segment_removal", penalties)


# Segment-cache insertion evaluation


def insertion_penalty_from_cache(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    position: int,
) -> tuple[float, int]:

    route = solution.routes[vehicle]
    cache = solution.route_cache[vehicle]
    distance = inst.distance
    vehicle_skills = inst.vehicle_skills[vehicle]
    m = len(route)

    if position == 0:
        previous = 0
        time_now = inst.vehicle_start[vehicle]
        penalty = 0.0
    else:
        previous = route[position - 1]
        time_now = cache.departure[position - 1]
        penalty = cache.prefix_penalty[position]

    next_node = 0 if position == m else route[position]
    travel_delta = (
        distance[previous][customer]
        + distance[customer][next_node]
        - distance[previous][next_node]
    )

    if not inst.required_skills[customer].issubset(vehicle_skills):
        penalty += penalties.skill_penalty

    arrival = time_now + distance[previous][customer]

    if arrival > inst.due[customer]:
        penalty += penalties.time_window_penalty * (arrival - inst.due[customer])

    time_now = max(arrival, inst.ready[customer]) + inst.service[customer]
    previous = customer

    departures = cache.departure
    prefix_penalty = cache.prefix_penalty

    for j in range(position, m):
        if j > position:
            # From here on the predecessor node equals the cached one, so once
            # the timeline re-synchronizes the cached suffix applies verbatim.
            cached_departure = departures[j - 1]

            if time_now == cached_departure:
                return penalty + (cache.penalty - prefix_penalty[j]), travel_delta

            if time_now < cached_departure and cache.penalty == prefix_penalty[j]:
                # Earlier timeline + penalty-free cached suffix: arrivals only
                # get earlier, so the suffix stays penalty-free.
                return penalty, travel_delta

        suffix_customer = route[j]

        if not inst.required_skills[suffix_customer].issubset(vehicle_skills):
            penalty += penalties.skill_penalty

        arrival = time_now + distance[previous][suffix_customer]

        if arrival > inst.due[suffix_customer]:
            penalty += penalties.time_window_penalty * (arrival - inst.due[suffix_customer])

        time_now = max(arrival, inst.ready[suffix_customer]) + inst.service[suffix_customer]
        previous = suffix_customer

    end_time = time_now + distance[previous][0]

    if end_time > inst.vehicle_end[vehicle]:
        penalty += penalties.shift_penalty * (end_time - inst.vehicle_end[vehicle])

    return penalty, travel_delta


def insertion_move_for_position(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    position: int,
) -> InsertionMove:
    new_route_penalty, travel_delta = insertion_penalty_from_cache(
        inst=inst,
        solution=solution,
        penalties=penalties,
        customer=customer,
        vehicle=vehicle,
        position=position,
    )

    old_route_penalty = solution.route_cache[vehicle].penalty
    additional_penalty = new_route_penalty - old_route_penalty

    delta_score = inst.profit[customer] - additional_penalty

    return InsertionMove(
        customer=customer,
        vehicle=vehicle,
        position=position,
        delta_score=delta_score,
        travel_delta=travel_delta,
        new_route_penalty=new_route_penalty,
    )


def insertion_sort_key(move: InsertionMove) -> tuple[float, int, float]:
    return (
        -move.delta_score,
        move.travel_delta,
        move.new_route_penalty,
    )


def add_top_move(top_moves: list[InsertionMove], move: InsertionMove, k: int) -> None:

    top_moves.append(move)
    top_moves.sort(key=insertion_sort_key)

    if len(top_moves) > k:
        top_moves.pop()

def build_repair_candidates(
    inst,
    solution: Solution,
    removed_customers: list[int],
    extra_unserved_limit: int,
    rng: random.Random | None = None,
) -> list[int]:
    candidates = []
    seen = set()
    for customer in removed_customers:
        if customer in solution.unserved:
            candidates.append(customer)
            seen.add(customer)

    extra = [
        c for c in solution.unserved
        if c not in seen and inst.solo_feasible_vehicles[c]
    ]
    if extra_unserved_limit > 0 and extra:
        # Half exploitation (best static profit density), half exploration
        # (uniform sample) -- a purely static top list would offer the same
        # extra candidates in every iteration and starve the rest.
        top_limit = extra_unserved_limit // 2 if rng is not None else extra_unserved_limit
        extra_top = heapq.nsmallest(
            top_limit,
            extra,
            key=lambda c: (
                -inst.profit_density_lb[c],
                -inst.profit[c],
                len(inst.solo_feasible_vehicles[c]),
            ),
        )
        candidates.extend(extra_top)

        if rng is not None:
            chosen = set(extra_top)
            rest = [c for c in extra if c not in chosen]
            sample_size = min(extra_unserved_limit - len(extra_top), len(rest))
            if sample_size > 0:
                candidates.extend(rng.sample(rest, sample_size))
    return candidates


def vehicle_top_moves(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    k: int,
    noise_amp: float = 0.0,
    rng: random.Random | None = None,
) -> list[InsertionMove]:
    """Top-k insertion moves for `customer` restricted to a single vehicle.

    Tight inlined variant of insertion_move_for_position: one pass over all
    positions with the early-exit suffix evaluation, manual top-k tracking
    (k <= 2 in practice) and a pruning bound that stops scanning positions
    once the unavoidable time-window violation alone wipes out the profit
    (departures are non-decreasing along the route).
    """
    route = solution.routes[vehicle]
    cache = solution.route_cache[vehicle]
    distance = inst.distance
    dist_customer = distance[customer]
    departures = cache.departure
    prefix_penalty = cache.prefix_penalty
    old_penalty = cache.penalty

    ready = inst.ready
    due = inst.due
    service = inst.service
    required = inst.required_skills

    ready_c = ready[customer]
    due_c = due[customer]
    service_c = service[customer]
    profit_c = inst.profit[customer]

    tw_pen = penalties.time_window_penalty
    shift_pen = penalties.shift_penalty
    skill_pen = penalties.skill_penalty

    vehicle_skills = inst.vehicle_skills[vehicle]
    v_start = inst.vehicle_start[vehicle]
    v_end = inst.vehicle_end[vehicle]
    m = len(route)

    base_penalty = skill_pen if not required[customer].issubset(vehicle_skills) else 0.0

    if cache.penalty == 0.0 and base_penalty == 0.0:
        # Fast path for feasible routes (the common case): O(1) feasibility
        # per position via the cached forward time slack; only feasible
        # insertions are generated (delta_score = profit, penalty stays 0).
        service_starts = cache.service_start
        slack = cache.slack
        best: list[tuple[int, int]] = []  # (travel_delta, position)

        for position in range(m + 1):
            if position == 0:
                previous = 0
                time_prev = v_start
            else:
                time_prev = departures[position - 1]

                if time_prev > due_c:
                    break  # departures are non-decreasing: no later fit

                previous = route[position - 1]

            d_prev_customer = distance[previous][customer]
            arrival = time_prev + d_prev_customer

            if arrival > due_c:
                continue

            depart_c = (arrival if arrival >= ready_c else ready_c) + service_c

            if position == m:
                next_node = 0

                if depart_c + dist_customer[0] > v_end:
                    continue
            else:
                next_node = route[position]
                arrival_next = depart_c + dist_customer[next_node]
                ready_n = ready[next_node]
                begin_next = arrival_next if arrival_next >= ready_n else ready_n
                delay = begin_next - service_starts[position]

                if delay > 0 and delay > slack[position]:
                    continue

            travel_delta = (
                d_prev_customer
                + dist_customer[next_node]
                - distance[previous][next_node]
            )

            if noise_amp > 0.0 and rng is not None:
                # Noised insertion (Ropke & Pisinger 2006): perturb the
                # ranking so the deterministic cheapest position is not
                # always chosen. Only the ranking is noised; the executed
                # move itself stays feasible.
                travel_delta += rng.uniform(-noise_amp, noise_amp)

            if len(best) < k:
                best.append((travel_delta, position))
                best.sort()
            elif travel_delta < best[-1][0]:
                best[-1] = (travel_delta, position)
                best.sort()

        return [
            InsertionMove(
                customer=customer,
                vehicle=vehicle,
                position=position,
                delta_score=profit_c,
                travel_delta=travel_delta,
                new_route_penalty=0.0,
            )
            for travel_delta, position in best
        ]

    # Positions after this departure time can only yield delta_score <= 0
    # (arrival - due_c alone costs more than the profit); such moves are
    # discarded by the repair operators anyway.
    prune_departure = due_c + profit_c / tw_pen

    results: list[tuple[tuple, int, float, int]] = []  # (sort_key, position, penalty, travel_delta)

    for position in range(m + 1):
        if position == 0:
            previous = 0
            time_now = v_start
            penalty = base_penalty
        else:
            time_now = departures[position - 1]
            if time_now > prune_departure:
                break
            previous = route[position - 1]
            penalty = prefix_penalty[position] + base_penalty

        next_node = 0 if position == m else route[position]
        d_prev_customer = distance[previous][customer]
        travel_delta = (
            d_prev_customer
            + dist_customer[next_node]
            - distance[previous][next_node]
        )

        arrival = time_now + d_prev_customer
        if arrival > due_c:
            penalty += tw_pen * (arrival - due_c)

        time_now = (arrival if arrival >= ready_c else ready_c) + service_c
        previous = customer

        j = position
        while j < m:
            if j > position:
                cached_departure = departures[j - 1]

                if time_now == cached_departure:
                    penalty += old_penalty - prefix_penalty[j]
                    break

                if time_now < cached_departure and old_penalty == prefix_penalty[j]:
                    break

            suffix_customer = route[j]

            if not required[suffix_customer].issubset(vehicle_skills):
                penalty += skill_pen

            arrival = time_now + distance[previous][suffix_customer]
            due_s = due[suffix_customer]

            if arrival > due_s:
                penalty += tw_pen * (arrival - due_s)

            ready_s = ready[suffix_customer]
            time_now = (arrival if arrival >= ready_s else ready_s) + service[suffix_customer]
            previous = suffix_customer
            j += 1
        else:
            end_time = time_now + distance[previous][0]

            if end_time > v_end:
                penalty += shift_pen * (end_time - v_end)

        delta_score = profit_c - (penalty - old_penalty)
        sort_key = (-delta_score, travel_delta, penalty)

        if len(results) < k:
            results.append((sort_key, position, penalty, travel_delta))
            results.sort()
        elif sort_key < results[-1][0]:
            results[-1] = (sort_key, position, penalty, travel_delta)
            results.sort()

    return [
        InsertionMove(
            customer=customer,
            vehicle=vehicle,
            position=position,
            delta_score=-sort_key[0],
            travel_delta=travel_delta,
            new_route_penalty=penalty,
        )
        for sort_key, position, penalty, travel_delta in results
    ]


class MoveCache:
    """Per-(customer, vehicle) top-k insertion-move cache.

    Inserting a customer only modifies one route, so cached moves on all other
    vehicles stay exact. After each insertion only the (candidate, vehicle)
    entries of the modified vehicle are recomputed (Ropke & Pisinger 2006).
    """

    __slots__ = ("inst", "solution", "penalties", "k", "candidates", "by_customer")

    def __init__(
        self,
        inst,
        solution: Solution,
        penalties: PenaltyParams,
        candidates: list[int],
        k: int,
        deadline: float | None = None,
    ):
        self.inst = inst
        self.solution = solution
        self.penalties = penalties
        self.k = k
        self.candidates: list[int] = []
        self.by_customer: dict[int, dict[int, list[InsertionMove]]] = {}

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 7) == 0
                and time.perf_counter() > deadline
            ):
                break

            self.by_customer[customer] = {
                vehicle: vehicle_top_moves(inst, solution, penalties, customer, vehicle, k)
                for vehicle in inst.solo_feasible_vehicles[customer]
            }
            self.candidates.append(customer)

    def top_moves(self, customer: int) -> list[InsertionMove]:
        best_move = None
        best_key = None
        second_move = None
        second_key = None
        want_two = self.k > 1

        for moves in self.by_customer[customer].values():
            for move in moves:
                key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

                if best_key is None or key < best_key:
                    second_move, second_key = best_move, best_key
                    best_move, best_key = move, key
                elif want_two and (second_key is None or key < second_key):
                    second_move, second_key = move, key

        if best_move is None:
            return []

        if not want_two or second_move is None:
            return [best_move]

        return [best_move, second_move]

    def remove_candidate(self, customer: int) -> None:
        self.candidates.remove(customer)
        del self.by_customer[customer]

    def invalidate_vehicle(self, vehicle: int) -> None:
        for customer in self.candidates:
            per_vehicle = self.by_customer[customer]

            if vehicle in per_vehicle:
                per_vehicle[vehicle] = vehicle_top_moves(
                    self.inst, self.solution, self.penalties, customer, vehicle, self.k
                )


###########################################

# Repair operators


class GreedyBestInsertionRepair(RepairOperator):
    # Repair operators declare which destroy-result tags they need to run;
    # "customer_pool" means "a list of customers available for insertion".
    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("greedy_best_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        move_cache = MoveCache(
            inst=inst,
            solution=solution,
            penalties=penalties,
            candidates=candidates,
            k=1,
            deadline=deadline,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            best_move = None

            for customer in move_cache.candidates:
                moves = move_cache.top_moves(customer)

                if not moves:
                    continue

                move = moves[0]

                if move.delta_score <= self.min_delta_score:
                    continue

                if best_move is None:
                    best_move = move
                elif insertion_sort_key(move) < insertion_sort_key(best_move):
                    best_move = move

            if best_move is None:
                break

            apply_insertion(solution, inst, penalties, best_move)
            move_cache.remove_candidate(best_move.customer)
            move_cache.invalidate_vehicle(best_move.vehicle)
            inserted += 1

        return solution


class Regret2InsertionRepair(RepairOperator):
    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("regret2_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        move_cache = MoveCache(
            inst=inst,
            solution=solution,
            penalties=penalties,
            candidates=candidates,
            k=2,
            deadline=deadline,
        )

        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            selected_move = None
            selected_regret = None

            for customer in move_cache.candidates:
                moves = move_cache.top_moves(customer)

                if not moves:
                    continue

                best = moves[0]

                if best.delta_score <= self.min_delta_score:
                    continue

                if len(moves) == 1:
                    regret = 1_000_000.0 + best.delta_score
                else:
                    regret = best.delta_score - moves[1].delta_score

                if selected_move is None:
                    selected_move = best
                    selected_regret = regret
                elif regret > selected_regret:
                    selected_move = best
                    selected_regret = regret
                elif regret == selected_regret and insertion_sort_key(best) < insertion_sort_key(selected_move):
                    selected_move = best
                    selected_regret = regret

            if selected_move is None:
                break

            apply_insertion(solution, inst, penalties, selected_move)
            move_cache.remove_candidate(selected_move.customer)
            move_cache.invalidate_vehicle(selected_move.vehicle)
            inserted += 1

        return solution



class SequentialCheapestInsertionRepair(RepairOperator):
    """Single-pass repair (Kovacs et al. 2012 style): candidates are visited
    once in a fixed order (by profit, or shuffled) and each is immediately
    inserted at its cheapest feasible position. Roughly an order of magnitude
    cheaper per call than the best-first repairs, trading insertion-order
    optimality for iteration throughput -- the adaptive weights decide when
    that trade is worth it.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        order: str,
        initial_weight: float,
        noise_amp: float = 0.0,
    ):
        name = f"sequential_insertion_{order}"
        if noise_amp > 0.0:
            name += "_noise"
        super().__init__(name, initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.order = order
        self.noise_amp = noise_amp

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        if self.order == "profit":
            candidates.sort(
                key=lambda c: (-inst.profit[c], inst.due[c] - inst.ready[c])
            )
        else:
            rng.shuffle(candidates)

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 15) == 0
                and time.perf_counter() > deadline
            ):
                break

            best_move = None
            best_key = None

            for vehicle in inst.solo_feasible_vehicles[customer]:
                moves = vehicle_top_moves(
                    inst, solution, penalties, customer, vehicle, 1,
                    noise_amp=self.noise_amp, rng=rng,
                )

                if not moves:
                    continue

                move = moves[0]

                if move.delta_score <= 0.0:
                    continue

                key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

                if best_key is None or key < best_key:
                    best_move, best_key = move, key

            if best_move is not None:
                apply_insertion(solution, inst, penalties, best_move)

        return solution



# --- Scanner infrastructure for the repair operators merged from alns.py ---
# NoisyGreedyInsertionRepair and ScarceSkillFirstRepair below both rely on the
# O(1)-slack insertion move, best_insertion_on_vehicle and the InsertionScanner
# defined here, so this block is placed directly in front of those operators.
#
# The O(1) move is renamed slack_insertion_move_for_position to avoid clashing
# with the penalty-cache insertion_move_for_position used by vehicle_top_moves /
# MoveCache above. It uses the forward time slack, which build_route_cache only
# fills for feasible routes -- during repair the partial solution is feasible,
# so slack is always populated when these operators run.


def slack_insertion_move_for_position(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
    position: int,
) -> InsertionMove | None:
    """O(1) insertion evaluation via cached departures + forward time slack.

    Returns None when the insertion would violate a time window or the shift;
    the search stays strictly inside the feasible region (visits are optional,
    so a feasible completion always exists), hence delta_score is simply the
    profit and penalties are never incurred. Skill compatibility is NOT checked
    here: callers iterate inst.solo_feasible_vehicles[customer] only.
    """
    route = solution.routes[vehicle]
    cache = solution.route_cache[vehicle]
    distance = inst.distance

    if position == 0:
        previous = 0
        prev_departure = inst.vehicle_start[vehicle]
    else:
        previous = route[position - 1]
        prev_departure = cache.departure[position - 1]

    arrival = prev_departure + distance[previous][customer]

    if arrival > inst.due[customer]:
        return None

    departure = max(arrival, inst.ready[customer]) + inst.service[customer]

    if position == len(route):
        # Customer becomes the last stop before returning to the depot.
        if departure + distance[customer][0] > inst.vehicle_end[vehicle]:
            return None
        next_node = 0
    else:
        next_node = route[position]
        next_arrival = departure + distance[customer][next_node]

        if next_arrival > inst.due[next_node]:
            return None

        new_service_start = max(next_arrival, inst.ready[next_node])
        delay = new_service_start - cache.service_start[position]

        if delay > cache.slack[position]:
            return None

    travel_delta = (
        distance[previous][customer]
        + distance[customer][next_node]
        - distance[previous][next_node]
    )

    return InsertionMove(
        customer=customer,
        vehicle=vehicle,
        position=position,
        delta_score=float(inst.profit[customer]),
        travel_delta=travel_delta,
        new_route_penalty=0.0,
    )


def best_insertion_on_vehicle(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    customer: int,
    vehicle: int,
) -> InsertionMove | None:
    """Best feasible insertion of `customer` on one specific vehicle."""
    route = solution.routes[vehicle]
    best = None

    for position in range(len(route) + 1):
        move = slack_insertion_move_for_position(
            inst=inst,
            solution=solution,
            penalties=penalties,
            customer=customer,
            vehicle=vehicle,
            position=position,
        )

        if move is None:
            continue

        if best is None or insertion_sort_key(move) < insertion_sort_key(best):
            best = move

    return best


class InsertionScanner:
    """Incremental best-insertion bookkeeping for one repair phase.

    Maintains, for every candidate customer, the best feasible insertion on
    each compatible courier. Two invariants make this fast and correct:

    1. Inserting into vehicle v leaves every other route untouched, so after
       an insertion only the v-column needs re-scanning.
    2. Within one repair phase only insertions happen; routes only get
       tighter, so a (customer, vehicle) pair with no feasible insertion can
       never become feasible again. None entries are dropped permanently and
       are never re-evaluated.

    Combined with the O(1) slack feasibility check this reduces the repair
    loop from  O(insertions * candidates * vehicles * positions * suffix)
    to         O(candidates * vehicles * positions
                 + insertions * candidates * positions_on_changed_route).
    """

    __slots__ = ("inst", "solution", "penalties", "moves")

    def __init__(self, inst, solution: Solution, penalties: PenaltyParams,
                 candidates: list[int], deadline: float | None = None):
        self.inst = inst
        self.solution = solution
        self.penalties = penalties
        self.moves: dict[int, dict[int, InsertionMove]] = {}

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 7) == 0
                and time.perf_counter() > deadline
            ):
                break

            per_vehicle: dict[int, InsertionMove] = {}

            for vehicle in inst.solo_feasible_vehicles[customer]:
                move = best_insertion_on_vehicle(
                    inst, solution, penalties, customer, vehicle,
                )

                if move is not None:
                    per_vehicle[vehicle] = move

            if per_vehicle:
                self.moves[customer] = per_vehicle

    def customers(self) -> list[int]:
        """Candidates that still have at least one feasible insertion."""
        return list(self.moves.keys())

    def best(self, customer: int) -> InsertionMove | None:
        per_vehicle = self.moves.get(customer)

        if not per_vehicle:
            return None

        return min(per_vehicle.values(), key=insertion_sort_key)

    def best_two(self, customer: int) -> list[InsertionMove]:
        """Best insertions on the two best DISTINCT couriers (regret over
        couriers -- the meaningful regret for skill-scarce customers)."""
        per_vehicle = self.moves.get(customer)

        if not per_vehicle:
            return []

        ranked = sorted(per_vehicle.values(), key=insertion_sort_key)
        return ranked[:2]

    def notify_insertion(self, applied_move: InsertionMove) -> None:
        """Call after apply_insertion: re-scan only the changed vehicle."""
        self.moves.pop(applied_move.customer, None)
        vehicle = applied_move.vehicle
        exhausted = []

        for customer, per_vehicle in self.moves.items():
            if vehicle not in per_vehicle:
                continue

            move = best_insertion_on_vehicle(
                self.inst, self.solution, self.penalties, customer, vehicle,
            )

            if move is None:
                del per_vehicle[vehicle]

                if not per_vehicle:
                    exhausted.append(customer)
            else:
                per_vehicle[vehicle] = move

        for customer in exhausted:
            del self.moves[customer]


class NoisyGreedyInsertionRepair(RepairOperator):
    """Greedy best insertion with multiplicative noise on the comparison.

    The plain greedy repair is deterministic given a partial solution, so it
    tends to rebuild the same solution after similar destroys. Perturbing the
    delta_score by a random factor in [1 - noise, 1 + noise] during the
    comparison (the *applied* move is still evaluated exactly) diversifies
    the reconstruction without giving up greedy quality.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        noise: float,
        initial_weight: float,
    ):
        super().__init__("noisy_greedy_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score
        self.noise = noise

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        scanner = InsertionScanner(inst, solution, penalties, candidates, deadline)
        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            best_move = None
            best_perturbed = None

            for customer in scanner.customers():
                move = scanner.best(customer)

                if move is None:
                    continue

                if move.delta_score <= self.min_delta_score:
                    continue

                noise_factor = 1.0 + self.noise * (2.0 * rng.random() - 1.0)
                perturbed = (
                    move.delta_score / (1.0 + max(0, move.travel_delta))
                ) * noise_factor

                if best_move is None or perturbed > best_perturbed:
                    best_move = move
                    best_perturbed = perturbed

            if best_move is None:
                break

            apply_insertion(solution, inst, penalties, best_move)
            scanner.notify_insertion(best_move)
            inserted += 1

        return solution


class ScarceSkillFirstRepair(RepairOperator):
    """Single-pass insertion ordered by courier scarcity.

    Customers with few skill/time-compatible couriers must be placed first:
    a customer with only one compatible courier loses their slot forever if
    a flexible customer takes it. Sorting by
    (#compatible couriers, -profit, -profit density) is a cheap, targeted
    substitute for computing regret across couriers, and it is O(candidates)
    insertions instead of the greedy loop's O(candidates^2) evaluations,
    making it the fastest repair in the portfolio.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("scarce_skill_first_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        scanner = InsertionScanner(inst, solution, penalties, candidates, deadline)

        candidates.sort(
            key=lambda c: (
                len(inst.solo_feasible_vehicles[c]),
                -inst.profit[c],
                -inst.profit_density_lb[c],
            )
        )

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 15) == 0
                and time.perf_counter() > deadline
            ):
                break

            move = scanner.best(customer)

            if move is None:
                continue

            if move.delta_score <= self.min_delta_score:
                continue

            apply_insertion(solution, inst, penalties, move)
            scanner.notify_insertion(move)

        return solution


# Local-search polish (Or-opt) and capacity refill


def polish_route(inst, vehicle: int, route: list[int], deadline: float | None = None):
    """Or-opt within one route: relocate segments of length 1..3 to a position
    that strictly reduces travel time, keeping the route feasible. First
    improvement, repeated until no move improves. Profit is unchanged; the
    point is to free shift capacity for additional insertions.

    Returns (route, improved_flag)."""
    distance = inst.distance

    if len(route) < 3:
        return route, False

    improved_any = False
    improved = True

    while improved:
        improved = False

        if deadline is not None and time.perf_counter() > deadline:
            break

        m = len(route)

        for seg_len in (1, 2, 3):
            if improved:
                break

            for i in range(m - seg_len + 1):
                a = route[i - 1] if i > 0 else 0
                b = route[i]
                c = route[i + seg_len - 1]
                d = route[i + seg_len] if i + seg_len < m else 0

                remove_gain = distance[a][b] + distance[c][d] - distance[a][d]

                if remove_gain <= 0:
                    continue

                rest = route[:i] + route[i + seg_len:]
                segment = route[i:i + seg_len]
                n_rest = len(rest)

                for j in range(n_rest + 1):
                    if j == i:
                        continue  # original position

                    p = rest[j - 1] if j > 0 else 0
                    q = rest[j] if j < n_rest else 0
                    add_cost = distance[p][b] + distance[c][q] - distance[p][q]

                    if add_cost >= remove_gain:
                        continue

                    candidate = rest[:j] + segment + rest[j:]

                    if inst.route_is_feasible(vehicle, candidate):
                        route = candidate
                        improved = True
                        improved_any = True
                        break

                if improved:
                    break

    return route, improved_any


def fill_solution(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    rng: random.Random,
    limit: int,
    deadline: float | None = None,
) -> None:
    """Sequential pass inserting unserved candidates into free capacity
    (used after polishing, which shortens routes but does not add profit)."""
    candidates = build_repair_candidates(
        inst=inst,
        solution=solution,
        removed_customers=[],
        extra_unserved_limit=limit,
        rng=rng,
    )
    candidates.sort(key=lambda c: (-inst.profit[c], inst.due[c] - inst.ready[c]))

    for index, customer in enumerate(candidates):
        if (
            deadline is not None
            and (index & 15) == 0
            and time.perf_counter() > deadline
        ):
            break

        best_move = None
        best_key = None

        for vehicle in inst.solo_feasible_vehicles[customer]:
            moves = vehicle_top_moves(inst, solution, penalties, customer, vehicle, 1)

            if not moves:
                continue

            move = moves[0]

            if move.delta_score <= 0.0:
                continue

            key = (-move.delta_score, move.travel_delta, move.new_route_penalty)

            if best_key is None or key < best_key:
                best_move, best_key = move, key

        if best_move is not None:
            apply_insertion(solution, inst, penalties, best_move)


def polish_and_fill(
    inst,
    solution: Solution,
    penalties: PenaltyParams,
    rng: random.Random,
    deadline: float | None = None,
    fill_limit: int = 100,
) -> Solution:
    """Or-opt every route of a (feasible) solution, then refill freed
    capacity with unserved customers. Returns a new solution whose profit is
    >= the input's."""
    polished = copy_solution_with_state(solution)
    changed = False

    for vehicle in range(inst.num_vehicles):
        if deadline is not None and time.perf_counter() > deadline:
            break

        new_route, improved = polish_route(
            inst, vehicle, polished.routes[vehicle], deadline
        )

        if improved:
            polished.routes[vehicle] = new_route
            changed = True

    if changed:
        initialize_solution_state(inst, polished, penalties)

    fill_solution(inst, polished, penalties, rng, fill_limit, deadline)
    return polished



# Operator selection and weight update


def select_operator(operators: list[ALNSOperator], rng: random.Random) -> ALNSOperator:
    total_weight = sum(op.weight for op in operators)
    draw = rng.random() * total_weight
    cumulative = 0.0

    for op in operators:
        cumulative += op.weight

        if cumulative >= draw:
            return op

    return operators[-1]


def update_operator_weights(
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
    params: ALNSParams,
) -> None:
    for op in destroy_operators:
        op.update_weight(params)

    for op in repair_operators:
        op.update_weight(params)


def operator_summary(operators: list[ALNSOperator]) -> list[dict]:
    rows = []

    for op in operators:
        avg_time = op.total_time / op.total_uses if op.total_uses else 0.0

        rows.append(
            {
                "name": op.name,
                "weight": op.weight,
                "uses": op.total_uses,
                "avg_time": avg_time,
                "total_time": op.total_time,
            }
        )

    return rows



# Simulated Annealing

def accept_sa(
    current_score: float,
    candidate_score: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    if candidate_score >= current_score:
        return True

    probability = math.exp((candidate_score - current_score) / temperature)

    return rng.random() < probability
# Result


@dataclass(slots=True)
class ALNSResult:
    best_solution: Solution
    best_evaluation: SolutionEvaluation

    current_solution: Solution
    current_evaluation: SolutionEvaluation

    iterations: int
    runtime_seconds: float
    restarts: int

    destroy_summary: list[dict]
    repair_summary: list[dict]




##################################

# Main ALNS


def run_alns(
    runtime: float,
    inst,
    start_solution: Solution,
    sa: SAParams,
    params: ALNSParams,
    penalties: PenaltyParams,
    destroy_operators: list[DestroyOperator],
    repair_operators: list[RepairOperator],
) -> ALNSResult:
    rng = random.Random(params.random_seed)

    current_solution = start_solution.copy()
    current_eval = initialize_solution_state(inst, current_solution, penalties)


    best_solution = copy_solution_with_state(current_solution)
    best_eval = current_eval


    # Time-based cooling: temperature decays exponentially from the current
    # curve start T0 to min_temperature over the remaining wall-clock time.
    # This makes the schedule independent of the iteration throughput, which
    # varies by two orders of magnitude between the smallest and largest
    # instances.
    temperature = sa.initial_temperature
    curve_t0 = sa.initial_temperature
    curve_start = time.perf_counter()


    iterations = 0
    restarts = 0
    no_accept_counter = 0
    last_polish = 0.0
    start_time = time.perf_counter()
    deadline = start_time + runtime

    while time.perf_counter() < deadline:
        iterations += 1

        destroy_op = select_operator(destroy_operators, rng)

        t0 = time.perf_counter()
        destroy_result = destroy_op.apply(
            inst=inst,
            solution=current_solution,
            rng=rng,
            penalties=penalties,
        )
        destroy_time = time.perf_counter() - t0

        compatible_repairs = [
            repair_op
            for repair_op in repair_operators
            if repair_op.compatible_with(destroy_result)
        ]

        repair_op = select_operator(compatible_repairs, rng)

        t0 = time.perf_counter()
        candidate_solution = repair_op.apply(
            inst=inst,
            destroy_result=destroy_result,
            rng=rng,
            penalties=penalties,
            deadline=deadline,
        )
        repair_time = time.perf_counter() - t0

        candidate_eval = evaluation_from_solution(inst, candidate_solution)

        accepted = accept_sa(
            current_score=current_eval.score,
            candidate_score=candidate_eval.score,
            temperature=temperature,
            rng=rng,
        )

        new_global_best = (
            candidate_eval.feasible
            and candidate_eval.profit > best_eval.profit
        )

        if new_global_best:
            operator_score = params.score_global_best
        elif accepted and candidate_eval.score > current_eval.score:
            operator_score = params.score_better_current
        elif accepted:
            operator_score = params.score_accepted
        else:
            operator_score = params.score_rejected

        if new_global_best:
            best_solution = copy_solution_with_state(candidate_solution)
            best_eval = candidate_eval

            now = time.perf_counter()
            if (
                params.polish_interval_seconds > 0
                and now - last_polish >= params.polish_interval_seconds
            ):
                last_polish = now
                polished = polish_and_fill(
                    inst, best_solution, penalties, rng, deadline=deadline
                )
                polished_eval = evaluation_from_solution(inst, polished)

                if (
                    polished_eval.feasible
                    and polished_eval.profit >= best_eval.profit
                ):
                    best_solution = polished
                    best_eval = polished_eval
                    # continue the search from the compressed solution
                    candidate_solution = copy_solution_with_state(polished)
                    candidate_eval = polished_eval


        if accepted:
            current_solution = candidate_solution
            current_eval = candidate_eval
            no_accept_counter = 0
        else:
            no_accept_counter += 1

        destroy_op.observe(operator_score, destroy_time)
        repair_op.observe(operator_score, repair_time)

        if iterations % params.segment_length == 0:
            update_operator_weights(
                destroy_operators=destroy_operators,
                repair_operators=repair_operators,
                params=params,
            )

            if params.verbose:
                print(
                    f"iter={iterations:>7} | "
                    f"best={best_eval.profit:>8} | "
                    f"current_score={current_eval.score:>10.2f} | "
                    f"current_feasible={current_eval.feasible} | "
                    f"T={temperature:>8.4f}"
                )

        now = time.perf_counter()
        span = max(1e-9, deadline - curve_start)
        frac = min(1.0, (now - curve_start) / span)
        temperature = max(
            sa.min_temperature,
            curve_t0 * (sa.min_temperature / curve_t0) ** frac,
        )

        if no_accept_counter >= params.no_accept_limit:
            current_solution = random_start_solution(inst, rng, penalties, deadline=deadline)
            current_eval = evaluation_from_solution(inst, current_solution)

            curve_t0 = max(temperature, sa.initial_temperature * sa.reheat_factor)
            curve_start = time.perf_counter()
            temperature = curve_t0

            no_accept_counter = 0
            restarts += 1

            if params.verbose:
                print(
                    f"random_restart={restarts} at iter={iterations} | "
                    f"new current profit={current_eval.profit} | "
                    f"best profit={best_eval.profit} | "
                    f"T={temperature:.4f}"
                )

    update_operator_weights(
        destroy_operators=destroy_operators,
        repair_operators=repair_operators,
        params=params,
    )

    return ALNSResult(
        best_solution=best_solution,
        best_evaluation=best_eval,
        current_solution=current_solution,
        current_evaluation=current_eval,
        iterations=iterations,
        runtime_seconds=time.perf_counter() - start_time,
        restarts=restarts,
        destroy_summary=operator_summary(destroy_operators),
        repair_summary=operator_summary(repair_operators),
    )


###########################################

# Additional destroy/repair operators from the literature.
#
# Purely additive block: nothing above this line was changed. All classes
# below only use the existing helpers (build_destroy_result, number_to_remove,
# InsertionScanner, slack_insertion_move_for_position, ...).
#
# Sources:
# * Kovacs, Parragh, Doerner & Hartl (2012), J Sched 15:579-600 --
#   cluster removal (Sect. 3.1.4), relatedness on current service-begin
#   times (Eq. 18-19), regret-q repair (Sect. 3.2.3).
# * Hammami, Rekik & Coelho (2020), Comput. Oper. Res. 123:105034 --
#   sequence removal and largest-saving removal (Sect. 4.2.2), LRFI order
#   (Sect. 4.2.1) and random-feasible-position insertion (Sect. 4.2.3).


class SequenceRemoval(DestroyOperator):
    """Hammami et al. 2020 (Sect. 4.2.2, "sequence removal"): removes a
    contiguous sequence of customers from a randomly chosen route, creating
    one large contiguous time slot that the repair step can refill with a
    different (more profitable) sequence.

    TimeWindowSegmentRemoval empties a time slice ACROSS all routes; this is
    the intra-route counterpart. If the chosen route is shorter than the
    target size, further shuffled routes are drawn until q is reached.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        initial_weight: float,
    ):
        super().__init__("sequence_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "sequence_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        vehicles = [v for v in inst.vehicles if solution.routes[v]]
        rng.shuffle(vehicles)

        removed: list[int] = []

        for vehicle in vehicles:
            if len(removed) >= q:
                break

            route = solution.routes[vehicle]
            take = min(q - len(removed), len(route))
            start = rng.randrange(len(route) - take + 1)
            removed.extend(route[start:start + take])

        return build_destroy_result(inst, solution, removed, "sequence_removal", penalties)


def mst_two_clusters(inst, nodes: list[int]) -> tuple[list[int], list[int]]:
    """Splits `nodes` into the two clusters obtained by building a minimum
    spanning tree over them (Kruskal) and dropping its longest edge
    (Kovacs et al. 2012, cluster removal). Kruskal adds edges by increasing
    length, so the last MST edge is the longest one; re-unioning all but that
    edge leaves exactly the two target components."""
    n = len(nodes)
    distance = inst.distance

    edges = sorted(
        (distance[nodes[i]][nodes[j]], i, j)
        for i in range(n)
        for j in range(i + 1, n)
    )

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst_edges: list[tuple[int, int]] = []

    for _, i, j in edges:
        root_i, root_j = find(i), find(j)

        if root_i != root_j:
            parent[root_i] = root_j
            mst_edges.append((i, j))

            if len(mst_edges) == n - 1:
                break

    parent = list(range(n))

    for i, j in mst_edges[:-1]:
        root_i, root_j = find(i), find(j)

        if root_i != root_j:
            parent[root_i] = root_j

    first_root = find(0)
    cluster_a = [nodes[i] for i in range(n) if find(i) == first_root]
    cluster_b = [nodes[i] for i in range(n) if find(i) != first_root]
    return cluster_a, cluster_b


class ClusterRemoval(DestroyOperator):
    """Kovacs et al. 2012 (Sect. 3.1.4, after Ropke & Pisinger): picks a
    route with at least three customers, splits its customers into two
    spatial clusters via the minimum spanning tree (longest edge removed)
    and removes one cluster entirely. Then hops to the route of the closest
    still-scheduled customer and repeats until q customers are removed.

    Unlike RelatedRemoval/ShawRelatedRemoval (which pick similar customers
    ACROSS routes), this removes spatially coherent blocks PER route --
    aimed at breaking up geographically wrong courier assignments.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        initial_weight: float,
    ):
        super().__init__("cluster_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "cluster_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        removed: list[int] = []
        removed_set: set[int] = set()

        def remaining(vehicle: int) -> list[int]:
            return [c for c in solution.routes[vehicle] if c not in removed_set]

        eligible = [v for v in inst.vehicles if len(solution.routes[v]) >= 3]

        if not eligible:
            # No route is long enough to split: degrade to a random sample
            # (Kovacs stops here; an empty destroy would waste the iteration).
            removed = rng.sample(served, q)
            return build_destroy_result(inst, solution, removed, "cluster_removal", penalties)

        vehicle = rng.choice(eligible)

        while len(removed) < q:
            nodes = remaining(vehicle)

            if len(nodes) < 3:
                break

            cluster_a, cluster_b = mst_two_clusters(inst, nodes)
            chosen = cluster_a if rng.random() < 0.5 else cluster_b

            removed.extend(chosen)
            removed_set.update(chosen)

            if len(removed) >= q:
                break

            # Hop to the route of the closest still-scheduled customer of a
            # randomly drawn just-removed one; that route must be a different
            # one and still splittable. A few seeds are tried before giving up.
            next_vehicle = None

            for _ in range(8):
                seed = rng.choice(removed)

                for other in inst.nearest_customers[seed]:
                    if other in removed_set:
                        continue

                    other_vehicle = solution.customer_to_vehicle.get(other)

                    if other_vehicle is None:
                        continue  # not currently scheduled

                    # `other` is the closest still-scheduled customer.
                    if other_vehicle != vehicle and len(remaining(other_vehicle)) >= 3:
                        next_vehicle = other_vehicle

                    break

                if next_vehicle is not None:
                    break

            if next_vehicle is None:
                break

            vehicle = next_vehicle

        return build_destroy_result(inst, solution, removed, "cluster_removal", penalties)


class LargestSavingRemoval(DestroyOperator):
    """Hammami et al. 2020 (Sect. 4.2.2, "largest saving in traveling
    time"): removes the customers whose removal frees the most travel time,
    deliberately ignoring profit.

    The profit-blind counterpart of WorstDetourRemoval(V2): those rank by
    profit/detour and therefore spare expensive-but-profitable customers,
    while this operator removes the biggest time consumers outright and
    lets the repair step decide whether the freed shift time buys more
    total profit elsewhere. Biased sampling rand()^bias from the largest
    savings keeps repeated calls from removing identical sets.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        selection_bias: float,
        initial_weight: float,
    ):
        super().__init__("largest_saving_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.selection_bias = selection_bias

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "largest_saving_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        distance = inst.distance
        scored = []

        for route in solution.routes:
            last = len(route) - 1

            for i, customer in enumerate(route):
                prev_node = route[i - 1] if i > 0 else 0
                next_node = route[i + 1] if i < last else 0
                saving = (
                    distance[prev_node][customer]
                    + distance[customer][next_node]
                    - distance[prev_node][next_node]
                )
                scored.append((-saving, customer))

        scored.sort()  # largest saving first
        pool = [customer for _, customer in scored]

        removed: list[int] = []

        while len(removed) < q and pool:
            index = int((rng.random() ** self.selection_bias) * len(pool))
            index = min(index, len(pool) - 1)
            removed.append(pool.pop(index))

        return build_destroy_result(inst, solution, removed, "largest_saving_removal", penalties)


class TemporalShawRemoval(DestroyOperator):
    """Shaw relatedness computed on the CURRENT solution (Kovacs et al.
    2012, Eq. 18-19): the time term compares the actual service-begin times
    of the current tours (cached service_start) instead of the static ready
    times used by ShawRelatedRemoval. The skill term is the normalized
    Hamming distance of the required skill sets (equivalent to the Jaccard
    dissimilarity, matching ShawRelatedRemoval) -- the solution-dependent
    time term is what distinguishes the two operators.

    Customers that are close, served at a similar time of day and need
    similar skills are the easiest to exchange between couriers; that
    similarity depends on the current schedule, which the static variant
    cannot see.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        p_determinism: float,
        w_distance: float,
        w_time: float,
        w_skill: float,
        initial_weight: float,
        neighbor_limit: int = 100,
    ):
        super().__init__("temporal_shaw_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.p_determinism = p_determinism
        self.w_distance = w_distance
        self.w_time = w_time
        self.w_skill = w_skill
        self.neighbor_limit = neighbor_limit

        # Lazily computed normalizers (instance is fixed per run).
        self._max_distance: float | None = None
        self._time_horizon: float | None = None

    def _ensure_normalizers(self, inst) -> None:
        if self._max_distance is None:
            self._max_distance = max(
                (max(row) for row in inst.distance),
                default=1,
            ) or 1
            self._time_horizon = max(inst.due) or 1

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "temporal_shaw_removal", penalties)

        self._ensure_normalizers(inst)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        service_start: dict[int, int] = {}

        for vehicle, route in enumerate(solution.routes):
            starts = solution.route_cache[vehicle].service_start

            for position, customer in enumerate(route):
                service_start[customer] = starts[position]

        def relatedness(i: int, j: int) -> float:
            distance_term = inst.distance[i][j] / self._max_distance
            time_term = abs(service_start[i] - service_start[j]) / self._time_horizon

            skills_i = inst.required_skills[i]
            skills_j = inst.required_skills[j]
            union = skills_i | skills_j
            skill_term = len(skills_i ^ skills_j) / len(union) if union else 0.0

            return (
                self.w_distance * distance_term
                + self.w_time * time_term
                + self.w_skill * skill_term
            )

        seed = rng.choice(served)
        removed = [seed]
        removed_set = {seed}
        served_set = set(served)

        while len(removed) < q:
            reference = rng.choice(removed)

            pool = []

            for other in inst.nearest_customers[reference]:
                if other in served_set and other not in removed_set:
                    pool.append(other)

                    if len(pool) >= self.neighbor_limit:
                        break

            if not pool:
                break

            pool.sort(key=lambda c: relatedness(reference, c))

            index = int((rng.random() ** self.p_determinism) * len(pool))
            index = min(index, len(pool) - 1)

            chosen = pool[index]
            removed.append(chosen)
            removed_set.add(chosen)

        return build_destroy_result(inst, solution, removed, "temporal_shaw_removal", penalties)


def scanner_best_k(scanner: InsertionScanner, customer: int, k: int) -> list[InsertionMove]:
    """Best insertions on the k best DISTINCT couriers (generalizes
    InsertionScanner.best_two to arbitrary k)."""
    per_vehicle = scanner.moves.get(customer)

    if not per_vehicle:
        return []

    ranked = sorted(per_vehicle.values(), key=insertion_sort_key)
    return ranked[:k]


class RegretKInsertionRepair(RepairOperator):
    """Regret-q insertion over distinct couriers (Kovacs et al. 2012,
    Sect. 3.2.3): insert next the customer that loses the most if it cannot
    take its best option.

    Profits do not depend on the courier, so the regret is measured on the
    travel-time cost of the best insertion per courier:
    regret(c) = sum_{i=2..k} (travel_i - travel_1). Customers with fewer
    than k compatible couriers get a large urgency bonus per missing
    alternative (the regret-m rationale) -- for the skill variant this
    naturally places skill-scarce customers first.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        max_insertions: int | None,
        min_delta_score: float,
        k: int,
        initial_weight: float,
    ):
        super().__init__(f"regret{k}_courier_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.max_insertions = max_insertions
        self.min_delta_score = min_delta_score
        self.k = k

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        scanner = InsertionScanner(inst, solution, penalties, candidates, deadline)
        inserted = 0

        while self.max_insertions is None or inserted < self.max_insertions:
            if deadline is not None and time.perf_counter() > deadline:
                break

            selected_move = None
            selected_key = None

            for customer in scanner.customers():
                moves = scanner_best_k(scanner, customer, self.k)

                if not moves:
                    continue

                best = moves[0]

                if best.delta_score <= self.min_delta_score:
                    continue

                regret = 1_000_000.0 * (self.k - len(moves))

                for move in moves[1:]:
                    regret += move.travel_delta - best.travel_delta

                key = (regret, best.delta_score, -best.travel_delta)

                if selected_key is None or key > selected_key:
                    selected_move = best
                    selected_key = key

            if selected_move is None:
                break

            apply_insertion(solution, inst, penalties, selected_move)
            scanner.notify_insertion(selected_move)
            inserted += 1

        return solution


class LastRemovedFirstInsertedRepair(RepairOperator):
    """Hammami et al. 2020 (Sect. 4.2.1, "last removed, first inserted"):
    reinserts the removed customers in reverse removal order, each at its
    best feasible position, then offers the extra unserved candidates (in
    build_repair_candidates order). Cheap ordering strategy the portfolio
    does not have yet: it gives the customers removed last a chance to grab
    better positions than the ones they just lost.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        min_delta_score: float,
        initial_weight: float,
    ):
        super().__init__("lrfi_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.min_delta_score = min_delta_score

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        removed_set = set(destroy_result.removed_customers)
        lrfi = [
            c for c in reversed(destroy_result.removed_customers)
            if c in solution.unserved
        ]
        extras = [c for c in candidates if c not in removed_set]
        order = lrfi + extras

        scanner = InsertionScanner(inst, solution, penalties, order, deadline)

        for index, customer in enumerate(order):
            if (
                deadline is not None
                and (index & 15) == 0
                and time.perf_counter() > deadline
            ):
                break

            move = scanner.best(customer)

            if move is None:
                continue

            if move.delta_score <= self.min_delta_score:
                continue

            apply_insertion(solution, inst, penalties, move)
            scanner.notify_insertion(move)

        return solution


class RandomPositionInsertionRepair(RepairOperator):
    """Hammami et al. 2020 (Sect. 4.2.3, "random available position
    insertion"): visits the candidates in random order and inserts each at
    a uniformly random feasible position instead of the best one.

    Pure diversification repair: every other repair in the portfolio lands
    near a local optimum of the insertion order; this one deliberately does
    not. Only slack-checked feasible positions are generated, so the result
    stays penalty-free like the other slack-based repairs.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        initial_weight: float,
    ):
        super().__init__("random_position_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        rng.shuffle(candidates)

        for index, customer in enumerate(candidates):
            if (
                deadline is not None
                and (index & 7) == 0
                and time.perf_counter() > deadline
            ):
                break

            feasible_moves = []

            for vehicle in inst.solo_feasible_vehicles[customer]:
                if solution.route_cache[vehicle].penalty != 0.0:
                    continue  # forward slack only exists for feasible routes

                for position in range(len(solution.routes[vehicle]) + 1):
                    move = slack_insertion_move_for_position(
                        inst=inst,
                        solution=solution,
                        penalties=penalties,
                        customer=customer,
                        vehicle=vehicle,
                        position=position,
                    )

                    if move is not None:
                        feasible_moves.append(move)

            if feasible_moves:
                apply_insertion(solution, inst, penalties, rng.choice(feasible_moves))

        return solution


class HistoryBasedRemoval(DestroyOperator):
    """History-based removal (Ropke & Pisinger 2006; Pisinger & Ropke 2007;
    survey: Mara et al. 2022): remembers, per customer, the smallest travel
    detour ever observed for its route position and removes the customers
    currently sitting furthest above their historical best. This taps the
    search history -- an information channel none of the geometry/time/skill
    based operators can see: a customer whose current position is much worse
    than one the search has already found is a promising relocation target.

    Purely additive caveat: operators have no per-iteration hook, so the
    history is only refreshed whenever THIS operator is applied -- an
    approximation of the every-iteration bookkeeping in the original paper.
    Like all stateful operators, a fresh instance per run resets the history.
    """

    def __init__(
        self,
        fraction: float,
        min_remove: int,
        max_remove: int,
        selection_bias: float,
        initial_weight: float,
    ):
        super().__init__("history_based_removal", initial_weight)
        self.fraction = fraction
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.selection_bias = selection_bias

        # customer -> smallest detour seen in any of this operator's calls
        self.best_detour: dict[int, int] = {}

    def apply(
        self,
        inst,
        solution: Solution,
        rng: random.Random,
        penalties: PenaltyParams,
    ) -> DestroyResult:
        served = served_customers(solution)

        if not served:
            return build_destroy_result(inst, solution, [], "history_based_removal", penalties)

        q = number_to_remove(
            n_served=len(served),
            fraction=self.fraction,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
            rng=rng,
        )

        distance = inst.distance
        best_detour = self.best_detour
        scored = []

        for route in solution.routes:
            last = len(route) - 1

            for i, customer in enumerate(route):
                prev_node = route[i - 1] if i > 0 else 0
                next_node = route[i + 1] if i < last else 0
                detour = (
                    distance[prev_node][customer]
                    + distance[customer][next_node]
                    - distance[prev_node][next_node]
                )

                known = best_detour.get(customer)

                if known is None or detour < known:
                    best_detour[customer] = detour
                    known = detour

                # Gap to the historical best position; 0 on first sighting.
                scored.append((known - detour, customer))

        scored.sort()  # largest gap first (gap stored negated)
        pool = [customer for _, customer in scored]

        removed: list[int] = []

        while len(removed) < q and pool:
            index = int((rng.random() ** self.selection_bias) * len(pool))
            index = min(index, len(pool) - 1)
            removed.append(pool.pop(index))

        return build_destroy_result(inst, solution, removed, "history_based_removal", penalties)


class ShawInsertionRepair(RepairOperator):
    """Shaw insertion (Coelho et al. 2012; survey: Mara et al. 2022): the
    next customer to insert is the one most similar to the LAST inserted
    customer, each placed at its best feasible position. Uses the same
    static relatedness terms as ShawRelatedRemoval (distance, ready-time
    difference, skill Jaccard dissimilarity), so related customers re-enter
    the solution as a group instead of scattered singletons -- the natural
    counterpart to the Shaw-style removals. p_determinism > 1 biases toward
    the most related candidate; the first customer is drawn at random.
    """

    requires = {"customer_pool"}

    def __init__(
        self,
        extra_unserved_limit: int,
        min_delta_score: float,
        p_determinism: float,
        w_distance: float,
        w_time: float,
        w_skill: float,
        initial_weight: float,
    ):
        super().__init__("shaw_insertion", initial_weight)
        self.extra_unserved_limit = extra_unserved_limit
        self.min_delta_score = min_delta_score
        self.p_determinism = p_determinism
        self.w_distance = w_distance
        self.w_time = w_time
        self.w_skill = w_skill

        # Lazily computed normalizers (instance is fixed per run).
        self._max_distance: float | None = None
        self._time_horizon: float | None = None

    def _ensure_normalizers(self, inst) -> None:
        if self._max_distance is None:
            self._max_distance = max(
                (max(row) for row in inst.distance),
                default=1,
            ) or 1
            self._time_horizon = max(inst.due) or 1

    def _relatedness(self, inst, i: int, j: int) -> float:
        distance_term = inst.distance[i][j] / self._max_distance
        time_term = abs(inst.ready[i] - inst.ready[j]) / self._time_horizon

        skills_i = inst.required_skills[i]
        skills_j = inst.required_skills[j]
        union = skills_i | skills_j
        jaccard = (len(skills_i & skills_j) / len(union)) if union else 1.0
        skill_term = 1.0 - jaccard

        return (
            self.w_distance * distance_term
            + self.w_time * time_term
            + self.w_skill * skill_term
        )

    def apply(
        self,
        inst,
        destroy_result: DestroyResult,
        rng: random.Random,
        penalties: PenaltyParams,
        deadline: float | None = None,
    ) -> Solution:
        solution = destroy_result.partial_solution
        self._ensure_normalizers(inst)

        candidates = build_repair_candidates(
            inst=inst,
            solution=solution,
            removed_customers=destroy_result.removed_customers,
            extra_unserved_limit=self.extra_unserved_limit,
            rng=rng,
        )

        scanner = InsertionScanner(inst, solution, penalties, candidates, deadline)

        pending = scanner.customers()
        reference: int | None = None

        while pending:
            if deadline is not None and time.perf_counter() > deadline:
                break

            if reference is None:
                index = rng.randrange(len(pending))
            else:
                pending.sort(key=lambda c: self._relatedness(inst, reference, c))
                index = int((rng.random() ** self.p_determinism) * len(pending))
                index = min(index, len(pending) - 1)

            customer = pending.pop(index)

            move = scanner.best(customer)

            # Infeasible/unprofitable candidates are skipped; the reference
            # stays the last actually inserted customer.
            if move is None or move.delta_score <= self.min_delta_score:
                continue

            apply_insertion(solution, inst, penalties, move)
            scanner.notify_insertion(move)
            reference = customer

        return solution
