from __future__ import annotations

from dataclasses import dataclass

from solution import Solution


@dataclass
class Insertion:
    customer: int
    vehicle: int
    position: int
    travel_delta: int


def find_best_insertion(inst, routes: list[list[int]], customer: int) -> Insertion | None:
    best = None

    for vehicle in inst.solo_feasible_vehicles[customer]:
        route = routes[vehicle]

        for position in range(len(route) + 1):
            previous = 0 if position == 0 else route[position - 1]
            next_node = 0 if position == len(route) else route[position]

            travel_delta = inst.insertion_travel_delta(
                previous,
                customer,
                next_node,
            )

            route.insert(position, customer)
            feasible = inst.route_is_feasible(vehicle, route)
            route.pop(position)

            if not feasible:
                continue

            if best is None or travel_delta < best.travel_delta:
                best = Insertion(
                    customer=customer,
                    vehicle=vehicle,
                    position=position,
                    travel_delta=travel_delta,
                )

    return best


def insert_customer(solution: Solution, inst, insertion: Insertion) -> None:

    customer = insertion.customer
    vehicle = insertion.vehicle
    position = insertion.position

    solution.routes[vehicle].insert(position, customer)
    solution.unserved.remove(customer)
    solution.customer_to_vehicle[customer] = vehicle
    solution.objective += inst.profit[customer]


def get_customer_order(inst, strategy: str) -> list[int]:

    serviceable = [
        c for c in inst.customers
        if inst.solo_feasible_vehicles[c]
    ]

    if strategy == "profit":
        return sorted(
            serviceable,
            key=lambda c: inst.profit[c],
            reverse=True,
        )

    if strategy == "density":
        return sorted(
            serviceable,
            key=lambda c: inst.profit_density_lb[c],
            reverse=True,
        )

    if strategy == "fewest_vehicles":
        return sorted(
            serviceable,
            key=lambda c: (
                len(inst.solo_feasible_vehicles[c]),
                -inst.profit[c],
            ),
        )

    if strategy == "hybrid":
        return sorted(
            serviceable,
            key=lambda c: (
                len(inst.solo_feasible_vehicles[c]),
                -inst.profit_density_lb[c],
                -inst.profit[c],
                inst.min_required_time[c],
            ),
        )

    raise ValueError(f"Unknown greedy strategy: {strategy}")


def greedy_build_with_order(inst, customer_order: list[int]) -> Solution:
    """
    Builds one greedy solution using a fixed customer order.
    """
    solution = Solution.empty(inst)

    for customer in customer_order:
        insertion = find_best_insertion(
            inst=inst,
            routes=solution.routes,
            customer=customer,
        )

        if insertion is not None:
            insert_customer(solution, inst, insertion)

    return solution


def greedy_initial_solution(
    inst,
    strategy: str = "multi_start",
    verbose: bool = False,
) -> Solution:

    if strategy != "multi_start":
        order = get_customer_order(inst, strategy)
        solution = greedy_build_with_order(inst, order)

        if verbose:
            print(f"Greedy strategy: {strategy}")
            print(f"Objective: {solution.objective}")
            print(f"Served customers: {inst.num_customers - len(solution.unserved)}/{inst.num_customers}")

        return solution

    strategies = [
        "hybrid",
        "profit",
        "density",
        "fewest_vehicles",
    ]

    best_solution = None
    best_strategy = None

    for current_strategy in strategies:
        order = get_customer_order(inst, current_strategy)
        solution = greedy_build_with_order(inst, order)

        if verbose:
            print(f"Greedy strategy: {current_strategy}")
            print(f"  Objective: {solution.objective}")
            print(f"  Served customers: {inst.num_customers - len(solution.unserved)}/{inst.num_customers}")

        if best_solution is None or solution.objective > best_solution.objective:
            best_solution = solution
            best_strategy = current_strategy

    if verbose:
        print()
        print(f"Selected greedy strategy: {best_strategy}")
        print(f"Best objective: {best_solution.objective}")
        print(f"Served customers: {inst.num_customers - len(best_solution.unserved)}/{inst.num_customers}")

    return best_solution