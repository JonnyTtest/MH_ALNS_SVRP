from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Solution:
    
    # Internal solution representation.

    routes: list[list[int]]
    unserved: set[int]
    objective: int
    customer_to_vehicle: dict[int, int]

    @classmethod
    def empty(cls, inst) -> "Solution":
        return cls(
            routes=[[] for _ in range(inst.num_vehicles)],
            unserved=set(inst.customers),
            objective=0,
            customer_to_vehicle={},
        )

    def copy(self) -> "Solution":
        return Solution(
            routes=deepcopy(self.routes),
            unserved=set(self.unserved),
            objective=self.objective,
            customer_to_vehicle=dict(self.customer_to_vehicle),
        )


    def print_for_checker(self) -> None:
        print("###RESULT: Feasible")
        print(f"###OBJECTIVE: {self.objective}")

        for b, route in enumerate(self.routes):
            route_with_depot = [0] + route + [0]
            route_string = " ".join(map(str, route_with_depot))
            print(f"###VEHICLE {b + 1}: {route_string}")

    def write_for_checker(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            print("###RESULT: Feasible", file=f)
            print(f"###OBJECTIVE: {self.objective}", file=f)

            for b, route in enumerate(self.routes):
                route_with_depot = [0] + route + [0]
                route_string = " ".join(map(str, route_with_depot))
                print(f"###VEHICLE {b + 1}: {route_string}", file=f)