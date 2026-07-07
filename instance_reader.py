from __future__ import annotations

from dataclasses import dataclass, field
import math
import re


@dataclass
class Instance:
    # Basic data
    name: str
    num_nodes: int
    num_customers: int
    num_vehicles: int
    num_skills: int

    # Node data, indexed by node id
    x: list[float]
    y: list[float]
    ready: list[int]
    due: list[int]
    service: list[int]
    profit: list[int]
    required_skills: list[set[int]]

    # Vehicle data, indexed internally by 0..num_vehicles-1
    vehicle_start: list[int]
    vehicle_end: list[int]
    vehicle_skills: list[set[int]]

    # Derived data
    customers: list[int] = field(init=False)
    vehicles: list[int] = field(init=False)
    distance: list[list[int]] = field(init=False)

    skill_feasible_vehicles: list[list[int]] = field(init=False)
    solo_feasible_vehicles: list[list[int]] = field(init=False)

    min_required_time: list[int] = field(init=False)
    depot_roundtrip_time: list[int] = field(init=False)
    profit_density_lb: list[float] = field(init=False)

    # nearest_customers[c] = all other customers sorted by increasing distance to c
    nearest_customers: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.customers = list(range(1, self.num_nodes))
        self.vehicles = list(range(self.num_vehicles))

        self._compute_distances()
        self._compute_skill_feasible_vehicles()
        self._compute_solo_feasible_vehicles()
        self._compute_customer_bounds()
        self._compute_nearest_customers()

    @classmethod
    def from_file(cls, path: str) -> "Instance":
        """Whitespace-agnostic parser: any two values may be separated by one
        or more blanks, tabs or newlines (as the project spec allows), so the
        file is parsed as a flat token stream instead of line by line.
        Underscores are normalized to spaces first, which also accepts the
        "NODE SECTION" / "NUM SKILLS" keyword spellings."""
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()

        tokens = raw.replace("_", " ").split()
        upper = [t.upper() for t in tokens]

        def find_marker(words: list[str], start: int = 0, end: int | None = None) -> int:
            if end is None:
                end = len(upper)
            span = len(words)
            for idx in range(start, end - span + 1):
                if upper[idx:idx + span] == words:
                    return idx
            return -1

        node_marker = find_marker(["NODE", "SECTION"])
        vehicle_marker = find_marker(["VEHICLE", "SECTION"])
        if node_marker < 0 or vehicle_marker < 0:
            raise ValueError("could not locate NODE/VEHICLE sections")

        def header_int(words: list[str]) -> int:
            idx = find_marker(words, 0, node_marker)
            if idx < 0:
                raise ValueError("missing header field: " + " ".join(words))
            for k in range(idx + len(words), node_marker):
                try:
                    return int(tokens[k])
                except ValueError:
                    continue
            raise ValueError("no integer after header field: " + " ".join(words))

        num_nodes = header_int(["DIMENSION"])
        num_customers = num_nodes - 1
        num_vehicles = header_int(["VEHICLES"])
        num_skills = header_int(["NUM", "SKILLS"])

        name_match = re.search(r"NAME\s*:\s*(\S+)", raw, flags=re.IGNORECASE)
        name = name_match.group(1) if name_match else "instance"

        x = [0.0] * num_nodes
        y = [0.0] * num_nodes
        ready = [0] * num_nodes
        due = [0] * num_nodes
        service = [0] * num_nodes
        profit = [0] * num_nodes
        required_skills = [set() for _ in range(num_nodes)]

        c = node_marker + 2
        for _ in range(num_nodes):
            node = int(tokens[c]); c += 1
            x[node] = float(tokens[c]); c += 1
            y[node] = float(tokens[c]); c += 1
            ready[node] = int(tokens[c]); c += 1
            due[node] = int(tokens[c]); c += 1
            service[node] = int(tokens[c]); c += 1
            profit[node] = int(tokens[c]); c += 1
            k = int(tokens[c]); c += 1
            required_skills[node] = set(int(tokens[c + s]) for s in range(k))
            c += k

        vehicle_start = [0] * num_vehicles
        vehicle_end = [0] * num_vehicles
        vehicle_skills = [set() for _ in range(num_vehicles)]

        c = vehicle_marker + 2
        for _ in range(num_vehicles):
            vehicle = int(tokens[c]) - 1; c += 1
            vehicle_start[vehicle] = int(tokens[c]); c += 1
            vehicle_end[vehicle] = int(tokens[c]); c += 1
            k = int(tokens[c]); c += 1
            vehicle_skills[vehicle] = set(int(tokens[c + s]) for s in range(k))
            c += k

        return cls(
            name=name,
            num_nodes=num_nodes,
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            num_skills=num_skills,
            x=x,
            y=y,
            ready=ready,
            due=due,
            service=service,
            profit=profit,
            required_skills=required_skills,
            vehicle_start=vehicle_start,
            vehicle_end=vehicle_end,
            vehicle_skills=vehicle_skills,
        )

    def d(self, i: int, j: int) -> int:
        return self.distance[i][j]

    def can_vehicle_serve_customer_by_skill(self, vehicle: int, customer: int) -> bool:
        return self.required_skills[customer].issubset(self.vehicle_skills[vehicle])

    def route_is_feasible(self, vehicle: int, route: list[int]) -> bool:
        time = self.vehicle_start[vehicle]
        previous = 0

        for customer in route:
            if not self.can_vehicle_serve_customer_by_skill(vehicle, customer):
                return False

            arrival = time + self.distance[previous][customer]

            if arrival > self.due[customer]:
                return False

            time = max(arrival, self.ready[customer]) + self.service[customer]
            previous = customer

        time += self.distance[previous][0]

        return time <= self.vehicle_end[vehicle]

    def insertion_travel_delta(self, previous: int, customer: int, next_node: int) -> int:
        return (
            self.distance[previous][customer]
            + self.distance[customer][next_node]
            - self.distance[previous][next_node]
        )

    def _compute_distances(self) -> None:
        n = self.num_nodes
        self.distance = [[0] * n for _ in range(n)]

        for i in range(n):
            xi = self.x[i]
            yi = self.y[i]

            for j in range(i + 1, n):
                dx = xi - self.x[j]
                dy = yi - self.y[j]
                dist = int(round(math.hypot(dx, dy)))

                self.distance[i][j] = dist
                self.distance[j][i] = dist

    def _compute_skill_feasible_vehicles(self) -> None:
        self.skill_feasible_vehicles = [[] for _ in range(self.num_nodes)]

        for customer in self.customers:
            required = self.required_skills[customer]

            for vehicle in self.vehicles:
                if required.issubset(self.vehicle_skills[vehicle]):
                    self.skill_feasible_vehicles[customer].append(vehicle)

    def _compute_solo_feasible_vehicles(self) -> None:
        self.solo_feasible_vehicles = [[] for _ in range(self.num_nodes)]

        for customer in self.customers:
            for vehicle in self.skill_feasible_vehicles[customer]:
                arrival = self.vehicle_start[vehicle] + self.distance[0][customer]

                if arrival > self.due[customer]:
                    continue

                finish_service = max(arrival, self.ready[customer]) + self.service[customer]
                return_time = finish_service + self.distance[customer][0]

                if return_time <= self.vehicle_end[vehicle]:
                    self.solo_feasible_vehicles[customer].append(vehicle)

    def _compute_customer_bounds(self) -> None:
        self.min_required_time = [0] * self.num_nodes
        self.depot_roundtrip_time = [0] * self.num_nodes
        self.profit_density_lb = [0.0] * self.num_nodes

        for customer in self.customers:
            best_1 = float("inf")
            best_2 = float("inf")

            for other in range(self.num_nodes):
                if other == customer:
                    continue

                dist = self.distance[customer][other]

                if dist < best_1:
                    best_2 = best_1
                    best_1 = dist
                elif dist < best_2:
                    best_2 = dist

            self.min_required_time[customer] = (
                int(best_1)
                + int(best_2)
                + self.service[customer]
            )

            self.depot_roundtrip_time[customer] = (
                self.distance[0][customer]
                + self.service[customer]
                + self.distance[customer][0]
            )

            self.profit_density_lb[customer] = (
                self.profit[customer]
                / max(1, self.min_required_time[customer])
            )

    def _compute_nearest_customers(self) -> None:
        self.nearest_customers = [[] for _ in range(self.num_nodes)]

        for customer in self.customers:
            neighbors = [
                other
                for other in self.customers
                if other != customer
            ]

            neighbors.sort(
                key=lambda other: self.distance[customer][other]
            )

            self.nearest_customers[customer] = neighbors


def normalize_key(text: str) -> str:
    return text.strip().upper().replace(" ", "_")


def read_instance(path: str) -> Instance:
    return Instance.from_file(path)