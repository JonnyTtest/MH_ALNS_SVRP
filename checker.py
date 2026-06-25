"""Standalone solution checker for the prize-collecting Skill-VRP.

It verifies a solution file against an instance:
  * every customer is visited at most once (visiting is optional),
  * each route starts and ends at the depot (node 0),
  * each route respects the driver's skills, the customer time windows and the
    driver's working shift,
  * the reported objective matches the recomputed total collected profit.

Travel durations are rounded Euclidean distances between node coordinates and
are identical for all drivers. The objective is to MAXIMISE collected profit.

Usage:
    python3 checker.py <path_to_instance> <path_to_solution>

Exit code 0 on a valid feasible solution, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Set, Tuple


# --------------------------------------------------------------------------- #
# Instance parsing
# --------------------------------------------------------------------------- #
class Instance:
    def __init__(self, path: str):
        header: Dict[str, str] = {}
        sections: Dict[str, List[str]] = {}
        current = None
        with open(path) as f:
            text = f.read()
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ":" in line and not line.endswith("SECTION"):
                key, val = line.split(":", 1)
                header[key.strip().upper()] = val.strip()
                current = None
            elif line.endswith("SECTION"):
                current = line.upper()
                sections[current] = []
            elif line.upper() == "EOF":
                current = None
            elif current is not None:
                sections[current].append(line)

        self.name = header.get("NAME", "instance")
        dim = int(header["DIMENSION"])
        self.num_customers = dim - 1          # excluding depot (node 0)
        self.num_vehicles = int(header["VEHICLES"])
        self.num_skills = int(header["NUM_SKILLS"])

        # One row per node: id x y ready due service profit k skill1 .. skillk
        self.coords: List[Tuple[float, float]] = [(0.0, 0.0)] * dim
        self.time_windows: List[Tuple[int, int]] = [(0, 0)] * dim
        self.service_times: List[int] = [0] * dim
        self.profits: List[int] = [0] * dim
        self.cust_skills: List[Set[int]] = [set() for _ in range(dim)]
        for line in sections["NODE_SECTION"]:
            p = line.split()
            idx = int(p[0])
            self.coords[idx] = (float(p[1]), float(p[2]))
            self.time_windows[idx] = (int(p[3]), int(p[4]))
            self.service_times[idx] = int(p[5])
            self.profits[idx] = int(p[6])
            k = int(p[7])
            self.cust_skills[idx] = set(int(s) for s in p[8:8 + k])

        # Vehicle rows are 1-indexed in the file.
        self.veh_shift: List[Tuple[int, int]] = [(0, 0)] * self.num_vehicles
        self.veh_skills: List[Set[int]] = [set() for _ in range(self.num_vehicles)]
        for line in sections["VEHICLE_SECTION"]:
            p = line.split()
            vid = int(p[0]) - 1
            self.veh_shift[vid] = (int(p[1]), int(p[2]))
            k = int(p[3])
            self.veh_skills[vid] = set(int(s) for s in p[4:4 + k])

        self.num_nodes = dim
        self._dist = [[0] * dim for _ in range(dim)]
        for i in range(dim):
            xi, yi = self.coords[i]
            for j in range(dim):
                xj, yj = self.coords[j]
                self._dist[i][j] = int(round(math.hypot(xi - xj, yi - yj)))

    def dist(self, i: int, j: int) -> int:
        return self._dist[i][j]


def route_duration_and_feasibility(inst: Instance, vehicle: int,
                                   route: List[int]) -> Tuple[int, bool, str]:
    """Evaluate a single route (customer ids, without depot endpoints).

    Returns (travel_duration, feasible, message). travel_duration includes the
    depot legs.
    """
    shift_start, shift_end = inst.veh_shift[vehicle]
    skills = inst.veh_skills[vehicle]

    travel = 0
    time = shift_start
    prev = 0  # depot
    for c in route:
        if not inst.cust_skills[c].issubset(skills):
            return travel, False, (
                f"vehicle {vehicle + 1} lacks skills for customer {c} "
                f"(needs {sorted(inst.cust_skills[c])}, has {sorted(skills)})")
        d = inst.dist(prev, c)
        travel += d
        arrival = time + d
        ready, due = inst.time_windows[c]
        if arrival > due:
            return travel, False, (
                f"vehicle {vehicle + 1} arrives at customer {c} at {arrival} "
                f"> due {due}")
        time = max(arrival, ready) + inst.service_times[c]
        prev = c
    d = inst.dist(prev, 0)
    travel += d
    time += d
    if time > shift_end:
        return travel, False, (
            f"vehicle {vehicle + 1} returns to depot at {time} > shift end "
            f"{shift_end}")
    return travel, True, "ok"


# --------------------------------------------------------------------------- #
# Solution parsing and checking
# --------------------------------------------------------------------------- #
def parse_solution(path: str) -> Tuple[str, int, Dict[int, List[int]]]:
    """Return (result, objective, {vehicle_id: [nodes...]})."""
    result = None
    objective = None
    routes: Dict[int, List[int]] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("###RESULT:"):
                result = line.split(":", 1)[1].strip()
            elif line.startswith("###OBJECTIVE:"):
                objective = int(line.split(":", 1)[1].strip())
            elif line.startswith("###VEHICLE"):
                head, body = line.split(":", 1)
                vid = int(head.replace("###VEHICLE", "").strip())
                routes[vid] = [int(x) for x in body.split()]
    return result, objective, routes


def check(inst: Instance, sol_path: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    result, objective, routes = parse_solution(sol_path)

    if result == "Timeout":
        return True, ["solution reports Timeout (no feasibility claim)"]
    if result != "Feasible":
        return False, [f"unexpected ###RESULT: {result!r}"]

    visited: Dict[int, int] = {}

    for vid, nodes in sorted(routes.items()):
        if vid < 1 or vid > inst.num_vehicles:
            errors.append(f"vehicle id {vid} out of range")
            continue
        if not nodes or nodes[0] != 0 or nodes[-1] != 0:
            errors.append(f"vehicle {vid} route must start and end at depot 0: {nodes}")
            continue
        inner = nodes[1:-1]
        if 0 in inner:
            errors.append(f"vehicle {vid} visits depot mid-route: {nodes}")
        for c in inner:
            if c < 1 or c > inst.num_customers:
                errors.append(f"vehicle {vid} visits invalid node {c}")
            elif c in visited:
                errors.append(f"customer {c} visited by vehicles "
                              f"{visited[c]} and {vid}")
            else:
                visited[c] = vid

        _, ok, msg = route_duration_and_feasibility(inst, vid - 1, inner)
        if not ok:
            errors.append(f"vehicle {vid} infeasible: {msg}")

    # Objective: total profit of the visited customers (visiting is optional).
    profit = sum(inst.profits[c] for c in visited)

    if objective is None:
        errors.append("missing ###OBJECTIVE line")
    elif objective != profit:
        errors.append(f"reported objective {objective} != recomputed {profit}")

    if not errors:
        n_total = inst.num_customers
        return True, [f"OK: feasible, profit = {profit}, "
                      f"visited {len(visited)}/{n_total} customers"]
    return False, errors


def main():
    p = argparse.ArgumentParser(description="Skill-VRP solution checker")
    p.add_argument("instance", type=str)
    p.add_argument("solution", type=str)
    args = p.parse_args()

    inst = Instance(args.instance)
    ok, msgs = check(inst, args.solution)
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


# python checker.py /home/r3h/Master_DS/Metaheuristics/project/data/dataset/skillvrp_n50_v2_s3_k0.0_1.txt  /home/r3h/Master_DS/Metaheuristics/project/data/sol_ins01.txt