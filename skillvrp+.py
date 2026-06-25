#!/usr/bin/env python3
"""
skillvrp.py
Prize-Collecting Skill Vehicle Routing Problem (SkillVRP).

A metaheuristic that selects and routes a subset of customers in order to
maximize the collected profit, while respecting:
  * skills      (a courier may serve a customer only if it holds all required skills)
  * time windows (service must begin no later than the customer's due time; early
                  arrival waits until the ready time)
  * courier shifts (leave the pharmacy >= start_b, return <= end_b)

High level approach:
  1. Greedy insertion heuristic  -> initial feasible solution
  2. Large Neighborhood Search   -> destroy (remove customers) + repair (greedy
     re-insertion), with Record-to-Record-Travel acceptance.

Usage:
    python3 skillvrp.py <path to instance> <timeout (s)>
"""

import argparse
import math
import random
import sys
import time


# ===========================================================================
# Instance / data model
# ===========================================================================
class Instance:
    """Holds all data of a SkillVRP instance. Node 0 is the pharmacy."""

    def __init__(self):
        self.name = ""
        self.n_nodes = 0          # DIMENSION = C + 1 (incl. pharmacy)
        self.n_vehicles = 0       # B
        self.n_skills = 0         # S

        # Node attributes, indexed by node id (0 == pharmacy)
        self.x = []
        self.y = []
        self.ready = []           # e_c
        self.due = []             # l_c
        self.service = []         # s_c
        self.profit = []          # p_c
        self.req_skills = []      # list[frozenset]  -> R_c

        # Vehicle attributes, indexed 0..B-1  (courier id == index + 1)
        self.v_start = []
        self.v_end = []
        self.v_skills = []        # list[frozenset]  -> Q_b

        self.dist = None          # distance matrix, list[list[int]]

    @property
    def n_customers(self):
        return self.n_nodes - 1


def _euclid_rounded(dx, dy):
    """Euclidean distance rounded to the nearest integer (round-half-up).

    NOTE: The PDF only says 'rounded to the nearest integer'. This uses the
    common TSPLIB EUC_2D convention int(d + 0.5). If the provided checker.py
    rounds differently, change THIS single line to match it, otherwise a
    solution you believe is feasible may be rejected on tight time windows.
    """
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def parse_instance(path):
    """Parse an instance file into an Instance object.

    The parser is whitespace-agnostic (blanks, tabs, newlines may separate any
    two values) and tolerant to the '_' vs ' ' spelling of keywords
    (NODE_SECTION vs 'NODE SECTION', EUC_2D vs 'EUC 2D', ...): underscores are
    normalized to spaces before tokenizing, then the grammar is parsed from a
    flat token stream.
    """
    with open(path, "r") as f:
        raw = f.read()
    tok = raw.replace("_", " ").split()

    def find_marker(words, start=0, end=None):
        if end is None:
            end = len(tok)
        wlen = len(words)
        words_up = [w.upper() for w in words]
        for idx in range(start, end - wlen + 1):
            if [t.upper() for t in tok[idx:idx + wlen]] == words_up:
                return idx
        return -1

    node_marker = find_marker(["NODE", "SECTION"])
    veh_marker = find_marker(["VEHICLE", "SECTION"])
    if node_marker < 0 or veh_marker < 0:
        raise ValueError("Could not locate NODE/VEHICLE sections in instance.")

    def header_int(key_words):
        idx = find_marker(key_words, 0, node_marker)
        if idx < 0:
            raise ValueError("Missing header field: " + " ".join(key_words))
        j = idx + len(key_words)
        while j < node_marker:
            try:
                return int(tok[j])
            except ValueError:
                j += 1
        raise ValueError("No integer found after header field " + " ".join(key_words))

    inst = Instance()
    inst.n_nodes = header_int(["DIMENSION"])
    inst.n_vehicles = header_int(["VEHICLES"])
    inst.n_skills = header_int(["NUM", "SKILLS"])

    n = inst.n_nodes
    inst.x = [0] * n
    inst.y = [0] * n
    inst.ready = [0] * n
    inst.due = [0] * n
    inst.service = [0] * n
    inst.profit = [0] * n
    inst.req_skills = [frozenset()] * n

    # ---- NODE SECTION : id x y e l s p k r1..rk  (C+1 rows) ----
    c = node_marker + 2  # skip the two marker tokens
    for _ in range(n):
        nid = int(tok[c]); c += 1
        x = int(tok[c]); c += 1
        y = int(tok[c]); c += 1
        e = int(tok[c]); c += 1
        l = int(tok[c]); c += 1
        s = int(tok[c]); c += 1
        p = int(tok[c]); c += 1
        k = int(tok[c]); c += 1
        skills = []
        for _ in range(k):
            skills.append(int(tok[c])); c += 1
        inst.x[nid] = x
        inst.y[nid] = y
        inst.ready[nid] = e
        inst.due[nid] = l
        inst.service[nid] = s
        inst.profit[nid] = p
        inst.req_skills[nid] = frozenset(skills)

    # ---- VEHICLE SECTION : id start end k q1..qk  (B rows) ----
    B = inst.n_vehicles
    inst.v_start = [0] * B
    inst.v_end = [0] * B
    inst.v_skills = [frozenset()] * B
    c = veh_marker + 2
    for _ in range(B):
        vid = int(tok[c]); c += 1
        start = int(tok[c]); c += 1
        end = int(tok[c]); c += 1
        k = int(tok[c]); c += 1
        skills = []
        for _ in range(k):
            skills.append(int(tok[c])); c += 1
        b = vid - 1                      # couriers are 1-indexed in the file
        inst.v_start[b] = start
        inst.v_end[b] = end
        inst.v_skills[b] = frozenset(skills)

    # ---- distance matrix (symmetric, integer) ----
    xs, ys = inst.x, inst.y
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = xs[i], ys[i]
        row = dist[i]
        for j in range(i + 1, n):
            d = _euclid_rounded(xi - xs[j], yi - ys[j])
            row[j] = d
            dist[j][i] = d
    inst.dist = dist
    return inst


# ===========================================================================
# Timing / feasibility
# ===========================================================================
def compute_timing(inst, b, route):
    """Forward time computation for `route` on courier b.

    Returns (feasible, begin, wait, depart, slack):
      begin[i]  : time service starts at route[i]
      wait[i]   : idle time before service at route[i]  (>=0)
      depart[i] : time service ends at route[i]
      slack[i]  : max amount by which begin[i] may be increased without making
                  any node i..end (incl. the return to the depot) infeasible.

    `slack` enables O(1) insertion-feasibility checks (Savelsbergh forward time
    slack). It is only valid for the courier the route was built for.
    """
    dist = inst.dist
    Q = inst.v_skills[b]
    m = len(route)
    begin = [0] * m
    wait = [0] * m
    depart = [0] * m

    t = inst.v_start[b]
    prev = 0
    for i, c in enumerate(route):
        if not inst.req_skills[c] <= Q:
            return False, None, None, None, None
        arr = t + dist[prev][c]
        if arr > inst.due[c]:                 # constraint 8: arrival must be <= l_c
            return False, None, None, None, None
        w = inst.ready[c] - arr
        if w < 0:
            w = 0
        bg = arr + w
        wait[i] = w
        begin[i] = bg
        depart[i] = bg + inst.service[c]
        t = depart[i]
        prev = c

    ret = t + dist[prev][0]
    if ret > inst.v_end[b]:                   # must return within the shift
        return False, None, None, None, None

    slack = [0] * m
    if m > 0:
        slack_end = inst.v_end[b] - ret       # >= 0 here
        slack[m - 1] = min(inst.due[route[m - 1]] - begin[m - 1], slack_end)
        for i in range(m - 2, -1, -1):
            slack[i] = min(inst.due[route[i]] - begin[i], wait[i + 1] + slack[i + 1])
    return True, begin, wait, depart, slack


def insertion_feasible(inst, b, route, timing, pos, u):
    """True iff customer u can be inserted at index `pos` (0..len(route)) of
    `route` on courier b. O(1) using the cached `timing` of that route."""
    if not inst.req_skills[u] <= inst.v_skills[b]:
        return False
    dist = inst.dist
    begin, wait, depart, slack = timing
    m = len(route)

    if pos == 0:
        prev_node = 0
        prev_depart = inst.v_start[b]
    else:
        prev_node = route[pos - 1]
        prev_depart = depart[pos - 1]

    a_u = prev_depart + dist[prev_node][u]
    if a_u > inst.due[u]:
        return False
    begin_u = a_u if a_u >= inst.ready[u] else inst.ready[u]
    depart_u = begin_u + inst.service[u]

    if pos == m:                               # u becomes the last stop
        return depart_u + dist[u][0] <= inst.v_end[b]

    nxt = route[pos]
    a_next = depart_u + dist[u][nxt]
    if a_next > inst.due[nxt]:
        return False
    new_begin_next = a_next if a_next >= inst.ready[nxt] else inst.ready[nxt]
    delay = new_begin_next - begin[pos]
    if delay <= 0:
        return True
    return delay <= slack[pos]


# ===========================================================================
# Solution representation
# ===========================================================================
class Solution:
    __slots__ = ("inst", "routes", "timing", "served", "profit")

    def __init__(self, inst):
        self.inst = inst
        self.routes = [[] for _ in range(inst.n_vehicles)]
        self.timing = [(_t[1], _t[2], _t[3], _t[4]) for _t in
                       (compute_timing(inst, b, []) for b in range(inst.n_vehicles))]
        self.served = set()
        self.profit = 0

    def recompute(self, b):
        feas, begin, wait, depart, slack = compute_timing(self.inst, b, self.routes[b])
        self.timing[b] = (begin, wait, depart, slack)
        return feas

    def insert(self, b, pos, u):
        self.routes[b].insert(pos, u)
        self.served.add(u)
        self.profit += self.inst.profit[u]
        self.recompute(b)

    def remove_customers(self, custs):
        cset = set(custs)
        affected = []
        for b in range(self.inst.n_vehicles):
            r = self.routes[b]
            if any(c in cset for c in r):
                self.routes[b] = [c for c in r if c not in cset]
                affected.append(b)
        for c in cset:
            if c in self.served:
                self.served.discard(c)
                self.profit -= self.inst.profit[c]
        for b in affected:
            self.recompute(b)

    def clone(self):
        s = Solution.__new__(Solution)
        s.inst = self.inst
        s.routes = [r[:] for r in self.routes]
        s.timing = list(self.timing)   # tuples are replaced wholesale, never mutated
        s.served = set(self.served)
        s.profit = self.profit
        return s


# ===========================================================================
# Construction / repair: greedy insertion
# ===========================================================================
def greedy_repair(inst, sol, candidates, deadline):
    """Insert `candidates` (customer ids) greedily. Each customer is placed at
    the feasible position with the smallest travel detour; customers are tried
    in order of decreasing profit (ties: tighter time window first). Reused both
    as the construction heuristic and as the LNS repair operator."""
    order = sorted(candidates,
                   key=lambda c: (-inst.profit[c], inst.due[c] - inst.ready[c]))
    dist = inst.dist
    B = inst.n_vehicles
    check_every = 64
    cnt = 0
    for u in order:
        cnt += 1
        if cnt % check_every == 0 and time.time() > deadline:
            break
        if u in sol.served:
            continue
        ru = inst.req_skills[u]
        best_b = -1
        best_pos = -1
        best_detour = None
        for b in range(B):
            if not ru <= inst.v_skills[b]:
                continue
            route = sol.routes[b]
            timing = sol.timing[b]
            L = len(route)
            du = dist[u]
            for pos in range(L + 1):
                if not insertion_feasible(inst, b, route, timing, pos, u):
                    continue
                prev_node = 0 if pos == 0 else route[pos - 1]
                next_node = 0 if pos == L else route[pos]
                detour = dist[prev_node][u] + du[next_node] - dist[prev_node][next_node]
                if best_detour is None or detour < best_detour:
                    best_detour = detour
                    best_b = b
                    best_pos = pos
        if best_b >= 0:
            sol.insert(best_b, best_pos, u)
    return sol


# ===========================================================================
# Destroy operators
# ===========================================================================
def _route_of(sol):
    """Map customer id -> (route index b, position)."""
    loc = {}
    for b, r in enumerate(sol.routes):
        for i, c in enumerate(r):
            loc[c] = (b, i)
    return loc


def random_removal(sol, q, rng):
    served = list(sol.served)
    if not served:
        return []
    q = min(q, len(served))
    chosen = rng.sample(served, q)
    sol.remove_customers(chosen)
    return chosen


def worst_removal(sol, q, rng):
    """Remove customers with the poorest profit-per-detour ratio (expensive to
    keep), with randomization so the same set is not removed every time."""
    inst = sol.inst
    dist = inst.dist
    served = list(sol.served)
    if not served:
        return []
    q = min(q, len(served))
    scored = []
    for b, r in enumerate(sol.routes):
        L = len(r)
        for i, c in enumerate(r):
            prev = 0 if i == 0 else r[i - 1]
            nxt = 0 if i == L - 1 else r[i + 1]
            detour = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
            # rounded distances may slightly violate the triangle inequality,
            # so `detour` can be negative -> clamp the denominator.
            ratio = inst.profit[c] / (max(detour, 0.0) + 1.0)   # low ratio == worst
            scored.append((ratio, c))
    scored.sort(key=lambda t: t[0])                   # ascending: worst first
    pool = [c for _, c in scored]
    chosen = []
    while len(chosen) < q and pool:
        idx = int((rng.random() ** 3) * len(pool))    # bias toward the worst
        chosen.append(pool.pop(idx))
    sol.remove_customers(chosen)
    return chosen


def related_removal(sol, q, rng):
    """Shaw-style removal: pick a seed customer and remove its nearest served
    neighbors, encouraging the repair to re-route a local cluster."""
    inst = sol.inst
    dist = inst.dist
    served = list(sol.served)
    if not served:
        return []
    q = min(q, len(served))
    seed = rng.choice(served)
    served.sort(key=lambda c: dist[seed][c])
    chosen = served[:q]
    sol.remove_customers(chosen)
    return chosen


# ===========================================================================
# Metaheuristic: Large Neighborhood Search
# ===========================================================================
def solve(inst, deadline, rng):
    """Run construction + LNS until `deadline` (wall-clock). Returns best Solution."""
    # Precompute the customers that can be served by at least one courier as a
    # singleton tour. Customers failing this can never be inserted -> skip them.
    feasible_customers = []
    for c in range(1, inst.n_nodes):
        ok = False
        for b in range(inst.n_vehicles):
            if inst.req_skills[c] <= inst.v_skills[b] and \
                    compute_timing(inst, b, [c])[0]:
                ok = True
                break
        if ok:
            feasible_customers.append(c)
    feasible_set = set(feasible_customers)

    current = Solution(inst)
    greedy_repair(inst, current, feasible_customers, deadline)
    best = current.clone()

    if not feasible_customers or time.time() >= deadline:
        return best

    # Record-to-Record-Travel acceptance band, shrinking with remaining time.
    start = time.time()
    total = max(1e-6, deadline - start)
    base_dev = max(1.0, 0.03 * max(1, best.profit))

    destroy_ops = [random_removal, worst_removal, related_removal]

    n_serv_cap = len(feasible_customers)
    while True:
        now = time.time()
        if now >= deadline:
            break
        remaining = deadline - now
        deviation = base_dev * (remaining / total)

        cand = current.clone()
        s = len(cand.served)
        q_max = max(1, min(s // 4 + 1, 40)) if s else 1
        q = rng.randint(1, q_max)
        op = rng.choice(destroy_ops)
        op(cand, q, rng)

        pool = [c for c in feasible_set if c not in cand.served]
        greedy_repair(inst, cand, pool, deadline)

        if cand.profit >= best.profit - deviation:
            current = cand
            if cand.profit > best.profit:
                best = cand.clone()
                if best.profit >= sum(inst.profit[c] for c in feasible_customers):
                    break  # every servable customer is already served
    return best


# ===========================================================================
# Output
# ===========================================================================
def print_solution(inst, sol, out=sys.stdout):
    lines = ["###RESULT: Feasible", "###OBJECTIVE: %d" % sol.profit]
    for b in range(inst.n_vehicles):
        seq = [0] + sol.routes[b] + [0]      # empty route -> "0 0"
        lines.append("###VEHICLE %d: %s" % (b + 1, " ".join(str(x) for x in seq)))
    out.write("\n".join(lines) + "\n")


# ===========================================================================
# Entry point
# ===========================================================================
def solve_instance(path, timeout, seed=0):
    t0 = time.time()
    # Reserve a small slice for parsing+output so we never overrun the limit.
    reserve = 0.20 + min(1.0, timeout * 0.02)
    deadline = t0 + max(0.0, timeout - reserve)

    inst = parse_instance(path)
    rng = random.Random(seed)
    best = solve(inst, deadline, rng)
    print_solution(inst, best)


def main():
    parser = argparse.ArgumentParser(description="SkillVRP metaheuristic")
    parser.add_argument("instance", help="Path to the instance file")
    parser.add_argument("timeout", type=int, help="Time limit in seconds")
    args = parser.parse_args()
    solve_instance(args.instance, args.timeout)


if __name__ == "__main__":
    main()


# python skillvrp+.py /home/r3h/Master_DS/Metaheuristics/project/data/dataset/skillvrp_n50_v2_s3_k0.0_1.txt 45
