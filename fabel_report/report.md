# 2. Citations and Discussion of Relevant Work in the Literature

## 2.1 Problem classification

The Prize-Collecting Skill VRP (PC-SVRP) combines two well-studied problem
families. Its *selective* aspect — customers are optional and the objective is
to maximize collected profit under route-duration/time-window restrictions —
makes it a member of the Orienteering Problem (OP) family
[vansteenwegen2011orienteering]. With multiple couriers and hard time windows
it is, at its routing core, a **Team Orienteering Problem with Time Windows
(TOPTW)** [labadie2012team; gunawan2016orienteering]. The *skill* aspect —
each request may only be served by a courier holding all required
certifications — corresponds to the **Skill VRP** introduced by Cappanera et
al. [cappanera2011skill] and to the **Technician Routing and Scheduling
Problem (TRSP)** [kovacs2012adaptive; pillac2013parallel], which additionally
features heterogeneous worker shifts exactly as in our setting. The PC-SVRP is
NP-hard, since it contains the TOP (and hence the TSP) as a special case.

## 2.2 Metaheuristics for the TOPTW

The TOPTW literature is dominated by trajectory-based local search.
Vansteenwegen et al. [vansteenwegen2009iterated] proposed a fast **Iterated
Local Search (ILS)** with a cheap insertion step and a *shake* removal step;
it remains the reference for very low running times. Labadie et al.
[labadie2012team] improved solution quality with an LP-based **granular VNS**,
and Hu & Lim [hu2014iterative] obtained many best-known solutions with an
**iterative three-component heuristic (I3CH)** combining local search,
simulated annealing and route recombination. Gunawan et al.
[gunawan2017welltuned] showed that a well-tuned ILS/SA hybrid matches these
results at lower cost. A key efficiency idea shared by these works is
constant-time feasibility checking of insertions via forward time slacks
(maximum shift values) [savelsbergh1992vehicle; vansteenwegen2009iterated],
which we adopt. Recent work extends TOPTW variants with learning-augmented
simulated annealing [yu2024set], confirming that SA/LNS-style acceptance over
insertion/removal neighborhoods is still the state of the art for selective
routing with time windows.

## 2.3 Skill and workforce constraints

Cappanera et al. [cappanera2011skill] study skill-compatibility in routing
from a polyhedral perspective; heuristically, skills are handled as hard
compatibility filters on the customer–vehicle assignment. Kovacs et al.
[kovacs2012adaptive] solve the TRSP — time windows, skill requirements and
outsourcing (i.e., optional service at a cost, structurally equivalent to our
prize collection) — with an **Adaptive Large Neighborhood Search (ALNS)** and
report it clearly outperforming exact solvers beyond small instances. Pillac
et al. [pillac2013parallel] confirm this with a parallel matheuristic for the
TRSP. The home-health-care routing literature, surveyed by Cissé et al.
[cisse2017or], reaches the same conclusion: destroy-and-repair frameworks
handle the combination of compatibility constraints and optional visits most
naturally, because the *selection* decision (which customers to serve) is
re-optimized in every repair step.

## 2.4 State-of-the-art VRP metaheuristics

For classical (non-selective) VRPs, two frameworks currently define the state
of the art. **LNS/ALNS** [shaw1998using; ropke2006adaptive; pisinger2007general]
repeatedly destroys part of the solution (random, related/Shaw, worst removal)
and repairs it with greedy or regret-k insertion; **SISR** ("slack induction
by string removals") [christiaens2020slack] is a remarkably simple LNS variant
that is state-of-the-art on many variants. **Hybrid Genetic Search (HGS)**
[vidal2012hybrid; vidal2022hybrid] — a genetic algorithm over giant-tour
chromosomes with aggressive local search (incl. SWAP*) and diversity
management — produces the best known results on CVRP/VRPTW and won the
EURO-meets-NeurIPS 2022 competition [kool2022euro], and has been generalized
to multi-attribute ("rich") VRPs [vidal2014unified].

## 2.5 Synthesis: choice of metaheuristic

Which of these is "optimal" for the PC-SVRP? HGS owes much of its strength to
giant-tour Split decoding and dense route-improvement neighborhoods, both of
which degrade when customer *selection* is part of the problem and when hard
time windows, shifts and skill filters make most crossover offspring
infeasible; published HGS results target settings where all customers are
served. In contrast, the evidence from the two closest problem classes points
in the same direction: for the TOPTW the best methods are ILS/SA-style
insertion–removal searches [vansteenwegen2009iterated; hu2014iterative;
gunawan2017welltuned], and for skill-constrained routing with optional service
the best published method is ALNS [kovacs2012adaptive]. We therefore adopt an
**ALNS/LNS framework with skill-aware regret-k insertion, Shaw-/SISR-style and
profit-based removal operators, SA acceptance, and constant-time time-window
feasibility checks**, i.e., the combination that constitutes the current state
of the art for prize-collecting, skill-constrained routing — and is moreover
robust within the short, Python-level time budgets imposed by the project
(cf. the time-limited command line), where the cheap-move efficiency of
ILS/ALNS matters more than the asymptotic quality of population methods.
