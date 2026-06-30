# Übersicht ALNS Framework
*Stand 30.06.* 

**zu nutzende Dateien sind;**
```
alns.py
instance_reader.py
initial_solution.py
solution.py
checher.py
test_run.md
```
### Phase 1: Initialisation
...
### Phase 2: Main
#### Destroy Operators
##### Übersicht über gängige Destroy-operatoren (Mara et al., 2022)

| Name | Beschreibung | Referenz(en) |
|------|--------------|--------------|
| Random removal | Dieser Operator entfernt iterativ einen zufällig ausgewählten Knoten aus einer Lösung. | Ropke and Pisinger (2006a; 2006b) |
| Worst removal | Dieser Operator entfernt iterativ einen Knoten, der erheblich zu den Gesamtkosten einer Lösung beiträgt. | Ropke and Pisinger (2006a; 2006b) |
| Shaw removal | Dieser Operator entfernt iterativ den Knoten mit dem höchsten Ähnlichkeitsindex aus einer Lösung. Der Ähnlichkeitsindex eines Knotens wird berechnet, indem der Knoten anhand einer Reihe vordefinierter Kriterien mit einem ausgewählten Saatknoten (seed node) verglichen wird. | Shaw (1998), Ropke and Pisinger (2006a; 2006b) |
| Route removal | Dieser im Vehicle-Routing-Bereich verbreitete Operator entfernt zufällig eine Anzahl von Routen samt aller zugehörigen Knoten aus einer Lösung. | Demir et al. (2012) |
| History-based removal | Dieser Operator entfernt iterativ den Knoten mit der größten Differenz zwischen seinen aktuellen Positionskosten und seinen historisch besten Positionskosten aus einer Lösung. | Ropke and Pisinger (2006a), Pisinger and Ropke (2007) |
| Neighborhood removal | Dieser ursprünglich im Vehicle-Routing-Bereich vorgeschlagene Operator entfernt iterativ einen Knoten, der im Hinblick auf die durchschnittliche Distanz der Route, zu der der Knoten gehört, von Bedeutung ist. | Demir et al. (2012) |
| Proximity-based removal | Dieser Operator ist ein Spezialfall der Shaw-Entfernung, bei dem der Distanzwert zwischen zwei Knoten das einzige als Ähnlichkeitsindex verwendete Kriterium ist. | Demir et al. (2012) |
| Time-based removal | Dieser Operator ist ein Spezialfall der Shaw-Entfernung, bei dem die Differenz eines Zeitmerkmals zwischen zwei Knoten – z. B. die früheste Startzeit der Bedienung – das einzige als Ähnlichkeitsindex verwendete Kriterium ist. | Demir et al. (2012) |
| Demand-based removal | Dieser Operator ist ein Spezialfall der Shaw-Entfernung, bei dem die Nachfragedifferenz zwischen zwei Knoten das einzige als Ähnlichkeitsindex verwendete Kriterium ist. | Demir et al. (2012) |
| Cluster removal | Dieser Operator nutzt einen Clustering-Algorithmus, z. B. Kruskals Algorithmus, um zwei Cluster zu bilden und alle Knoten eines zufällig ausgewählten Clusters aus der Lösung zu entfernen. | Ropke and Pisinger (2006a), Pisinger and Ropke (2007) |

#### Repair Operators
##### Übersicht über gängige Repair-operatoren (Mara et al., 2022)

| Name | Beschreibung | Referenz(en) |
|------|--------------|--------------|
| Greedy insertion | Dieser Operator wählt und fügt iterativ den Knoten ein, der unter allen verbleibenden Knoten in der Liste der entfernten Knoten die geringsten Einfügekosten verursacht. In einigen Arbeiten wird dieser Operator auch *Best insertion* oder *Cheapest insertion* genannt. | Ropke and Pisinger (2006a; 2006b), Pisinger and Ropke (2007) |
| (k-)Regret insertion | Dieser Operator wählt und fügt iterativ den Knoten ein, der unter den verbleibenden Knoten in der Liste der entfernten Knoten den größten Reue-Wert (regret value) aufweist. Der Reue-Wert eines Knotens ergibt sich aus der Differenz zwischen den Kosten der besten Einfügeposition und den Kosten der k-besten Einfügeposition. | Ropke and Pisinger (2006a; 2006b), Pisinger and Ropke (2007) |
| Random insertion | Dieser Operator wählt zufällig einen Knoten aus der Liste der entfernten Knoten und fügt ihn an der Position mit den geringsten zusätzlichen Kosten ein. | Coelho et al. (2012a; 2012b), Qu and Bard (2012; 2013) |
| Sequential insertion | Dieser Operator fügt die Knoten aus der Liste der entfernten Knoten nacheinander an der jeweils kostengünstigsten Einfügeposition in die Lösung ein. | Kovacs et al. (2012) |
| Shaw insertion | Dieser Operator nutzt das bei der Shaw-Entfernung beschriebene Konzept des Ähnlichkeitsindex, um den nächsten in die Lösung einzufügenden Knoten aus der Liste der entfernten Knoten auszuwählen. | Coelho et al. (2012a; 2012b) |
| Swap insertion | Dieser Operator wählt zufällig zwei Knoten in der Lösung aus und vertauscht sie. | Coelho et al. (2012a) |
| Zone insertion | Dieser ursprünglich für den Vehicle-Routing-Bereich vorgeschlagene Operator ähnelt der Greedy insertion, verwendet jedoch das Zeitfenster-Kriterium statt der Distanz, um die beste Einfügeposition eines Knotens zu bestimmen. | Demir et al. (2012) |
| Cluster insertion | Dieser Operator teilt die entfernten Knoten in eine Anzahl von Clustern ein, bevor die Einfügungen durchgeführt werden. | Maknoon and Laporte (2017), Santini (2019) |

#### Acceptance Criteria
Wir nutzen den SA Framework. Es besteht noch die Möglichkeit $T_0$ instance dependent festzulegen.
#### Adaptive Weights Adjustment
W`keit die Operatoren zu nutzen (wird periodisch angepasst)

## ToDo
- die passenden Operatoren auswählen
- PenaltyParams und SAParams tunen
- maybe $T_0$ instance dependent machen
- acceleration techniques nutzen?, wenn Feasibility Überprüfung nicht effizient ist (ALNS hat ja 30s, um zu performen)
