# Project 2 - Advanced Search Algorithms

**Bachelor's Degree in Digital Technologies and Artificial Intelligence - 2025/2026**

Authors: Ilie Iftime (112779), Ines Cruz (123557), Sofia Quintino (123554)

Course: Advanced Search Algorithms  
Lecturer: Ricardo Miranda

## Overview

This project implements, from scratch, a set of search and optimization algorithms applied to three classic problems in Artificial Intelligence and combinatorial optimization:

1. **N-Queens Problem** - Constraint Satisfaction Problems (CSP), Genetic Algorithm, and Tabu Search
2. **Travelling Salesman Problem (TSP)** - Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO)
3. **Capacitated Vehicle Routing Problem (CVRP)** - Genetic Algorithm with Giant-Tour + Split representation

All algorithms are implemented without external optimization libraries. Experiments are fully reproducible through a fixed random seed and are supported by quantitative metrics, statistical tests (Wilcoxon), and rich visualizations.

## Repository Structure

```text
.
├── Projeto_AAP_112779_123557_123554_VF_PART1.ipynb   Main notebook (Part 1)
├── Projeto_AAP_112779_123557_123554_VF_PART2.ipynb   Main notebook (Part 2)
├── utils_algoritmos.py                               Core algorithm library
├── reproduce_results.py                              Script to reproduce numerical results
├── requirements.txt                                  Python dependencies
├── data/                                             Synthetic instances and generator
├── results/                                          CSV outputs, seeds.csv and figures/ (PNG 300 DPI)
├── Relatório_AAP_112779_123557_123554PROJ1.pdf       Report for Project 1
├── Relatório2_AAP_112779.123557.123554 v2.pdf        Report for Project 2
└── README.md
```

## Prerequisites

- Python 3.9+
- pip
- Virtual environment tool (venv, included in Python standard library)

## Setting Up the Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

### 1. Create a virtual environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following dependencies:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
plotly>=5.18
ipywidgets>=8.0
jupyter>=1.0
nbformat>=5.9
```

## How to Run

1. Activate the virtual environment (if not already active):
   ```bash
   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

2. Launch the notebooks:
   ```bash
   jupyter notebook Projeto_AAP_112779_123557_123554_VF_PART1.ipynb
   jupyter notebook Projeto_AAP_112779_123557_123554_VF_PART2.ipynb
   ```

3. Run all cells in order.

4. To reproduce only the numerical results (without visualizations):
   ```bash
   python reproduce_results.py
   ```

## Reproducibility

All experiments use a fixed random seed:

```python
RANDOM_SEED = 42
```

The actual seeds used in each run are recorded in `results/seeds.csv` for full verification.

## Project Modules

### Problem 1 - N-Queens

- **CSP with Forward Checking and AC-3** - Backtracking search with Minimum Remaining Values (MRV) heuristic and constraint propagation.
- **Genetic Algorithm** - Permutation-based representation, tournament selection, Order Crossover (OX), swap mutation, and elitism.
- **Tabu Search** - Local search with short-term memory (tabu list) to escape local minima, carried over from Project 1.

Benchmarked for N = 8, 16, 32, and 64 with 10 independent runs per configuration.

### Problem 2 - Travelling Salesman Problem (TSP)

- **Particle Swarm Optimization (PSO)** - Adapted to the permutation domain using swap-based velocity representation.
- **Ant Colony Optimization (ACO)** - Pheromone-based construction with distance heuristic and evaporation, tuned via grid search over beta and rho.

Benchmarked on instances with 20, 50, and 100 cities.

### Problem 3 - Capacitated Vehicle Routing Problem (CVRP)

- **Genetic Algorithm with Giant-Tour + Split** - Chromosomes represent client permutations, with routes constructed automatically by a Split operator that respects vehicle capacity constraints.
- Tournament selection, Order Crossover (OX), and mutation via swaps and segment inversions.

Benchmarked on instances with 50 and 100 clients.

## Key Results

### N-Queens
- CSP with Forward Checking and AC-3 achieved **100% success rate** across all tested dimensions.
- Forward Checking was the most efficient overall, solving N=64 in tens of milliseconds.
- The Genetic Algorithm degraded progressively (100% at N=8 to 50% at N=64).
- Tabu Search remained robust at 100% success but with higher execution times than CSP.
- Wilcoxon test confirmed a statistically significant advantage for Forward Checking (W=0.0, p < 0.05).

### TSP
- ACO consistently outperformed PSO in all instances (20, 50, 100 cities).
- ACO produced lower costs and lower variability across runs.
- Best ACO configuration: beta = 5.0, rho = 0.1.
- Wilcoxon test: W = 0.0, p = 6.104e-5, confirming statistical significance.

### CVRP
- Genetic Algorithm with Giant-Tour + Split produced feasible solutions in all runs.
- No capacity constraint violations observed.
- Vehicle usage close to the theoretical minimum.
- Stable results across independent runs.

## Evaluation Checklist

| Requirement | Location |
|---|---|
| CSP with Forward Checking and AC-3 | `solve_csp_nqueens` |
| Genetic Algorithm for N-Queens | `ga_nqueens` |
| Comparison for N = 8, 16, 32, 64 | `reproduce_results.py`, `results/nqueens_results.csv` |
| Metrics: time, nodes/iterations, success rate | Tables and figures in notebooks |
| PSO for TSP | `pso_tsp` |
| ACO for TSP | `aco_tsp` |
| 3 TSP instances (20, 50, 100) | `data/tsp_*.csv` |
| Parameter analysis (PSO and ACO) | Grid search in notebook + `results/figures/` |
| GA for CVRP | `ga_cvrp` |
| 2 CVRP instances (50, 100) | `data/cvrp_*.csv` |
| Statistical tests (Wilcoxon) | Comparison sections in notebooks |
| Rich visualizations (300 DPI) | `results/figures/` |
| Interactive dashboard | Final section of notebook (Plotly) |

## Hardware Notes

Full experiments run in a few minutes on a standard laptop. The heaviest case in Problem 1 is the Genetic Algorithm at N=64.

## Limitations

- Experiments use moderate-size synthetic instances; real-world instances may present different characteristics.
- Parameter grids for PSO and ACO were limited by computational time constraints.
- Only pure algorithms were tested; hybrid approaches were not explored.

## Future Work

- Develop memetic algorithms combining evolutionary search with local improvement.
- Integrate 2-opt local search operators into TSP metaheuristics.
- Extend CVRP to more realistic variants: time windows, multiple depots, heterogeneous fleets, and fixed vehicle costs.
- Explore hybrid paradigms combining propagation, repair, and metaheuristic mechanisms.

## Report

- Project 1 report: `Relatório_AAP_112779_123557_123554PROJ1.pdf`
- Project 2 report: `Relatório2_AAP_112779.123557.123554 v2.pdf`

## License

Project developed for academic purposes.
