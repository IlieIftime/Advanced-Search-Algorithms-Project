"""Gera instâncias sintéticas para o Projeto 2 com semente fixa (`RANDOM_SEED=42`)
e grava os ficheiros CSV na pasta `data/`. Inclui também as funções de leitura
`carregar_tsp` e `carregar_cvrp` usadas no notebook.

Saídas:
- TSP : tsp_20.csv, tsp_50.csv, tsp_100.csv  (colunas: id, x, y)
- CVRP: cvrp_50.csv, cvrp_100.csv            (colunas: id, x, y, demand)
  • id=0 é o depósito (demand=0)
  • capacidade indicada numa linha de cabeçalho

Instâncias reais:
- Substituir por CSVs com as mesmas colunas.
- TSP : lista de coordenadas (id, x, y).
- CVRP: primeira linha = depósito (id=0); capacidade ajustável na linha
  “# capacity”. Ficheiros TSPLIB podem ser convertidos via NODE_COORD_SECTION.

Uso:
    python data/generate_instances.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

# Permite importar o modulo de algoritmos quando este script e corrido
# diretamente (adiciona a pasta-pai, raiz do projeto, ao sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_algoritmos as u  # noqa: E402

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta deste ficheiro
CVRP_CAPACITY = 50.0  # capacidade por defeito para as instancias de CVRP


def _write_tsp(n: int) -> None:
    """Gera ``n`` pontos de TSP e grava-os em ``data/tsp_{n}.csv``.
    Args:
        n: Numero de cidades a gerar.
    """
    coords = u.gerar_tsp(n, seed=u.RANDOM_SEED)  # coordenadas reprodutiveis
    path = os.path.join(DATA_DIR, f"tsp_{n}.csv")
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["id", "x", "y"])  # cabecalho
        for index, (x, y) in enumerate(coords):
            writer.writerow([index, f"{x:.4f}", f"{y:.4f}"])  # 4 casas decimais
    print(f"  escrito {path}")


def _write_cvrp(n: int) -> None:
    """Gera uma instância de CVRP (com deposito) e grava ``data/cvrp_{n}.csv``.
    O ficheiro comeca com uma linha "# capacity, <valor>", seguida do cabecalho
    de dados e de uma linha por no (a primeira e o deposito).
    Argumentos: n: Numero de clientes (sem contar o deposito).
    """
    instance = u.gerar_cvrp(n, capacity=CVRP_CAPACITY, seed=u.RANDOM_SEED)
    path = os.path.join(DATA_DIR, f"cvrp_{n}.csv")
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["# capacity", CVRP_CAPACITY])    # capacidade no topo
        writer.writerow(["id", "x", "y", "demand"])        # cabecalho de dados
        for index in range(len(instance.demands)):
            x, y = instance.coords[index]
            writer.writerow([index, f"{x:.4f}", f"{y:.4f}", int(instance.demands[index])])
    print(f"  escrito {path}")


def carregar_tsp(caminho: str) -> np.ndarray:
    """Carrega coordenadas (n x 2) de um CSV de TSP.
    Argumentos: caminho: Caminho para o ficheiro CSV com colunas id, x, y.
    Returns: Array NumPy (n x 2) com as coordenadas (x, y) de cada cidade.
    """
    points = []
    with open(caminho, encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            points.append([float(row["x"]), float(row["y"])])
    return np.array(points)


def carregar_cvrp(caminho: str) -> "u.CVRPInstance":
    """Carrega uma instancia de CVRP a partir de um CSV (deposito = id 0).
    Le a capacidade da linha "# capacity" se esta existir, em caso contrário usa
    o valor por defeito ``CVRP_CAPACITY``.
    Argumentos caminho: Caminho para o ficheiro CSV da instancia.
    Returns:
        Uma ``CVRPInstance`` pronta a usar (matriz de distancias incluida).
    """
    capacity = CVRP_CAPACITY  # valor por defeito
    coords, demands = [], []
    with open(caminho, encoding="utf-8") as file_handle:
        lines = file_handle.readlines()
    # Linha opcional de capacidade no topo do ficheiro.
    if lines[0].startswith("# capacity"):
        capacity = float(lines[0].strip().split(",")[1])
        lines = lines[1:]
    header = lines[0].strip().split(",")  # nomes das colunas de dados
    for line in lines[1:]:
        values = line.strip().split(",")
        record = dict(zip(header, values))
        coords.append([float(record["x"]), float(record["y"])])
        demands.append(float(record["demand"]))
    return u.CVRPInstance(coords=np.array(coords),
                          demands=np.array(demands), capacity=capacity)


def main() -> None:
    """Gera todas as instancias sinteticas (TSP 20/50/100 e CVRP 50/100)."""
    print("A gerar instancias sinteticas (seed=42)...")
    for n in (20, 50, 100):
        _write_tsp(n)
    for n in (50, 100):
        _write_cvrp(n)
    print("Concluido.")


if __name__ == "__main__":
    main()
