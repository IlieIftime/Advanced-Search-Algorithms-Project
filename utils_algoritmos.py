"""
Biblioteca central do Projeto 2 com implementações "do zero" dos algoritmos
pedidos, sem uso de bibliotecas que resolvam automaticamente os problemas.
Conteúdo:
- N-Rainhas:
    * solve_csp_nqueens — CSP com backtracking + forward checking ou AC-3
    * ga_nqueens        — algoritmo genético com representação por permutação
    * tabu_nqueens      — Tabu Search (baseline do Projeto 1)
- TSP:
    * pso_tsp — PSO discreto (velocidade representada por swaps)
    * aco_tsp — Ant Colony Optimization (Ant System clássico)
- CVRP:
    * ga_cvrp — GA com representação giant-tour + split
Inclui ainda utilitários para fixação de sementes, registo de execuções,
cronómetro de contexto, geração de instâncias sintéticas e estatísticas
descritivas.
Mantém-se a notação clássica dos hiperparâmetros das meta-heurísticas
(alpha, beta, rho, tau, eta no ACO; w, c1, c2 no PSO) para consistência com
a literatura; restantes variáveis usam nomes descritivos.
"""

from __future__ import annotations
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


# Semente global usada por todo o projeto para garantir reprodutibilidade.
RANDOM_SEED: int = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa as sementes dos geradores aleatorios (``random`` e ``numpy``)."""
    random.seed(seed)
    np.random.seed(seed)


def log_seed(path: str, problema: str, metodo: str, instancia: str,
             run: int, seed: int) -> None:
    """Regista a semente usada numa execucao, anexando uma linha a um CSV.
    Cria o cabecalho automaticamente se o ficheiro ainda nao existir.
    Args:
        path, problema, metodo, instancia, run, seed 
    """
    file_exists = os.path.exists(path)  # decide se e preciso escrever cabecalho
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if not file_exists:
            writer.writerow(["problema", "metodo", "instancia", "run", "seed"])
        writer.writerow([problema, metodo, instancia, run, seed])


@dataclass
class Timer:
    """Cronometro de contexto baseado em ``time.perf_counter``.
    """

    elapsed: float = 0.0                       # tempo total medido (segundos)
    _start: float = field(default=0.0, repr=False)  # instante de inicio (oculto no repr)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start



# PROBLEMA 1 - N-RAINHAS
"""
Representação usada no problema das N-Rainhas: uma rainha por coluna.
O tabuleiro é um vetor `board` de tamanho N, onde `board[col]` indica a linha
da rainha colocada nessa coluna. Esta representação elimina conflitos de coluna
por construção, restando apenas conflitos de linha e de diagonal.
"""


def nqueens_conflicts(board: Sequence[int]) -> int:
    """
    Conta o número de pares de rainhas em conflito.

    A contagem é feita em O(N) verificando linhas e diagonais: para cada grupo com
    k rainhas alinhadas soma-se C(k,2) = k*(k-1)/2 conflitos. Uma solução válida
    tem zero conflitos.

    Args:
        board: Vetor onde board[col] indica a linha da rainha nessa coluna.

    Returns:
        Número total de pares de rainhas que se atacam (inteiro >= 0).
    """

    n = len(board)
    rows: Dict[int, int] = {}    # quantas rainhas em cada linha
    diag_down: Dict[int, int] = {}  # diagonais '\': constante = linha - coluna
    diag_up: Dict[int, int] = {}    # diagonais '/': constante = linha + coluna
    for col in range(n):
        row = board[col]
        rows[row] = rows.get(row, 0) + 1
        diag_down[row - col] = diag_down.get(row - col, 0) + 1
        diag_up[row + col] = diag_up.get(row + col, 0) + 1
    total = 0
    # Em cada grupo alinhado com k rainhas existem C(k, 2) pares em ataque.
    for group in (rows, diag_down, diag_up):
        for count in group.values():
            if count > 1:
                total += count * (count - 1) // 2
    return total



# 1A. CSP - backtracking com forward checking e AC-3

def _is_consistent(row: int, col: int, assignment: Dict[int, int]) -> bool:
    """Verifica se colocar uma rainha em (row, col) e compativel com a atribuicao.
    Argumentos: row, col, assignment
    Returns:
        ``True`` se nao houver ataque com nenhuma rainha ja colocada.
    """
    for placed_col, placed_row in assignment.items():
        if placed_row == row:                              # mesma linha
            return False
        if abs(placed_row - row) == abs(placed_col - col):  # mesma diagonal
            return False
    return True


def _ac3(domains: Dict[int, List[int]], n: int) -> bool:
    """Aplica consistencia de arcos (AC-3) sobre o grafo de restricoes binarias.
    Argsumentos:
        domains: Dicionario {coluna: lista de linhas possiveis}.
        n: Dimensao do tabuleiro.
    Returns:
        ``True`` se a rede ficar arco-consistente; ``False`` se for inconsistente.
    """
    # Fila inicial com todos os arcos dirigidos (Xi, Xj), Xi != Xj.
    queue: List[Tuple[int, int]] = [(xi, xj) for xi in range(n)
                                    for xj in range(n) if xi != xj]
    while queue:
        xi, xj = queue.pop(0)
        if _revise(domains, xi, xj):
            if not domains[xi]:
                return False  # dominio vazio => inconsistencia
            # Se Xi mudou, reavaliar todos os arcos que apontam para Xi.
            for xk in range(n):
                if xk != xi and xk != xj:
                    queue.append((xk, xi))
    return True


def _revise(domains: Dict[int, List[int]], xi: int, xj: int) -> bool:
    """Remove de ``domains[xi]`` os valores sem qualquer suporte em ``domains[xj]``.
    Returns:
        ``True`` se o dominio de ``xi`` foi alterado.
    """
    revised = False
    for value_i in list(domains[xi]):
        # value_i tem suporte se existir value_j compativel (as rainhas em
        # xi=value_i e xj=value_j nao se atacam: linhas e diagonais distintas).
        has_support = any(value_i != value_j and abs(value_i - value_j) != abs(xi - xj)
                          for value_j in domains[xj])
        if not has_support:
            domains[xi].remove(value_i)
            revised = True
    return revised


def solve_csp_nqueens(
    N: int,
    use_ac3: bool = False,
    time_limit: float = 60.0,
) -> Tuple[Optional[List[int]], int, float, bool]:
    """
    Resolve o problema das N-Rainhas como CSP usando backtracking com forward
    checking ou AC-3. As variáveis são as colunas (0..N-1), cada uma com domínio
    de linhas (0..N-1), e as restrições impedem rainhas na mesma linha ou diagonal.
    A seleção de variáveis usa MRV; o forward checking poda domínios ainda não
    atribuídos e, se `use_ac3=True`, aplica-se AC-3 após cada atribuição.
    Args:
        N: Dimensão do tabuleiro (número de rainhas).
        use_ac3: Se True, ativa AC-3; caso contrário usa apenas forward checking.
        time_limit: Tempo máximo de execução em segundos.
    Returns:
        (solution, nodes_expanded, time_elapsed, success)"""

    start = time.perf_counter()
    domains: Dict[int, List[int]] = {col: list(range(N)) for col in range(N)}
    assignment: Dict[int, int] = {}
    nodes_expanded = [0]   # lista (mutavel) para ser partilhada com o closure
    timed_out = [False]

    def backtrack(current_domains: Dict[int, List[int]]) -> bool:
        if time.perf_counter() - start > time_limit:
            timed_out[0] = True
            return False
        if len(assignment) == N:
            return True
        nodes_expanded[0] += 1
        # MRV: escolher a coluna nao atribuida com o menor dominio.
        free_columns = [col for col in range(N) if col not in assignment]
        col = min(free_columns, key=lambda c: len(current_domains[c]))
        for row in list(current_domains[col]):
            if _is_consistent(row, col, assignment):
                assignment[col] = row
                # Forward checking: copiar e podar os dominios das colunas livres.
                pruned_domains = {c: list(values) for c, values in current_domains.items()}
                pruned_domains[col] = [row]
                feasible = True
                for other_col in free_columns:
                    if other_col == col:
                        continue
                    pruned_domains[other_col] = [
                        r for r in pruned_domains[other_col]
                        if r != row and abs(r - row) != abs(other_col - col)
                    ]
                    if not pruned_domains[other_col]:
                        feasible = False
                        break
                if feasible and use_ac3:
                    feasible = _ac3(pruned_domains, N)
                if feasible and backtrack(pruned_domains):
                    return True
                del assignment[col]  # desfazer atribuicao (backtrack)
        return False

    solved = backtrack(domains)
    time_elapsed = time.perf_counter() - start
    if solved:
        solution = [assignment[col] for col in range(N)]
        return solution, nodes_expanded[0], time_elapsed, True
    # Falha tanto por esgotar a procura como por time-out (sucesso = False).
    return None, nodes_expanded[0], time_elapsed, False

#1B
""" 
Algoritmo Genético para N‑Rainhas com representação por permutação.

Cada cromossoma é uma permutação de 0..N‑1: o índice representa a coluna e o
valor a linha da rainha. Esta representação elimina conflitos de linha e de
coluna por construção, restando apenas conflitos de diagonal. O fitness é o
número de pares que não se atacam: C(N,2) − conflitos, que queremos maximizar
(ótimo = C(N,2)).
"""

def _fitness_nqueens(permutation: Sequence[int]) -> int:
    """Numero de pares de rainhas que NAO se atacam (a maximizar).
    """
    n = len(permutation)
    max_pairs = n * (n - 1) // 2
    return max_pairs - nqueens_conflicts(permutation)


def _ordered_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Order Crossover (OX1) preservando a validade da permutacao.
    Copia um segmento contiguo de ``parent1`` e completa pela ordem de ``parent2``.
    """
    n = len(parent1)
    cut_start, cut_end = sorted(random.sample(range(n), 2))
    child: List[Optional[int]] = [None] * n
    child[cut_start:cut_end + 1] = parent1[cut_start:cut_end + 1]
    used_genes = set(parent1[cut_start:cut_end + 1])
    fill_position = (cut_end + 1) % n
    # Percorrer parent2 a partir do fim do segmento, em ordem circular.
    for gene in parent2[cut_end + 1:] + parent2[:cut_end + 1]:
        if gene not in used_genes:
            child[fill_position] = gene
            used_genes.add(gene)
            fill_position = (fill_position + 1) % n
    return [int(value) for value in child]  # type: ignore[arg-type]


def _swap_mutation(permutation: List[int], rate: float) -> List[int]:
    """Mutacao por troca de dois genes com probabilidade ``rate`` (devolve copia)."""
    mutated = list(permutation)
    if random.random() < rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def _tournament(population: List[List[int]], fitness_values: List[int],
                k: int = 3) -> List[int]:
    """Selecao por torneio de tamanho ``k`` (devolve copia do vencedor)."""
    contenders = random.sample(range(len(population)), k)
    winner = max(contenders, key=lambda i: fitness_values[i])
    return list(population[winner])


def ga_nqueens(
    N: int,
    pop_size: int = 200,
    max_gen: int = 1000,
    time_limit: float = 60.0,
    taxa_mut: float = 0.2,
    elitismo: int = 2,
    tournament_k: int = 3,
) -> Tuple[List[int], List[int], float, bool]:
    """Algoritmo Genetico para N-Rainhas com representacao por permutacao.
    Usa selecao por torneio, Order Crossover (OX1), mutacao por troca e elitismo.
    Argumentos:
        N: Dimensao do tabuleiro.
        pop_size: Dimensao da populacao.
        max_gen: Numero maximo de geracoes.
        time_limit: Tempo maximo em segundos.
        taxa_mut: Probabilidade de mutacao por descendente.
        elitismo: Numero de melhores individuos copiados sem alteracao.
        tournament_k: Tamanho do torneio de selecao.
    Returns:
        Tuplo ``(best_solution, fitness_history, time_elapsed, success)`` onde
        ``fitness_history`` regista o melhor fitness por geracao.
    """
    start = time.perf_counter()
    optimum = N * (N - 1) // 2
    population = [random.sample(range(N), N) for _ in range(pop_size)]
    fitness_values = [_fitness_nqueens(individual) for individual in population]
    fitness_history: List[int] = []
    best_index = max(range(pop_size), key=lambda i: fitness_values[i])
    best_individual, best_fitness = list(population[best_index]), fitness_values[best_index]

    for _ in range(max_gen):
        fitness_history.append(best_fitness)
        if best_fitness == optimum or time.perf_counter() - start > time_limit:
            break
        # Elitismo: preservar os melhores individuos para a geracao seguinte.
        ranking = sorted(range(pop_size), key=lambda i: fitness_values[i], reverse=True)
        next_population = [list(population[ranking[e]]) for e in range(elitismo)]
        # Preencher o resto por selecao -> cruzamento -> mutacao.
        while len(next_population) < pop_size:
            parent1 = _tournament(population, fitness_values, tournament_k)
            parent2 = _tournament(population, fitness_values, tournament_k)
            child = _ordered_crossover(parent1, parent2)
            child = _swap_mutation(child, taxa_mut)
            next_population.append(child)
        population = next_population
        fitness_values = [_fitness_nqueens(individual) for individual in population]
        generation_best = max(range(pop_size), key=lambda i: fitness_values[i])
        if fitness_values[generation_best] > best_fitness:
            best_individual, best_fitness = list(population[generation_best]), fitness_values[generation_best]

    time_elapsed = time.perf_counter() - start
    return best_individual, fitness_history, time_elapsed, (best_fitness == optimum)



# 1C. Tabu Search - baseline de procura local (reimplementacao do Projeto 1)
def tabu_nqueens(
    N: int,
    max_iters: int = 1500,
    tabu_size: int = 20,
    no_improve_limit: int = 200,
    time_limit: float = 60.0,
) -> Tuple[List[int], int, float, bool]:
    """Tabu Search para N-Rainhas (baseline de procura local do Projeto 1).
    Args:
        N: Dimensao do tabuleiro.
        max_iters: Numero maximo de iteracoes.
        tabu_size: Tamanho da lista tabu.
        no_improve_limit: Iteracoes sem melhoria antes de restart.
        time_limit: Tempo maximo em segundos.
    Returns:
        Tuplo ``(best_board, iters, time_elapsed, success)``.
    """
    start = time.perf_counter()
    current = [random.randint(0, N - 1) for _ in range(N)]
    current_conflicts = nqueens_conflicts(current)
    best, best_conflicts = list(current), current_conflicts
    tabu_list: List[Tuple[int, int]] = []
    no_improve = 0

    for iteration in range(1, max_iters + 1):
        if best_conflicts == 0 or time.perf_counter() - start > time_limit:
            return best, iteration, time.perf_counter() - start, best_conflicts == 0
        best_move, best_move_conflicts = None, math.inf
        # Avaliar a vizinhanca: mover cada rainha para outra linha da sua coluna.
        for col in range(N):
            current_row = current[col]
            for row in range(N):
                if row == current_row:
                    continue
                neighbour = list(current)
                neighbour[col] = row
                conflicts = nqueens_conflicts(neighbour)
                move = (col, row)
                aspiration = conflicts < best_conflicts  # supera o melhor global?
                if (move not in tabu_list or aspiration) and conflicts < best_move_conflicts:
                    best_move, best_move_conflicts = move, conflicts
        if best_move is None:
            # Vizinhanca toda tabu e sem aspiracao -> restart aleatorio.
            current = [random.randint(0, N - 1) for _ in range(N)]
            current_conflicts = nqueens_conflicts(current)
            continue
        col, row = best_move
        current[col] = row
        current_conflicts = best_move_conflicts
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        if current_conflicts < best_conflicts:
            best, best_conflicts = list(current), current_conflicts
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= no_improve_limit:
            # Estagnacao prolongada -> restart suave.
            current = [random.randint(0, N - 1) for _ in range(N)]
            current_conflicts = nqueens_conflicts(current)
            no_improve = 0

    return best, max_iters, time.perf_counter() - start, best_conflicts == 0

# PROBLEMA 2 - TSP (Travelling Salesman Problem)

def route_length(route: Sequence[int], dist: np.ndarray) -> float:
    """Comprimento total do ciclo fechado definido por ``route``.
    Args:
        route: Permutacao das cidades (indices 0..n-1).
        dist: Matriz de distancias (n x n).
    Returns:
        Soma das distancias consecutivas mais o regresso a cidade inicial.
    """
    n = len(route)
    total = 0.0
    for i in range(n):
        total += dist[route[i], route[(i + 1) % n]]  # arco i -> i+1 (fecha o ciclo)
    return float(total)


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclidianas a partir de coordenadas (n x 2)."""
    # Diferencas vetorizadas: diff[i, j] = coords[i] - coords[j].
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


"""
2A : PSO discreto para o TSP, onde a posição de cada partícula é uma rota e a
velocidade é representada por uma sequência de swaps. A diferença entre duas
rotas (p_best − x) corresponde aos swaps necessários para transformar x em
p_best. A atualização combina inércia (w), componente cognitiva (c1, em direção
ao p_best) e componente social (c2, em direção ao g_best), aplicando cada swap
com probabilidade definida por esses coeficientes.
"""

def _swap_sequence(source: List[int], target: List[int]) -> List[Tuple[int, int]]:
    """Sequencia de swaps que transforma ``source`` em ``target``.
    Implementa a "subtracao de permutacoes" do PSO discreto: percorre as
    posicoes e troca elementos ate igualar o destino.
    """
    working = list(source)
    index_of = {value: i for i, value in enumerate(working)}  # posicao de cada cidade
    swaps: List[Tuple[int, int]] = []
    for i in range(len(working)):
        if working[i] != target[i]:
            j = index_of[target[i]]              # onde esta a cidade que devia ficar em i
            swaps.append((i, j))
            # Atualizar o indice e aplicar a troca.
            index_of[working[i]], index_of[working[j]] = j, i
            working[i], working[j] = working[j], working[i]
    return swaps


def _apply_swaps(permutation: List[int], swaps: List[Tuple[int, int]]) -> List[int]:
    """Aplica uma sequencia de swaps a uma permutacao (devolve copia)."""
    result = list(permutation)
    for i, j in swaps:
        result[i], result[j] = result[j], result[i]
    return result


def pso_tsp(
    dist_matrix: np.ndarray,
    num_particles: int = 30,
    max_iter: int = 200,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], List[float], float]:
    """
    PSO discreto para o TSP, onde cada partícula representa uma rota (permutação)
    e a velocidade é uma lista de swaps. A atualização combina três componentes:
    inércia (fração w dos swaps anteriores), componente cognitiva (c1, movendo a
    partícula em direção ao seu melhor pessoal) e componente social (c2, em direção
    ao melhor global). A velocidade é aplicada à rota e os melhores são atualizados.

    Args:
        dist_matrix: Matriz de distâncias (n x n).
        num_particles: Número de partículas do enxame.
        max_iter: Número máximo de iterações.
        params: Dicionário opcional com 'w', 'c1' e 'c2'.

    Returns:
        (best_route, history, time_elapsed): melhor rota encontrada, histórico
        do melhor custo por iteração e tempo total de execução.
    """

    coeffs = {"w": 0.5, "c1": 0.6, "c2": 0.8}  # coeficientes por defeito
    if params:
        coeffs.update(params)
    start = time.perf_counter()
    n = dist_matrix.shape[0]

    positions = [random.sample(range(n), n) for _ in range(num_particles)]
    velocities: List[List[Tuple[int, int]]] = [[] for _ in range(num_particles)]
    personal_best = [list(route) for route in positions]
    personal_best_cost = [route_length(route, dist_matrix) for route in positions]
    global_index = min(range(num_particles), key=lambda i: personal_best_cost[i])
    global_best, global_best_cost = list(personal_best[global_index]), personal_best_cost[global_index]
    history: List[float] = []

    for _ in range(max_iter):
        for i in range(num_particles):
            new_velocity: List[Tuple[int, int]] = []
            # Inercia: manter uma fracao dos swaps anteriores.
            for swap in velocities[i]:
                if random.random() < coeffs["w"]:
                    new_velocity.append(swap)
            # Componente cognitiva: aproximar do melhor pessoal.
            for swap in _swap_sequence(positions[i], personal_best[i]):
                if random.random() < coeffs["c1"]:
                    new_velocity.append(swap)
            # Componente social: aproximar do melhor global.
            for swap in _swap_sequence(positions[i], global_best):
                if random.random() < coeffs["c2"]:
                    new_velocity.append(swap)
            velocities[i] = new_velocity
            positions[i] = _apply_swaps(positions[i], new_velocity)
            cost = route_length(positions[i], dist_matrix)
            if cost < personal_best_cost[i]:
                personal_best[i], personal_best_cost[i] = list(positions[i]), cost
                if cost < global_best_cost:
                    global_best, global_best_cost = list(positions[i]), cost
        history.append(global_best_cost)

    return global_best, history, time.perf_counter() - start

"""
Ant Colony Optimization (Ant System) para o TSP.

Usa uma matriz de feromonas `tau` e visibilidade `eta = 1/d`. Cada formiga
constrói uma rota escolhendo a próxima cidade com probabilidade proporcional
a `tau^alpha * eta^beta`. No final de cada iteração as feromonas evaporam
(fator `rho`) e cada formiga deposita `Q / L_k` nos arcos da sua rota, onde
L_k é o comprimento da rota construída.
"""

def aco_tsp(
    dist_matrix: np.ndarray,
    num_ants: int = 25,
    max_iter: int = 100,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], List[float], float]:
    """
    Ant Colony Optimization (Ant System) para o TSP.
    Constrói rotas usando uma regra de transição probabilística baseada em
    feromona (`tau^alpha`) e visibilidade (`eta^beta`, com `eta = 1/dist`). Após
    cada iteração aplica-se evaporação (`rho`) e depósito de feromona proporcional
    a `Q / comprimento_da_rota`, incluindo reforço elitista da melhor rota.
    Args:
        dist_matrix: Matriz de distâncias (n x n).
        num_ants: Número de formigas por iteração.
        max_iter: Número de iterações.
        params: Dicionário opcional com 'alpha', 'beta', 'rho', 'Q'.
    Returns:
        (best_route, history, time_elapsed): melhor rota encontrada, histórico
        do melhor custo por iteração e tempo total de execução.
    """

    coeffs = {"alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 100.0}
    if params:
        coeffs.update(params)
    start = time.perf_counter()
    n = dist_matrix.shape[0]

    eta = 1.0 / (dist_matrix + np.eye(n))  # visibilidade; +I evita divisao por zero
    np.fill_diagonal(eta, 0.0)             # sem auto-arcos
    tau = np.ones((n, n))                  # feromona inicial uniforme
    best_route: List[int] = list(range(n))
    best_cost = math.inf
    history: List[float] = []

    for _ in range(max_iter):
        routes: List[List[int]] = []
        costs: List[float] = []
        for _ant in range(num_ants):
            start_city = random.randrange(n)
            visited = [start_city]
            unvisited = set(range(n)) - {start_city}
            current_city = start_city
            # Construcao da rota cidade a cidade.
            while unvisited:
                candidates = list(unvisited)
                weights = np.array(
                    [(tau[current_city, j] ** coeffs["alpha"]) * (eta[current_city, j] ** coeffs["beta"])
                     for j in candidates])
                weight_sum = weights.sum()
                if weight_sum <= 0:
                    next_city = random.choice(candidates)  # fallback uniforme
                else:
                    # Amostragem por roleta sobre a distribuicao acumulada.
                    next_city = candidates[int(np.searchsorted(
                        np.cumsum(weights / weight_sum), random.random()))]
                visited.append(next_city)
                unvisited.discard(next_city)
                current_city = next_city
            routes.append(visited)
            costs.append(route_length(visited, dist_matrix))
        # Atualizacao de feromonas: evaporacao seguida de deposito.
        tau *= (1.0 - coeffs["rho"])
        for route, cost in zip(routes, costs):
            deposit = coeffs["Q"] / cost
            for k in range(n):
                city_a, city_b = route[k], route[(k + 1) % n]
                tau[city_a, city_b] += deposit
                tau[city_b, city_a] += deposit
            if cost < best_cost:
                best_route, best_cost = list(route), cost
        # Reforco elitista da melhor rota encontrada ate agora.
        deposit_best = coeffs["Q"] / best_cost
        for k in range(n):
            city_a, city_b = best_route[k], best_route[(k + 1) % n]
            tau[city_a, city_b] += deposit_best
            tau[city_b, city_a] += deposit_best
        history.append(best_cost)

    return best_route, history, time.perf_counter() - start


"""
Problema 3: CVRP com representação giant‑tour. A instância inclui um depósito (índice 0),
clientes com procura e uma capacidade de veículo. Cada cromossoma é uma
permutação dos clientes; um operador de split divide essa sequência em rotas
viáveis, respeitando a capacidade. O custo é a soma das distâncias de todas as
rotas, cada uma começando e terminando no depósito.
"""

@dataclass
class CVRPInstance:
    """Instancia de CVRP.
    Attributes:
        coords: Coordenadas (n x 2); a linha 0 e o deposito.
        demands: Procuras por no (a do deposito e 0).
        capacity: Capacidade do veiculo.
        dist: Matriz de distancias (calculada automaticamente se omitida).
    """
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    dist: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dist is None:
            self.dist = euclidean_matrix(self.coords)

    @property
    def n_clientes(self) -> int:
        """Numero de clientes (exclui o deposito)."""
        return len(self.demands) - 1


def split_rotas(perm: Sequence[int], inst: CVRPInstance) -> List[List[int]]:
    """
    Divide uma permutação de clientes em rotas viáveis (split/repair). Percorre a
    sequência acumulando procura; sempre que adicionar o próximo cliente excederia
    a capacidade, fecha a rota atual e inicia outra. Garante viabilidade de
    capacidade por construção.

    Args:
        perm: Permutação dos clientes (1..n).
        inst: Instância de CVRP.

    Returns:
        Lista de rotas, cada uma contendo apenas os índices dos clientes (sem depósito).
    """

    routes: List[List[int]] = []
    current_route: List[int] = []
    load = 0.0
    for customer in perm:
        demand = inst.demands[customer]
        if load + demand > inst.capacity and current_route:
            routes.append(current_route)   # fechar rota cheia
            current_route, load = [], 0.0   # abrir nova rota
        current_route.append(int(customer))
        load += demand
    if current_route:
        routes.append(current_route)
    return routes


def custo_cvrp(rotas: List[List[int]], inst: CVRPInstance) -> float:
    """Distância total de um conjunto de rotas (cada rota fecha no deposito)."""
    total = 0.0
    for route in rotas:
        if not route:
            continue
        previous = 0  # deposito (indice 0)
        for customer in route:
            total += inst.dist[previous, customer]
            previous = customer
        total += inst.dist[previous, 0]  # regresso ao deposito
    return float(total)


def violacoes_capacidade(rotas: List[List[int]], inst: CVRPInstance) -> int:
    """Número de rotas que excedem a capacidade (0 quando se usa ``split_rotas``)."""
    violations = 0
    for route in rotas:
        if sum(inst.demands[customer] for customer in route) > inst.capacity:
            violations += 1
    return violations


def ga_cvrp(
    instance: CVRPInstance, pop_size: int = 150,
    max_gen: int = 400, capacity: Optional[float] = None,
    time_limit: float = 120.0, taxa_mut: float = 0.3,
    elitismo: int = 2, penalizacao: float = 1000.0,
) -> Tuple[Dict[str, object], Dict[str, float], float]:
    """
    Algoritmo Genético para o CVRP com representação giant‑tour + split.

    O cromossoma é uma permutação dos clientes; o operador `split_rotas` converte
    essa permutação num conjunto de rotas viáveis, respeitando a capacidade. O
    fitness corresponde à distância total (com eventual penalização, que é zero
    quando o split é aplicado). Usa seleção por torneio, Order Crossover (OX1) e
    mutação por troca/inversão. O número de veículos resulta do número de rotas.

    Args:
        instance: Instância de CVRP.
        pop_size: Tamanho da população.
        max_gen: Número máximo de gerações.
        capacity: Capacidade do veículo (se None, usa instance.capacity).
        time_limit: Tempo máximo de execução em segundos.
        taxa_mut: Probabilidade de mutação.
        elitismo: Número de indivíduos de elite preservados.
        penalizacao: Peso da penalização por violações de capacidade.

    Returns:
        (best_solution, metrics, time_elapsed), onde:
            best_solution — dicionário com 'perm' e 'rotas'
            metrics — inclui 'custo_total', 'veiculos_usados' e 'violacoes_capacidade'
            time_elapsed — tempo total de execução.
    """

    if capacity is not None:
        instance.capacity = capacity
    start = time.perf_counter()
    customers = list(range(1, instance.n_clientes + 1))

    def evaluate(perm: List[int]) -> float:
        """Custo (distancia total) + penalizacao por rotas inviaveis."""
        routes = split_rotas(perm, instance)
        return (custo_cvrp(routes, instance)
                + penalizacao * violacoes_capacidade(routes, instance))

    population = [random.sample(customers, len(customers)) for _ in range(pop_size)]
    costs = [evaluate(individual) for individual in population]
    best_index = min(range(pop_size), key=lambda i: costs[i])
    best_individual, best_cost = list(population[best_index]), costs[best_index]

    for _ in range(max_gen):
        if time.perf_counter() - start > time_limit:
            break
        ranking = sorted(range(pop_size), key=lambda i: costs[i])  # menor custo primeiro
        next_population = [list(population[ranking[e]]) for e in range(elitismo)]
        while len(next_population) < pop_size:
            # Torneio de tamanho 3 (menor custo vence).
            def tournament() -> List[int]:
                contenders = random.sample(range(pop_size), 3)
                return list(population[min(contenders, key=lambda i: costs[i])])
            child = _ordered_crossover(tournament(), tournament())
            if random.random() < taxa_mut:
                if random.random() < 0.5:
                    # Mutacao por troca de dois clientes.
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                else:
                    # Mutacao por inversao de um segmento.
                    i, j = sorted(random.sample(range(len(child)), 2))
                    child[i:j + 1] = reversed(child[i:j + 1])
            next_population.append(child)
        population = next_population
        costs = [evaluate(individual) for individual in population]
        generation_best = min(range(pop_size), key=lambda i: costs[i])
        if costs[generation_best] < best_cost:
            best_individual, best_cost = list(population[generation_best]), costs[generation_best]

    routes = split_rotas(best_individual, instance)
    metrics = {
        "custo_total": custo_cvrp(routes, instance),
        "veiculos_usados": len(routes),
        "violacoes_capacidade": violacoes_capacidade(routes, instance),
    }
    solution = {"perm": best_individual, "rotas": routes}
    return solution, metrics, time.perf_counter() - start


# GERACAO DE INSTANCIAS SINTETICAS (seed fixo)

def gerar_tsp(n_cidades: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Gera coordenadas 2D uniformes para uma instancia de TSP.
    Argumentos:
        n_cidades: Número de cidades.
        seed: Semente para reprodutibilidade.
    Returns:
        Array (n_cidades x 2) de coordenadas em [0, 100].
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n_cidades, 2))


def gerar_cvrp(n_clientes: int, capacity: float = 50.0,
               seed: int = RANDOM_SEED) -> CVRPInstance:
    """ Gera uma instância sintética de CVRP.
    O depósito é fixado no centro (50, 50) e as procuras dos clientes são inteiros
    entre 1 e 10.
    Args: n_clientes: Número de clientes (excluindo o depósito).
        capacity: Capacidade do veículo.
        seed: Semente para reprodutibilidade.
    Returns: Uma CVRPInstance pronta a usar. """

    rng = np.random.default_rng(seed)
    coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(n_clientes, 2))])
    demands = np.concatenate([[0.0], rng.integers(1, 11, size=n_clientes)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


# ============================================================================
# ESTATISTICA DESCRITIVA
# ============================================================================

def resumo_estatistico(valores: Sequence[float]) -> Dict[str, float]:
    """Média, mediana e desvio padrão (amostral) de uma amostra."""
    array = np.asarray(valores, dtype=float)
    return {
        "media": float(array.mean()),
        "mediana": float(np.median(array)),
        "desvio": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }
