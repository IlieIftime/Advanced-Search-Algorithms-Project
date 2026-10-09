"""
Resumo — reproduce_results.py

Regenera todos os resultados do Projeto 2, executando as experiências do notebook
e reescrevendo os ficheiros em results/:

  - nqueens_results.csv
  - tsp_results.csv
  - cvrp_results.csv
  - tsp_grid.csv
  - seeds.csv
  - figures/*.png

Protocolo (seed = 42 + run):
  • N-Rainhas: {CSP-FC, CSP-AC3, GA, Tabu} * {8,16,32,64} * 10 runs
  • TSP: {PSO, ACO} * {20,50,100} * 5 runs
  • CVRP: GA * {50,100} * 5 runs
  • Grid TSP: PSO (w,c1,c2) e ACO (beta,rho) sobre tsp_50

Determinismo:
  Todos os resultados são reproduzíveis com semente fixa, exceto GA para
  N-Rainhas com N≥32, limitado por tempo (8 s para N=32; 12 s para N=64).
  Nestes casos, gerações e conflitos variam com a máquina, mas as conclusões
  e taxas de sucesso mantêm-se.

Utilização:
  python reproduce_results.py
  python reproduce_results.py --verify
  python reproduce_results.py --no-figures
"""


from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica
import matplotlib.pyplot as plt

import utils_algoritmos as u
from utils_algoritmos import RANDOM_SEED
from data.generate_instances import carregar_tsp, carregar_cvrp

# caminhos
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figures")

# variáveis globais e outros dados
NQUEENS_SIZES = [8, 16, 32, 64]
NQUEENS_METHODS = ["CSP-FC", "CSP-AC3", "GA", "Tabu"]
NQUEENS_RUNS = 10

TSP_SIZES = [20, 50, 100]
TSP_METHODS = ["PSO", "ACO"]
TSP_RUNS = 5

CVRP_SIZES = [50, 100]
CVRP_RUNS = 5

# Parametros do GA das N-Rainhas (confirmados por reproducao exata em N<=16).
GA_NQUEENS_POP = 150
GA_NQUEENS_MAXGEN = 2000
# Orcamento de tempo por N (so vincula para N>=32; ver nota de determinismo).
GA_NQUEENS_TIME = {8: 60.0, 16: 60.0, 32: 8.0, 64: 12.0}

# Paleta usada nas figuras do notebook.
C_ORANGE = "#E07A5F"
C_BLUE = "#2E86AB"
C_GREEN = "#1B998B"
C_PURPLE = "#7B68A6"
METHOD_COLOR = {"CSP-FC": C_BLUE, "CSP-AC3": C_GREEN, "GA": C_ORANGE, "Tabu": C_PURPLE}


#  N-Rainhas
def run_nqueens(seeds_log: List[list]) -> pd.DataFrame:
    """Corre o bloco das N-Rainhas e devolve o DataFrame de resultados."""
    
    rows = []   # lista onde cada execução acrescenta um dicionário de resultados
    
    for N in NQUEENS_SIZES:                     # percorre tamanhos do tabuleiro
        for run in range(NQUEENS_RUNS):         # número de execuções por tamanho
            seed = RANDOM_SEED + run            # semente fixa por run
            
            for method in NQUEENS_METHODS:      # métodos: CSP-FC, CSP-AC3, GA, Tabu
                
                # registo da semente usada (para seeds.csv)
                seeds_log.append(["nqueens", method, f"N={N}", run, seed])
                u.set_seed(seed)                # fixa a semente para reprodutibilidade

                # --- CSP com Forward Checking ---
                if method == "CSP-FC":
                    sol, effort, dt, ok = u.solve_csp_nqueens(
                        N, use_ac3=False, time_limit=60
                    )
                    # conflitos = diagonais (linhas/colunas já garantidas pela representação)
                    conflicts = u.nqueens_conflicts(sol) if sol is not None else N

                # --- CSP com AC-3 ---
                elif method == "CSP-AC3":
                    sol, effort, dt, ok = u.solve_csp_nqueens(
                        N, use_ac3=True, time_limit=60
                    )
                    conflicts = u.nqueens_conflicts(sol) if sol is not None else N

                # --- Algoritmo Genético ---
                elif method == "GA":
                    best, hist, dt, ok = u.ga_nqueens(
                        N,
                        pop_size=GA_NQUEENS_POP,
                        max_gen=GA_NQUEENS_MAXGEN,
                        time_limit=GA_NQUEENS_TIME[N]
                    )
                    effort = len(hist)          # esforço = número de gerações
                    conflicts = u.nqueens_conflicts(best)

                # --- Tabu Search ---
                else:  # Tabu
                    board, effort, dt, ok = u.tabu_nqueens(N)
                    conflicts = u.nqueens_conflicts(board)

                # guarda resultados desta execução
                rows.append(dict(
                    N=N,
                    metodo=method,
                    run=run,
                    seed=seed,
                    tempo=dt,
                    nos_ou_iters=effort,        # nós (CSP) ou iterações/gerações (GA/Tabu)
                    conflitos=conflicts,
                    sucesso=int(bool(ok))       # 1 se encontrou solução válida
                ))

            # barra de progresso simples no terminal
            print(f"  [N-Rainhas] N={N:>2} run={run}  ok", end="\r")

    # limpa a linha final da barra de progresso
    print(" " * 60, end="\r")

    # devolve DataFrame com colunas ordenadas
    return pd.DataFrame(
        rows,
        columns=["N", "metodo", "run", "seed", "tempo",
                 "nos_ou_iters", "conflitos", "sucesso"]
    )


#  TSP
#  TSP
def run_tsp(seeds_log: List[list]) -> pd.DataFrame:
    """Corre o bloco do TSP (PSO e ACO) e devolve o DataFrame de resultados."""
    rows = []   # lista onde serão acumulados os resultados de cada execução

    for n in TSP_SIZES:   # percorre instâncias com 20, 50 e 100 cidades
        coords = carregar_tsp(os.path.join(DATA, f"tsp_{n}.csv"))  # coordenadas
        D = u.euclidean_matrix(coords)  # matriz de distâncias euclidianas

        for run in range(TSP_RUNS):     # 5 execuções por instância
            seed = RANDOM_SEED + run

            for algo in TSP_METHODS:    # PSO e ACO
                seeds_log.append(["tsp", algo, f"tsp_{n}", run, seed])
                u.set_seed(seed)        # garante reprodutibilidade

                if algo == "PSO":       # parâmetros default: 30 partículas, 200 iterações
                    route, _, _ = u.pso_tsp(D)
                else:                   # ACO: 25 formigas, 100 iterações
                    route, _, _ = u.aco_tsp(D)

                cost = u.route_length(route, D)  # custo da rota encontrada

                rows.append(dict(
                    n_cidades=n, algoritmo=algo, run=run, seed=seed,
                    custo=cost, tempo=0.0  # tempo não medido aqui
                ))

            print(f"  [TSP] n={n:>3} run={run}  ok", end="\r")

    print(" " * 60, end="\r")
    return pd.DataFrame(rows, columns=["n_cidades", "algoritmo", "run", "seed",
                                       "custo", "tempo"])


#  CVRP
# =========================================================================== CVRP
def run_cvrp(seeds_log: List[list]) -> pd.DataFrame:
    """Corre o bloco do CVRP (GA) e devolve o DataFrame de resultados."""
    rows = []

    for n in CVRP_SIZES:   # instâncias com 50 e 100 clientes
        inst = carregar_cvrp(os.path.join(DATA, f"cvrp_{n}.csv"))

        for run in range(CVRP_RUNS):   # 5 execuções por instância
            seed = RANDOM_SEED + run
            seeds_log.append(["cvrp", "GA", f"cvrp_{n}", run, seed])
            u.set_seed(seed)

            # GA com parâmetros fixos (150 indivíduos, 400 gerações, 60s)
            _, metrics, dt = u.ga_cvrp(inst, pop_size=150, max_gen=400, time_limit=60)

            rows.append(dict(
                n_clientes=n, algoritmo="GA", run=run, seed=seed,
                custo_total=metrics["custo_total"],
                veiculos_usados=metrics["veiculos_usados"],
                violacoes_capacidade=metrics["violacoes_capacidade"],
                tempo=dt
            ))

            print(f"  [CVRP] n={n:>3} run={run}  ok", end="\r")

    print(" " * 60, end="\r")
    return pd.DataFrame(rows, columns=["n_clientes", "algoritmo", "run", "seed",
                                       "custo_total", "veiculos_usados",
                                       "violacoes_capacidade", "tempo"])


# Grid TSP
# =========================================================================== Grid TSP
def run_grid() -> pd.DataFrame:
    """Pesquisa de parametros (grid) no TSP-50 com semente fixa 42."""
    D = u.euclidean_matrix(carregar_tsp(os.path.join(DATA, "tsp_50.csv")))
    rows = []

    # --- Grid PSO: combinações de w, c1, c2 ---
    for w in [0.3, 0.5, 0.7]:
        for c1, c2 in [(0.4, 0.9), (0.7, 0.7), (0.9, 0.4)]:
            u.set_seed(RANDOM_SEED)
            route, _, _ = u.pso_tsp(D, params={"w": w, "c1": c1, "c2": c2})
            rows.append(dict(
                algoritmo="PSO",
                param1=f"w={w}",
                param2=f"c1={c1},c2={c2}",
                custo=u.route_length(route, D),
                tempo=0.0
            ))

    # --- Grid ACO: combinações de beta e rho ---
    for beta in [2.0, 3.0, 5.0]:
        for rho in [0.1, 0.3, 0.5]:
            u.set_seed(RANDOM_SEED)
            route, _, _ = u.aco_tsp(D, params={"beta": beta, "rho": rho})
            rows.append(dict(
                algoritmo="ACO",
                param1=f"beta={beta}",
                param2=f"rho={rho}",
                custo=u.route_length(route, D),
                tempo=0.0
            ))

    return pd.DataFrame(rows, columns=["algoritmo", "param1", "param2", "custo", "tempo"])


#  Figuras
# =========================================================================== Figuras
def make_figures(nq: pd.DataFrame, tsp: pd.DataFrame, cvrp: pd.DataFrame,
                 grid: pd.DataFrame) -> None:
    """Regenera as figuras agregadas a partir dos resultados."""
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    # ---- N-Rainhas: agregação por método e N ----
    res = nq.groupby(["N", "metodo"]).agg(
        sucesso_pct=("sucesso", lambda s: 100 * s.mean()),
        tempo_med=("tempo", "mean"),
        esforco_med=("nos_ou_iters", "mean")
    ).reset_index()

    # 3 gráficos lado a lado: sucesso, tempo, esforço
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.2
    Ns = NQUEENS_SIZES
    x = np.arange(len(Ns))

    # --- barras de taxa de sucesso ---
    for k, m in enumerate(NQUEENS_METHODS):
        sub = res[res.metodo == m].set_index("N").reindex(Ns)
        axes[0].bar(x + (k - 1.5) * width, sub.sucesso_pct, width,
                    label=m, color=METHOD_COLOR[m])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(Ns)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Taxa de sucesso (%)")
    axes[0].set_title("Taxa de sucesso por N")
    axes[0].legend(fontsize=8)

    # --- tempo e esforço (escala log) ---
    for m in NQUEENS_METHODS:
        sub = res[res.metodo == m].set_index("N").reindex(Ns)
        axes[1].plot(Ns, sub.tempo_med, "o-", label=m, color=METHOD_COLOR[m])
        axes[2].plot(Ns, sub.esforco_med, "s--", label=m, color=METHOD_COLOR[m])

    for ax, ttl, yl in [
        (axes[1], "Tempo de execucao (escala log)", "Tempo medio (s, log)"),
        (axes[2], "Esforco de procura", "Nos expandidos / iteracoes (log)")
    ]:
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "nqueens_comparacao.png"), dpi=150)
    plt.close(fig)

    # ---- Boxplot do esforço em N=32 ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [nq[(nq.N == 32) & (nq.metodo == m)].nos_ou_iters.values
            for m in NQUEENS_METHODS]
    bp = ax.boxplot(data, labels=NQUEENS_METHODS, patch_artist=True)

    for patch, m in zip(bp["boxes"], NQUEENS_METHODS):
        patch.set_facecolor(METHOD_COLOR[m])
        patch.set_alpha(0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Nos expandidos / iteracoes (log)")
    ax.set_title("Distribuicao do esforco em N=32")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "nqueens_boxplot_N32.png"), dpi=150)
    plt.close(fig)

    # ---- TSP: custo médio + dispersão ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # média + desvio padrão
    for algo, color in [("PSO", C_ORANGE), ("ACO", C_BLUE)]:
        agg = tsp[tsp.algoritmo == algo].groupby("n_cidades").custo
        mean, std = agg.mean(), agg.std()
        axes[0].errorbar(TSP_SIZES, mean.reindex(TSP_SIZES),
                         yerr=std.reindex(TSP_SIZES),
                         fmt="o-", capsize=4, label=algo, color=color)

    axes[0].set_xlabel("Numero de cidades")
    axes[0].set_ylabel("Custo medio")
    axes[0].set_title("Qualidade da solucao (TSP)")
    axes[0].legend()

    # boxplots lado a lado
    box_data, positions, colors = [], [], []
    for j, n in enumerate(TSP_SIZES):
        for algo, color in [("ACO", C_BLUE), ("PSO", C_ORANGE)]:
            box_data.append(tsp[(tsp.n_cidades == n) & (tsp.algoritmo == algo)].custo.values)
            positions.append(j * 2.5 + (0 if algo == "ACO" else 0.8))
            colors.append(color)

    bp = axes[1].boxplot(box_data, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_xticks([j * 2.5 + 0.4 for j in range(len(TSP_SIZES))])
    axes[1].set_xticklabels(TSP_SIZES)
    axes[1].set_xlabel("Numero de cidades")
    axes[1].set_ylabel("Custo")
    axes[1].set_title("Dispersao ACO (azul) vs PSO (laranja)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tsp_comparacao.png"), dpi=150)
    plt.close(fig)

    # ---- TSP: melhor rota ACO em tsp_50 ----
    coords = carregar_tsp(os.path.join(DATA, "tsp_50.csv"))
    D = u.euclidean_matrix(coords)
    u.set_seed(RANDOM_SEED)
    route, _, _ = u.aco_tsp(D)

    fig, ax = plt.subplots(figsize=(6, 6))
    loop = route + [route[0]]  # fecha o ciclo
    ax.plot(coords[loop, 0], coords[loop, 1], "-", color=C_BLUE, lw=1.2)
    ax.scatter(coords[:, 0], coords[:, 1], c=C_ORANGE, s=30)
    ax.set_title(f"Melhor rota ACO (tsp_50) — custo {u.route_length(route, D):.1f}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tsp_rota_aco50.png"), dpi=150)
    plt.close(fig)

    # ---- Heatmap do grid ACO ----
    aco = grid[grid.algoritmo == "ACO"].copy()
    betas = ["beta=2.0", "beta=3.0", "beta=5.0"]
    rhos = ["rho=0.1", "rho=0.3", "rho=0.5"]
    M = np.zeros((len(betas), len(rhos)))

    for i, b in enumerate(betas):
        for j, r in enumerate(rhos):
            M[i, j] = aco[(aco.param1 == b) & (aco.param2 == r)].custo.values[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels(rhos)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(betas)

    for i in range(len(betas)):
        for j in range(len(rhos)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="w", fontsize=9)

    fig.colorbar(im, ax=ax, label="custo")
    ax.set_title("Grid ACO (beta x rho) — tsp_50")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tsp_grid_aco.png"), dpi=150)
    plt.close(fig)

    # ---- CVRP: custo médio e veículos usados ----
    res = cvrp.groupby("n_clientes").agg(
        custo_med=("custo_total", "mean"),
        custo_dp=("custo_total", "std"),
        veiculos=("veiculos_usados", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar([str(n) for n in res.n_clientes], res.custo_med,
                yerr=res.custo_dp, capsize=5, color=C_BLUE, alpha=0.8)
    axes[0].set_xlabel("Numero de clientes")
    axes[0].set_ylabel("Custo total medio")
    axes[0].set_title("Custo da solucao (CVRP)")

    axes[1].bar([str(n) for n in res.n_clientes], res.veiculos,
                color=C_ORANGE, alpha=0.8)
    axes[1].set_xlabel("Numero de clientes")
    axes[1].set_ylabel("Veiculos usados (medio)")
    axes[1].set_title("Veiculos por instancia")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "cvrp_comparacao.png"), dpi=150)
    plt.close(fig)

    # ---- CVRP: rotas da melhor solução em cvrp_50 ----
    inst = carregar_cvrp(os.path.join(DATA, "cvrp_50.csv"))
    u.set_seed(RANDOM_SEED)
    sol, _, _ = u.ga_cvrp(inst, pop_size=150, max_gen=400, time_limit=60)

    # extrai rotas (dependendo da estrutura devolvida)
    routes = sol["rotas"] if isinstance(sol, dict) and "rotas" in sol \
             else u.split_rotas(sol["perm"], inst)

    coords = inst.coords
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 1)))

    for k, route in enumerate(routes):
        path = [0] + list(route) + [0]  # começa e termina no depósito
        ax.plot(coords[path, 0], coords[path, 1], "-o",
                color=cmap[k], ms=4, lw=1)

    ax.scatter(coords[0, 0], coords[0, 1], c="black", marker="s",
               s=90, label="deposito", zorder=3)
    ax.set_title("Rotas CVRP-50 (cor = veiculo)")
    ax.legend()
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "cvrp_rotas50.png"), dpi=150)
    plt.close(fig)

#  Verificacão
def verify_against_committed(nq, tsp, cvrp, grid) -> None:
    """Compara os resultados regenerados com os CSV ja existentes em results/."""
    print("\n== Verificacao contra os CSV entregues ==")

    # Função auxiliar: compara colunas numéricas entre DataFrames
    def cmp_numeric(name, new, old, key_cols, val_cols, tol=1e-4, skip=None):
        old = old.copy()  # evita alterar o original
        # junta new e old pelas colunas-chave
        merged = new.merge(old, on=key_cols, suffixes=("_new", "_old"))
        bad = 0  # contador de divergências

        for vc in val_cols:
            # diferença absoluta entre valores novos e antigos
            diff = (merged[f"{vc}_new"] - merged[f"{vc}_old"]).abs()
            # divergência significativa = maior que tolerância relativa
            mask = diff > tol * merged[f"{vc}_old"].abs().clip(lower=1)

            # permite ignorar casos especiais (ex.: GA N>=32)
            if skip is not None:
                mask = mask & ~skip(merged)

            bad += int(mask.sum())

        status = "OK (identico)" if bad == 0 else f"{bad} divergencias"
        print(f"  {name:<22} {status}")

    # --- Comparação N-Rainhas ---
    try:
        cmp_numeric(
            "nqueens (det.)", nq,
            pd.read_csv(os.path.join(RESULTS, "nqueens_results.csv")),
            ["N", "metodo", "run"],
            ["nos_ou_iters", "conflitos", "sucesso"],
            skip=lambda m: (m["metodo"] == "GA") & (m["N"] >= 32)  # exceção documentada
        )
        print("    (GA N>=32 excluido: limitado por tempo de relogio)")
    except FileNotFoundError:
        print("  nqueens_results.csv ausente — nada a comparar")

    # --- Comparação TSP ---
    try:
        cmp_numeric(
            "tsp", tsp,
            pd.read_csv(os.path.join(RESULTS, "tsp_results.csv")),
            ["n_cidades", "algoritmo", "run"],
            ["custo"]
        )
    except FileNotFoundError:
        print("  tsp_results.csv ausente")

    # --- Comparação CVRP ---
    try:
        cmp_numeric(
            "cvrp", cvrp,
            pd.read_csv(os.path.join(RESULTS, "cvrp_results.csv")),
            ["n_clientes", "run"],
            ["custo_total", "veiculos_usados", "violacoes_capacidade"]
        )
    except FileNotFoundError:
        print("  cvrp_results.csv ausente")

    # --- Comparação Grid TSP ---
    try:
        cmp_numeric(
            "tsp_grid", grid,
            pd.read_csv(os.path.join(RESULTS, "tsp_grid.csv")),
            ["algoritmo", "param1", "param2"],
            ["custo"]
        )
    except FileNotFoundError:
        print("  tsp_grid.csv ausente")


# main pipeline 
def main() -> int:
    # parser de argumentos da linha de comandos
    ap = argparse.ArgumentParser(description="Regenera os resultados do Projeto 2.")
    ap.add_argument("--verify", action="store_true",
                    help="compara os resultados regenerados com os CSV atuais")
    ap.add_argument("--no-figures", action="store_true",
                    help="nao gerar figuras")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    seeds_log: List[list] = []  # registo de todas as sementes usadas

    # --- Execução dos blocos experimentais ---
    print("==> N-Rainhas (4 metodos x 4 tamanhos x 10 runs)...")
    nq = run_nqueens(seeds_log)

    print("==> TSP (PSO/ACO x 3 instancias x 5 runs)...")
    tsp = run_tsp(seeds_log)

    print("==> CVRP (GA x 2 instancias x 5 runs)...")
    cvrp = run_cvrp(seeds_log)

    print("==> Grid de parametros (TSP-50)...")
    grid = run_grid()

    # --- Escrita dos CSV de resultados ---
    nq.to_csv(os.path.join(RESULTS, "nqueens_results.csv"), index=False)
    tsp.to_csv(os.path.join(RESULTS, "tsp_results.csv"), index=False)
    cvrp.to_csv(os.path.join(RESULTS, "cvrp_results.csv"), index=False)
    grid.to_csv(os.path.join(RESULTS, "tsp_grid.csv"), index=False)

    # CSV das sementes usadas
    with open(os.path.join(RESULTS, "seeds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problema", "metodo", "instancia", "run", "seed"])
        w.writerows(seeds_log)

    print("==> CSV escritos em results/.")

    # --- Geração das figuras (opcional) ---
    if not args.no_figures:
        print("==> A gerar figuras...")
        make_figures(nq, tsp, cvrp, grid)
        print("==> Figuras escritas em results/figures/.")

    # --- Verificação contra os CSV entregues ---
    if args.verify:
        verify_against_committed(nq, tsp, cvrp, grid)

    print("\nConcluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

