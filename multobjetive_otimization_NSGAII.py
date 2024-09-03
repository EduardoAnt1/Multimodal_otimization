import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from platypus import NSGAII, Problem, Real
from sklearn.metrics import pairwise_distances

# Definindo a função g(x) para o MMF11_I
def g(x, n_p):
    return 2 - np.exp(-2 * np.log(2) * (((x - 0.1) / 0.8) ** 2)) * np.sin(np.pi * n_p * x) ** 6

# Definindo a função para o MMF11_I
def mmf11_i(vars):
    x1 = vars[0]
    x2 = vars[1]
    n_p = 2  # Defina o valor apropriado de np_ aqui
    f1 = x1
    f2 = g(x2, n_p) / x1
    return [f1, f2]

# Funções para calcular IGDX e IGDF
def igdx(solutions):
    pareto_front = np.unique(np.array(solutions), axis=0)
    dist_matrix = pairwise_distances(pareto_front, pareto_front)
    igdx_value = np.mean(np.max(dist_matrix, axis=1))
    return igdx_value

def igdf(solutions):
    pareto_front = np.unique(np.array(solutions), axis=0)
    dist_matrix = pairwise_distances(pareto_front, pareto_front)
    igdf_value = np.mean(np.min(dist_matrix + np.eye(len(dist_matrix)) * 1e10, axis=1))
    return igdf_value

# Função para calcular a hipervolume
def hv(solutions):
    return 1 - np.mean([np.max(s) for s in solutions])

# Função para calcular o spread
def psp(solutions):
    return np.mean([np.max(s) - np.min(s) for s in solutions])

# Definindo as curvas de f2 para as soluções global e locais
def f2_curve_global(x1, np_):
    return g(1 / (2 * np_), np_) / x1

def f2_curve_local(x1, np_, i):
    return g(1 / (2 * np_) + (i - 1) / np_, np_) / x1

# Função para calcular estatísticas
def calculate_statistics(values):
    return {
        'Melhor': np.min(values),
        'Pior': np.max(values),
        'Média': np.mean(values),
        'Mediana': np.median(values),
        'Desvio Padrão': np.std(values)
    }

# Definindo pesos para os indicadores
weights = {
    '1/PSP': 0.05,
    'HV': 0.05,
    '1/IGDX': 0.4,
    '1/IGDF': 0.5
}

# Configurando o problema com Platypus
problem = Problem(2, 2)
problem.types[:] = [Real(0.1, 1.1), Real(0, 1)]
problem.function = mmf11_i

gen_ranges = [100, 1000, 10000]

for num_generations in gen_ranges:
    # Armazenando resultados
    results = []

    # Parâmetros do NSGA-II
    pop_size = 200
    mutation = 0.2

    # Executando o algoritmo com diferentes seeds
    for seed in range(21):
        np.random.seed(seed)
        algorithm = NSGAII(problem, population_size=pop_size, mutation_rate=mutation, seed=seed)
        
        # Executando o algoritmo
        algorithm.run(num_generations)
        
        # Extraindo soluções e variáveis
        solutions = np.array([s.objectives for s in algorithm.result])
        variables = np.array([s.variables for s in algorithm.result])
        
        # Calculando indicadores
        igdx_value = igdx(solutions)
        igdf_value = igdf(solutions)
        hv_value = hv(solutions)
        psp_value = psp(solutions)
        
        # Salvando os resultados
        results.append({
            'Seed': seed,
            'IGDX': igdx_value,
            'IGDF': igdf_value,
            'HV': hv_value,
            'PSP': psp_value,
            'Score': (weights['1/PSP'] * (1 / psp_value) +
                    weights['HV'] * hv_value +
                    weights['1/IGDX'] * (1 / igdx_value) +
                    weights['1/IGDF'] * (1 / igdf_value))
        })

    # Criando uma tabela com os resultados
    df_results = pd.DataFrame(results)

    # Calculando estatísticas para os scores
    score_statistics = calculate_statistics(df_results['Score'])

    # Criando DataFrame com estatísticas
    df_statistics = pd.DataFrame([score_statistics])

    # Salvando os resultados em arquivos CSV
    df_results.to_csv('indicadores_execucoes_' + str(num_generations) + '.csv', index=False)
    df_statistics.to_csv('estatisticas_indicadores_' + str(num_generations) + '.csv', index=False)

    # O restante do código para salvar gráficos e encontrar a melhor execução permanece inalterado.

    # Executando o algoritmo NSGA-II novamente para capturar a população inicial e a população final
    algorithm_initial = NSGAII(problem, population_size=pop_size, mutation_rate=mutation)
    algorithm_initial.run(1)  # Executa apenas uma geração para obter a população inicial

    # Extraindo a população inicial
    initial_solutions = np.array([s.objectives for s in algorithm_initial.result])
    initial_variables = np.array([s.variables for s in algorithm_initial.result])

    # Extraindo a população final
    final_solutions = np.array([s.objectives for s in algorithm.result])
    final_variables = np.array([s.variables for s in algorithm.result])

    # Calculando a fronteira de Pareto total
    all_solutions = np.concatenate([initial_solutions, final_solutions])
    all_solutions = np.unique(all_solutions, axis=0)

    # Salvando gráficos da população inicial
    plt.figure(figsize=(12, 6))

    # Gráfico da fronteira de Pareto da população inicial
    plt.subplot(1, 2, 1)
    plt.scatter(initial_solutions[:, 0], initial_solutions[:, 1], c='green', marker='o', label='Fronteira de Pareto Inicial')
    plt.title("Fronteira de Pareto Inicial")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()

    # Gráfico da distribuição da população inicial
    plt.subplot(1, 2, 2)
    plt.scatter(initial_variables[:, 0], initial_variables[:, 1], c='purple', marker='x', label='População Inicial')
    plt.title("Distribuição da População Inicial")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('pareto_e_distribuicao_populacao_inicial_' + str(num_generations) + '.png', bbox_inches='tight', dpi=300)  # Salvando com alta qualidade
    plt.close()

    # Gerando valores para a curva global f2
    x1_values = np.linspace(0.1, 1.1, 500)
    f2_values_global = f2_curve_global(x1_values, np_=2)
    
    # Gerando valores para a curva local f2 (para diferentes valores de i)
    f2_values_local_1 = f2_curve_local(x1_values, np_=2, i=2)


    # Gerando valores para a curva global de x1 x x2
    x1_ps = np.linspace(0.1, 1.1, 500)
    x2_global = 1 / (2 * 2)  # Exemplo com np = 2
    x2_local_1 = 1 / (2 * 2) + (2 - 1) / 2

    # Criando uma única curva combinada para x1 e x2
    plt.figure(figsize=(12, 6))

    # Gráfico da fronteira de Pareto da população final
    plt.subplot(1, 2, 2)
    plt.scatter(final_solutions[:, 0], final_solutions[:, 1], c='yellow', marker='*', label='Fronteira de PFs')
    plt.plot(x1_values, f2_values_global, color='red', linestyle='--', label='Curva PF Global')
    plt.plot(x1_values, f2_values_local_1, color='blue', linestyle='--', label='Curva PF Local')
    plt.title("Fronteira de Pfs")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()

    # Gráfico da distribuição da população final
    plt.subplot(1, 2, 1)
    plt.scatter(final_variables[:, 0], final_variables[:, 1], c='green', marker='*', label='PSs')
    plt.plot(x1_ps, x2_global * np.ones(len(x1_ps)), color='red', linestyle='--', label='Curva PS Global')
    plt.plot(x1_ps, x2_local_1 * np.ones(len(x1_ps)), color='blue', linestyle='--', label='Curva PS Local')
    plt.title("Distribuição da PSs")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('pareto_e_distribuicao_populacao_final_' + str(num_generations) + '.png', bbox_inches='tight', dpi=300)  # Salvando com alta qualidade
    plt.close()
