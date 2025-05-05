import os
import io
import csv
import math
import imageio
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from brian2 import *
from mpl_toolkits.mplot3d import Axes3D
import time
import networkx as nx
from scipy.signal import resample
from numpy.fft import rfft, rfftfreq

# Установка параметров вывода NumPy
np.set_printoptions(threshold=np.inf)

# -- Создание/проверка директории для сохранения результатов --
directory_path = 'results_ext_test1'
if os.path.exists(directory_path):
    print(f"Директория '{directory_path}' существует")
else:
    os.mkdir(directory_path)
    print(f"Директория '{directory_path}' создана")

directory_path2 = 'results_ext_test1/spikes'
if os.path.exists(directory_path2):
    print(f"Директория '{directory_path2}' существует")
else:
    os.mkdir(directory_path2)
    print(f"Директория '{directory_path2}' создана")


directory_path3 = 'results_ext_test1/rates'
if os.path.exists(directory_path3):
    print(f"Директория '{directory_path3}' существует")
else:
    os.mkdir(directory_path3)
    print(f"Директория '{directory_path3}' создана")

# -- Инициализация параметров сети --
n_neurons = 500
num_of_clusters = 2
cluster_sizes = [250, 250]

# Основные диапазоны параметров
p_within_values = np.arange(0.15, 0.16, 0.05)
p_input_values = np.arange(0.1, 0.21, 0.05)

num_tests = 10
rate_tick_step = 50
t_range = [0, 1000] # in ms
rate_range = [0, 200] # in Hz
max_rate_on_graph = 2
time_window_size = 1000  # in ms
refractory_period = 10*ms # -> max freq=100Hz
use_stdp_values = [False]

# Для осцилляций/стимулов
oscillation_frequencies = [10]
I0_values = [1000] # pA

measure_names = [
    "degree",  
    "betweenness",
    "closeness",
    "random",
    "eigenvector",
    "percolation",
    "harmonic",
]


# measure_names = [
#     "degree",
#     "random"
# ]

# boost_factor_list = [1.1, 2, 2, 0, 1.1, 2, 2]
boost_factor_list = [2, 100, 50, 0, 20, 50, 30]


simulation_times = [5000] # in ms


# Определение границ кластеров на основе размеров кластеров
def fram(cluster_sizes):
    frames = np.zeros(len(cluster_sizes))
    frames[0] = cluster_sizes[0]
    for i in range(1, len(cluster_sizes)):
        frames[i] = frames[i - 1] + cluster_sizes[i]
    return frames

# Проверка принадлежности узла к определенному кластеру
def clcheck(a, cluster_sizes):
    frames = fram(cluster_sizes)
    if a >= 0 and a < frames[0]:
        return 0
    else:
        for i in range(len(frames) - 1):
            if a >= frames[i] and a < frames[i + 1]:
                return i + 1
        return len(frames) - 1

# -- Формируем вектор меток кластеров --
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

def measure_connectivity(C, cluster_sizes):
    """
    Функция для вычисления фактических долей внутрикластерных (p_in_measured)
    и межкластерных (p_out_measured) связей по итоговой матрице C.
    """

    n_neurons = C.shape[0]
    labels = np.empty(n_neurons, dtype=int)
    start = 0
    for idx, size in enumerate(cluster_sizes):
        labels[start:start + size] = idx
        start += size

    intra_possible = 0
    intra_actual = 0
    inter_possible = 0
    inter_actual = 0

    # Перебор пар нейронов (i < j)
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if labels[i] == labels[j]:
                intra_possible += 1
                if C[i, j]:
                    intra_actual += 1
            else:
                inter_possible += 1
                if C[i, j]:
                    inter_actual += 1

    p_in_measured = intra_actual / intra_possible if intra_possible > 0 else 0
    p_out_measured = inter_actual / inter_possible if inter_possible > 0 else 0
    return p_in_measured, p_out_measured



def enforce_target_connectivity(G, cluster_sizes, p_intra, p_inter):
    """
    Корректирует граф G таким образом, чтобы фактические внутрикластерные (p_in) 
    и межкластерные (p_out) связи соответствовали целевым значениям p_intra и p_inter.
    """
    # Вычисляем число возможных внутрикластерных и межкластерных пар
    intra_possible = sum([size * (size-1) // 2 for size in cluster_sizes])
    inter_possible = 0
    for i in range(len(cluster_sizes)):
        for j in range(i + 1, len(cluster_sizes)):
            inter_possible += cluster_sizes[i] * cluster_sizes[j]
            
    # Целевые количества рёбер
    target_intra_count = round(p_intra * intra_possible)
    target_inter_count = round(p_inter * inter_possible)
    
    # G = nx.convert_node_labels_to_integers(G, first_label=0)  # <-- Исправление
    # Определяем метки кластеров (предполагается, что вершины пронумерованы последовательно)
    n_neurons = G.number_of_nodes()
    labels = np.empty(n_neurons, dtype=int)
    start = 0
    for idx, size in enumerate(cluster_sizes):
        labels[start:start + size] = idx
        start += size

    # Собираем списки уже существующих рёбер по типу связи
    intra_edges = []
    inter_edges = []
    for (u, v) in list(G.edges()):
        if labels[u] == labels[v]:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))
    
    current_intra = len(intra_edges)
    current_inter = len(inter_edges)
    
    # Если количество внутрикластерных рёбер превышает целевое, удаляем лишние
    if current_intra > target_intra_count:
        extra = current_intra - target_intra_count
        indices = np.random.choice(len(intra_edges), size=extra, replace=False)
        for idx in indices:
            edge = intra_edges[idx]
            if G.has_edge(*edge):
                G.remove_edge(*edge)
    
    # Аналогично для межкластерных рёбер
    if current_inter > target_inter_count:
        extra = current_inter - target_inter_count
        indices = np.random.choice(len(inter_edges), size=extra, replace=False)
        for idx in indices:
            edge = inter_edges[idx]
            if G.has_edge(*edge):
                G.remove_edge(*edge)
                
    return G

def generate_sbm_with_high_centrality(
    n, cluster_sizes, p_intra, p_inter,
    target_cluster_index, proportion_high_centrality=0.2,
    centrality_type="degree", boost_factor=2
):
    """
    Генерация модели SBM с повышенной центральностью для выбранного кластера.
    После процедуры повышения центральности проводится компенсация, 
    чтобы итоговые значения p_in и p_out оставались равными p_intra и p_inter.
    """
    G = nx.Graph()
    current_node = 0
    clusters = []

    # Формирование кластеров
    for size in cluster_sizes:
        cluster = list(range(current_node, current_node + size))
        clusters.append(cluster)
        current_node += size

    # Добавление внутрикластерных связей
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if np.random.rand() < p_intra:
                    G.add_edge(cluster[i], cluster[j])

    # Добавление межкластерных связей
    for idx1 in range(len(clusters)):
        for idx2 in range(idx1 + 1, len(clusters)):
            for u in clusters[idx1]:
                for v in clusters[idx2]:
                    if np.random.rand() < p_inter:
                        G.add_edge(u, v)
                        
    # Определение целевого кластера и выбор узлов для повышения центральности
    target_cluster = clusters[target_cluster_index]
    num_high_centrality_nodes = int(proportion_high_centrality * len(target_cluster))
    high_centrality_nodes = np.random.choice(target_cluster, size=num_high_centrality_nodes, replace=False)

    # Для центральностей, требующих вычисления глобальных метрик
    if centrality_type in ["eigenvector", "pagerank", "betweenness", "closeness", "harmonic"]:
        if centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        elif centrality_type == "pagerank":
            centrality = nx.pagerank(G)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        elif centrality_type == "harmonic":
            centrality = nx.harmonic_centrality(G)
        
        central_nodes = sorted(centrality, key=centrality.get, reverse=True)
        other_cluster_central_nodes = [node for node in central_nodes if node not in target_cluster][:int(0.1 * n)]

    # Механизм повышения центральности
    if centrality_type == "degree":
        for node in high_centrality_nodes:
            current_degree = G.degree[node]
            desired_degree = current_degree * boost_factor
            potential_neighbors = [v for v in G.nodes if v != node and not G.has_edge(node, v)]
            num_new_edges = int(min(desired_degree - current_degree, len(potential_neighbors)))
            if num_new_edges > 0:
                new_neighbors = np.random.choice(potential_neighbors, size=num_new_edges, replace=False)
                G.add_edges_from([(node, nn) for nn in new_neighbors])
                
    elif centrality_type in ["eigenvector", "pagerank", "betweenness", "closeness", "harmonic"]:
        for node in high_centrality_nodes:
            for neighbor in other_cluster_central_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < p_inter * boost_factor:
                    G.add_edge(node, neighbor)
                    
    elif centrality_type == "local_clustering":
        for node in high_centrality_nodes:
            neighbors = list(G.neighbors(node))
            np.random.shuffle(neighbors)
            current_cc = nx.clustering(G, node)
            max_edges = len(neighbors) * (len(neighbors) - 1) // 2
            current_edges = int(current_cc * max_edges)
            desired_edges = min(int(current_edges * boost_factor), max_edges)
            edges_needed = desired_edges - current_edges
            if edges_needed <= 0:
                continue
            non_edges = [(u, v) for i, u in enumerate(neighbors) for v in neighbors[i+1:] if not G.has_edge(u, v)]
            add_edges = min(edges_needed, len(non_edges))
            if add_edges > 0:
                selected_edges = np.random.choice(len(non_edges), size=add_edges, replace=False)
                G.add_edges_from([non_edges[idx] for idx in selected_edges])
                    
    elif centrality_type == "percolation":
        all_high_degree_nodes = sorted(G.nodes, key=lambda x: G.degree[x], reverse=True)[:int(0.1 * n)]
        for node in high_centrality_nodes:
            for neighbor in all_high_degree_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < p_inter * boost_factor:
                    G.add_edge(node, neighbor)
                    
    elif centrality_type == "cross_clique":
        cliques = list(nx.find_cliques(G))
        for node in high_centrality_nodes:
            other_cliques = [c for c in cliques if node not in c]
            for clique in other_cliques:
                candidates = [v for v in clique if not G.has_edge(node, v)]
                if candidates:
                    neighbor = np.random.choice(candidates)
                    if np.random.rand() < p_inter * boost_factor:
                        G.add_edge(node, neighbor)
    elif centrality_type == "random":
        pass
    else:
        raise ValueError("Unsupported centrality type.")

    # Шаг компенсации: корректировка сети для сохранения целевых значений p_in и p_out
    G = enforce_target_connectivity(G, cluster_sizes, p_intra, p_inter)
    
    C_total_matrix = nx.to_numpy_array(G)

    N_E = int(n * 80 / 100)
    N_I = n - N_E
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(n / 2)
    ext_neurons1 = np.arange(0, N_E_2) 
    inh_neurons1 = np.arange(N_E_2, N_E_2 + N_I_2) 
    ext_neurons2 = np.arange(N_2, N_E_2 + N_2)
    inh_neurons2 = np.arange(N_E_2 + N_2, n)  
    p_in_measured, p_out_measured = measure_connectivity(C_total_matrix, cluster_sizes)
    print(f"Фактическая p_intra: {p_in_measured:.3f}")
    print(f"Фактическая p_inter: {p_out_measured:.3f}")

    return G, p_in_measured, p_out_measured


class GraphCentrality:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def calculate_betweenness_centrality(self):
        return nx.betweenness_centrality(self.graph)

    def calculate_eigenvector_centrality(self):
        return nx.eigenvector_centrality(self.graph, max_iter=1000)

    def calculate_pagerank_centrality(self, alpha=0.85):
        return nx.pagerank(self.graph, alpha=alpha)

    def calculate_flow_coefficient(self):
        flow_coefficient = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                flow_coefficient[node] = 0.0
            else:
                edge_count = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if self.graph.has_edge(neighbors[i], neighbors[j]):
                            edge_count += 1
                flow_coefficient[node] = (2 * edge_count) / (len(neighbors) * (len(neighbors) - 1))
        return flow_coefficient

    def calculate_degree_centrality(self):
        return nx.degree_centrality(self.graph)

    def calculate_closeness_centrality(self):
        return nx.closeness_centrality(self.graph)

    def calculate_harmonic_centrality(self):
        return nx.harmonic_centrality(self.graph)

    def calculate_percolation_centrality(self, attribute=None):
        if attribute is None:
            attribute = {node: 1 for node in self.graph.nodes()}
        return nx.percolation_centrality(self.graph, states=attribute)

    def calculate_cross_clique_centrality(self):
        cross_clique_centrality = {}
        cliques = list(nx.find_cliques(self.graph))
        for node in self.graph.nodes():
            cross_clique_centrality[node] = sum(node in clique for clique in cliques)
        return cross_clique_centrality



def plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time, oscillation_frequency, use_stdp, measure_name):
    
    ax_spikes.scatter(spike_times, spike_indices, marker='|')
    step_size = time_window_size
    total_time_ms = sim_time / ms
    time_steps = np.arange(0, total_time_ms + step_size, step_size)
    for t in time_steps:
        ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

    ax_spikes.set_xlim([0,5000])
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Neuron index")
    ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms, {measure_name}", fontsize=16)
def plot_rates(ax_rates, N1, N2, rate_monitor, t_range):
    ax_rates.set_title(f'Num. of spikes\n neurons\n 0-{N1}', fontsize=16)
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label=f'Group 1 (0-{n_neurons/2})')
    ax_rates.set_xlim(t_range)
    ax_rates.set_ylim([0,500])
    ax_rates.set_xlabel("t [ms]")


def plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range):
    ax_rates2.set_title(f'Num. of spikes\n neurons\n {N1}-{N2}', fontsize=16)
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label=f'Group 2 ({n_neurons/2}-{n_neurons})')
    ax_rates2.set_xlim(t_range)
    ax_rates2.set_ylim([0,500])
    ax_rates2.set_xlabel("t [ms]")    

def plot_psd(rate_monitor, N1, N2, ax_psd):
    rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
    N = len(rate_monitor.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / ms)
        sampling_rate = 1000 / dt
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd.set_title(f"PSD neurons\n 0-{N1}", fontsize=16)
        ax_psd.plot(x, yn, c='k', label='Function')
        ax_psd.set_xlim([0,50])
        ax_psd.set_xlabel('Hz')


def plot_psd2(rate_monitor2, N1, N2, ax_psd2):
    rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
    N = len(rate_monitor2.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / ms) 
        sampling_rate = 1000 / dt 
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd2.set_title(f"PSD neurons\n {N1}-{N2}", fontsize=16)
        ax_psd2.plot(x, yn, c='k', label='Function')
        ax_psd2.set_xlim([0,50])
        ax_psd2.set_xlabel('Hz')

def plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram):
    dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / second)  # dt в секундах
    N_freq = len(rate_monitor.t)
    xf = rfftfreq(len(rate_monitor.t), d=dt)[:N_freq]
    ax_spectrogram.plot(xf, np.abs(rfft(rate_monitor.rate / Hz)), label=f'{0}-{N1} neurons')
    
    dt2 = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / second)
    N_freq2 = len(rate_monitor2.t)
    xf2 = rfftfreq(len(rate_monitor2.t), d=dt2)[:N_freq2]
    ax_spectrogram.plot(xf2, np.abs(rfft(rate_monitor2.rate / Hz)), label=f'{N1}-{N2} neurons')
    
    ax_spectrogram.set_xlim(0, 50)
    ax_spectrogram.legend()
    ax_spectrogram.set_title(f'Global\n Frequencies', fontsize=16)

def save_spike_data(spike_times, spike_indices, base_name,
                    precision: int = 3, skip_empty: bool = False):
    """
    Сохраняет список времён спайков для каждого нейрона в файл *.txt*.
    
    Parameters
    ----------
    spike_times : 1‑D array‑like
        Временные метки (мс) всех зафиксированных спайков.
    spike_indices : 1‑D array‑like (int)
        Индексы нейронов, сгенерировавших соответствующие spike_times.
    base_name : str
        Базовое имя файла без расширения (например, «sync_within0_35»).
    precision : int, optional
        Количество десятичных знаков при выводе (по умолчанию 3).
    skip_empty : bool, optional
        Если True, пропускает строки для нейронов без спайков.
    """
    spike_times  = np.asarray(spike_times,  dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=int)

    n_neurons = spike_indices.max() + 1
    spikes_by_neuron = [[] for _ in range(n_neurons)]

    # Группируем спайки по нейронам
    for t, i in zip(spike_times, spike_indices):
        spikes_by_neuron[i].append(float(t))

    # Формируем строки для записи
    lines = []
    fmt = f'%.{precision}f'
    for times in spikes_by_neuron:
        if not times and skip_empty:
            continue

        # --- форматируем КАЖДОЕ число ---
        tokens = []
        for t in sorted(times):
            s = fmt % t                 # '650.000'
            s = s.rstrip('0').rstrip('.')  # '650'
            if s == '':                 # случай, когда t == 0.000
                s = '0' 
            tokens.append(s)

        line = ' '.join(tokens)
        lines.append(line)

    # Записываем в файл
    fname = f'results_ext_test1/spikes/{base_name}_spikes.txt'
    with open(fname, 'w', encoding='utf‑8') as f:
        f.write('\n'.join(lines))
    print(f'[INFO] Спайковые данные сохранены в «{fname}»')

def plot_sync(spike_times, spike_indices,
              p_within, p_between, p_input,
              measure_name, test_num):
    """
    Сохраняет spike‑данные каждого нейрона в *.txt*‑файл.
    Файл получает префикс test<номер_теста>_… .
    """
    base_name = (
        f"{measure_name}_test{test_num+1}_"     # ← номер теста
        f"within{p_within:.2f}_"
        f"between{p_between:.2f}_"
        f"input{p_input:.2f}"
    ).replace('.', '_')

    save_spike_data(spike_times, spike_indices, base_name)

# def plot_connectivity(n_neurons, S_intra1, S_intra2, S_12, S_21, connectivity2, centrality, p_in_measured, p_out_measured, percent_central):


def print_centrality(C_total, N1, cluster_nodes, p_input, boost_factor, test_num, num_tests, measure_name, direct_1_2=True):
    """
    Вычисляет указанную меру центральности (measure_name), выводит список топ-узлов 
    по этой метрике (но только среди узлов одного кластера), а также возвращает этот список.

    Параметры:
    ----------
    C_total : numpy.ndarray
        Квадратная матрица смежности, описывающая граф.
    cluster_nodes : int или iterable
        Число узлов или список узлов (индексов), составляющих кластер, 
        по которым выбираем топ-узлы.
    p_input : float
        Доля от числа узлов кластера, которая берётся для формирования топ-листа.
        Если cluster_nodes - целое число, то используется именно это число для определения размера кластера.
    measure_name : str
        Название метрики центральности, которую нужно вычислить.
    direct_1_2 : bool, optional
        Если True, то анализ проводится для первой половины узлов кластера, иначе – для узлов с индексами 250-499.

    Возвращает:
    ----------
    list
        Список индексов узлов, являющихся топ-узлами по заданной метрике (только внутри выбранного кластера).
    """
    if measure_name != 'random':
        graph_centrality = GraphCentrality(C_total)
        measure_func_map = {
            "betweenness": graph_centrality.calculate_betweenness_centrality,
            "eigenvector": graph_centrality.calculate_eigenvector_centrality,
            "pagerank": graph_centrality.calculate_pagerank_centrality,
            "flow": graph_centrality.calculate_flow_coefficient,
            "degree": graph_centrality.calculate_degree_centrality,
            "closeness": graph_centrality.calculate_closeness_centrality,
            "harmonic": graph_centrality.calculate_harmonic_centrality,
            "percolation": graph_centrality.calculate_percolation_centrality,
            "cross_clique": graph_centrality.calculate_cross_clique_centrality
        }
        if measure_name not in measure_func_map:
            raise ValueError(
                f"Метрика '{measure_name}' не поддерживается. "
                f"Доступные метрики: {list(measure_func_map.keys())}."
            )
        measure_values = measure_func_map[measure_name]()
        for node in measure_values:
            measure_values[node] = round(measure_values[node], 5)

        cluster_list_1 = list(range(0, int(cluster_nodes/2)))
        cluster_list_2 = list(range(N1, cluster_nodes))

        # Формируем словарь значений метрики для выбранных узлов
        measure_values_cluster_1 = {
            node: measure_values[node] for node in cluster_list_1 if node in measure_values
        }
        measure_values_cluster_2 = {
            node: measure_values[node] for node in cluster_list_2 if node in measure_values
        }
        # Определяем количество топ-узлов для выборки
        top_k = int(p_input * len(cluster_list_1))
        sorted_neurons_cluster_1 = sorted(
            measure_values_cluster_1,
            key=lambda n: measure_values_cluster_1[n],
            reverse=True
        )
        sorted_neurons_cluster_2 = sorted(
            measure_values_cluster_2,
            key=lambda n: measure_values_cluster_2[n],
            reverse=True
        )
        top_neurons = sorted_neurons_cluster_1[:top_k]
        # print(measure_values_cluster)
        # print(sorted_neurons_cluster)
        # print(top_neurons)
        if test_num == num_tests-1:
            plot_centrality_by_neuron_number(measure_values_cluster_1, top_neurons, measure_name, boost_factor, top_percent=p_input*100)
    else:
        cluster_indices_1 = np.arange(0, int(n_neurons/2))
        cluster_indices_2 = np.arange(N1, cluster_nodes)
        num_chosen = int(p_input * len(cluster_indices_1))
        sorted_neurons_cluster_1 = cluster_indices_1
        sorted_neurons_cluster_2 = cluster_indices_2
        top_neurons = np.random.choice(cluster_indices_1, size=num_chosen, replace=False)

    return sorted_neurons_cluster_1, sorted_neurons_cluster_2, top_neurons


def plot_centrality_by_neuron_number(measure_values_cluster, top_neurons, measure_name, boost_factor, top_percent=10):
    """
    Строит график зависимости значения метрики центральности от номера нейрона.
    
    Параметры:
    -----------
    measure_values_cluster : dict
         Словарь, где ключ – номер нейрона, а значение – метрика центральности.
    top_neurons : list
         Список номеров нейронов, отобранных как топ по заданной метрике.
    top_percent : float, optional
         Процент для выделения порогового значения (по умолчанию 10%).
    """
    # Сортируем нейроны по их номерам для корректного отображения по оси x
    neurons = sorted(measure_values_cluster.keys())
    values = [measure_values_cluster[n] for n in neurons]
    
    plt.figure(figsize=(12, 6))
    # Линейный график, показывающий зависимость значения метрики от номера нейрона
    plt.plot(neurons, values, 'bo-', label='Значение метрики')
    plt.xlabel('Номер нейрона')
    plt.ylabel('Значение метрики центральности')
    plt.title(f'Зависимость {measure_name} от номера нейрона после увеличения boost_factor={boost_factor}')
    plt.grid(True)
    
    # Определим пороговое значение метрики для топ-нейронов.
    # Для этого сначала сортируем значения метрики по возрастанию.
    sorted_values = sorted(values)
    n = len(sorted_values)
    threshold_index = int(np.ceil((1 - top_percent/100) * n))
    threshold_value = sorted_values[threshold_index] if threshold_index < n else sorted_values[-1]
    
    # Отмечаем горизонтальной пунктирной линией пороговое значение метрики
    plt.axhline(threshold_value, color='green', linestyle='--', linewidth=2,
                label=f'Пороговая метрика (топ {top_percent}%) = {threshold_value:.3f}')
    
    # Выделяем на графике нейроны, входящие в топ (их номера берём из top_neurons)
    top_values = [measure_values_cluster[n] for n in top_neurons]
    plt.scatter(top_neurons, top_values, color='red', zorder=5, label='Топ-нейроны')
    
    plt.legend()
    plt.savefig(f"results_ext_test1/{measure_name}_{top_percent}centrality_plot_after_{boost_factor}.png")
    plt.close()

def save_csv_data(csv_filename, data_for_csv):
    """
    Сохранение данных в CSV
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for row in data_for_csv:
            writer.writerow(row)

def save_gif(images, filename, duration=3000, loop=0):
    """
    Создание анимации из последовательности изображений
    """
    imageio.mimsave(filename, images, duration=duration, loop=loop)





import random

def save_rate_monitors(rate_mon, rate_mon2, fname):
    """
    Сохраняет данные из PopulationRateMonitor в файл.
    Вместо самого объекта монитор сохраняются только массивы времени и частоты.
    """
    data = {}

    # Группа 1
    if rate_mon is not None:
        # Приводим к numpy‑массивам (ms и Hz)
        data["t_group1"] = np.array(rate_mon.t / ms)
        data["rate_group1"] = np.array(rate_mon.rate / Hz)
    else:
        data["t_group1"] = None
        data["rate_group1"] = None

    # Группа 2
    if rate_mon2 is not None:
        data["t_group2"] = np.array(rate_mon2.t / ms)
        data["rate_group2"] = np.array(rate_mon2.rate / Hz)
    else:
        data["t_group2"] = None
        data["rate_group2"] = None

    # Сохраняем только числовые данные
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    print(f"[INFO] Rate‑данные сохранены в «{fname}»")




def plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc):
    # Параметры мультитаперного спектрального анализа.
    # Увеличиваем длительность временного окна для более чёткого частотного разрешения.

    # n_time_samples = time_series_s.shape[1]
    # print('n_tine', n_time_samples)
    # t_full = np.arange(n_time_samples) / 10

    # fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # # ax[0].plot(t_full, time_series[trial_idx, :, 0], 'o', label="x1")
    # ax[0].plot(t_full, time_series_s[trial_idx, :, 0], label="0-249")
    # ax[0].set_ylabel("Амплитуда")
    # ax[0].legend()

    # # ax[1].plot(t_full, time_series[trial_idx, :, 1], 'o', label="x2", color="orange")
    # ax[1].plot(t_full, time_series_s[trial_idx, :, 1], label="250-499", color="orange")
    # ax[1].set_xlabel("Время (сек)")
    # ax[1].set_ylabel("Амплитуда")
    # ax[1].legend()

    time_halfbandwidth_product = 5
    time_window_duration = 3
    time_window_step = 0.1

    print("Начало счета Multitaper")
    from spectral_connectivity import Multitaper, Connectivity
    print(time_series_v.shape)
    m = Multitaper(
        time_series_v,
        sampling_frequency=100,
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
    )
    # Рассчитываем объект Connectivity
    c = Connectivity.from_multitaper(m)

    # =============================================================================
    # 3. Расчет мер направленной связи
    # =============================================================================
    granger = c.pairwise_spectral_granger_prediction()
    dtf = c.directed_transfer_function()
    pdc = c.partial_directed_coherence()

    # 4.1. Спектральная Грейнджерова причинность
    ax_granger[0].pcolormesh(c.time, c.frequencies, granger[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_granger[0].set_title("GC: x1 → x2")
    ax_granger[0].set_ylabel("Frequency (Hz)")
    ax_granger[1].pcolormesh(c.time, c.frequencies, granger[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_granger[1].set_title("GC: x2 → x1")
    ax_granger[1].set_xlabel("Time (s)")
    ax_granger[1].set_ylabel("Frequency (Hz)")

    # 4.2. directed transfer function
    ax_dtf[0].pcolormesh(c.time, c.frequencies, dtf[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_dtf[0].set_title("DTF: x1 → x2")
    ax_dtf[0].set_ylabel("Frequency (Hz)")
    ax_dtf[1].pcolormesh(c.time, c.frequencies, dtf[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_dtf[1].set_title("DTF: x2 → x1")
    ax_dtf[1].set_xlabel("Time (s)")
    ax_dtf[1].set_ylabel("Frequency (Hz)")

    ax_pdc[0].pcolormesh(c.time, c.frequencies, pdc[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_pdc[0].set_title("PDC: x1 → x2")
    ax_pdc[0].set_ylabel("Frequency (Hz)")
    ax_pdc[1].pcolormesh(c.time, c.frequencies, pdc[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_pdc[1].set_title("PDC: x2 → x1")
    ax_pdc[1].set_xlabel("Time (s)")
    ax_pdc[1].set_ylabel("Frequency (Hz)")




def sim(p_within, p_between, refractory_period, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes,
        I0_value, oscillation_frequency, use_stdp, time_window_size, measure_names, boost_factor_list, test_num, num_tests,
        C_total_prev=None, p_within_prev=None, p_between_prev=None,
        p_input=None, measure_name=None, measure_name_prev=None, centrality=None):
    """
    Выполняет один прогон симуляции при заданных параметрах.
    Если C_total_prev=None, генерируем матрицу "с нуля",
    иначе используем прошлую (чтобы копить STDP).
    """

    start_scope()

    start_time = time.time()

    if plotting_flags is None:
        plotting_flags = {}

    n_neurons = len(cluster_labels)
    target_cluster_index = 0
    proportion_high_centrality = p_input
    centrality_type = measure_name

    if measure_name in measure_names:
        boost_factor = boost_factor_list[measure_names.index(measure_name)]
    else:
        boost_factor = 0  # значение по умолчанию, если measure_name не найден
    
    print("Используемый boost_factor:", boost_factor)

    C_total, p_in_measured, p_out_measured = generate_sbm_with_high_centrality(
        n_neurons, cluster_sizes, p_within, p_between,
        target_cluster_index, proportion_high_centrality,
        centrality_type, boost_factor, 
    )    

    # -- Параметры LIF --
    N = n_neurons
    N1 = int(n_neurons/2)
    N2 = n_neurons
    N_E = int(N * 80 / 100)
    N_I = N - N_E
    R = 80 * Mohm
    C = 0.25 * nfarad
    tau = R*C # 20 ms
    max_rate = 1/tau 
    v_threshold = -50 * mV
    v_reset = -70 * mV
    v_rest = -65 * mV
    defaultclock.dt = 0.01 * second
    phi = 0
    f = 10 * Hz
    # -- Создаём группу нейронов --
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (v_rest - v + R*I_ext)/tau : volt
        I_ext = I0 * sin(2 * pi * f * t + phi) : amp
        I0 : amp
        ''',
        threshold="v > v_threshold",
        reset="v = v_reset",
        method="euler",
    )


    # Шаг 1. Вычисляем метрики и получаем список лучших узлов
    if p_input is None:
        p_input = 1.0

    if isinstance(C_total, np.ndarray):
        C_total_matrix = C_total
        C_total = nx.from_numpy_array(C_total)
    elif isinstance(C_total, nx.Graph):
        C_total = C_total
        C_total_matrix = nx.to_numpy_array(C_total)
    else:
        raise ValueError("data должен быть либо numpy-массивом, либо объектом networkx.Graph")


    all_centrality_1, all_centrality_2, centrality = print_centrality(C_total_matrix, N1, n_neurons, p_input, boost_factor, test_num, num_tests, measure_name=measure_name, direct_1_2=True)


    if isinstance(centrality, np.ndarray):
        centrality = centrality.tolist()
    centrality  = sorted(list(centrality))

    # # Назначение параметров модуляции для соответствующих временных интервалов:
    # if centrality:
    neurons.I0[centrality] = I0_value * pA

    percent_central = len(centrality) * 100 / N1
    
    print(measure_name)
    print(percent_central)
    # p_intra1 = p_within
    # p_intra2 = p_within
    # p_12     = p_between
    # p_21     = 0.02

    # # Веса соединения
    # w_intra1 = 1
    # w_intra2 = 1
    # w_12     = 1
    # w_21     = 1

    # n_half = n_neurons // 2


    input_rate = 1 * Hz
    input_group = PoissonGroup(n_neurons, rates=input_rate)
    syn_input = Synapses(input_group, neurons,model="w : volt", on_pre='v_post += w')
    syn_input.connect()
    syn_input.w = 1 * mV

    # S_intra1 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra1.connect(
    #     condition='i <= n_half and j <= n_half',
    #     p=p_intra1
    # )
    # S_intra1.w = w_intra1

    # # 2) Синапсы внутри 2-го кластера
    # S_intra2 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra2.connect(
    #     condition='i >= n_half and j >= n_half',
    #     p=p_intra2
    # )
    # S_intra2.w = w_intra2

    # # 3) Синапсы из 1-го кластера во 2-й
    # S_12 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_12.connect(
    #     condition='i < n_half and j >= n_half',
    #     p=p_12
    # )
    # S_12.w = w_12

    # # 4) Синапсы из 2-го кластера в 1-й
    # S_21 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_21.connect(
    #     condition='i >= n_half and j < n_half',
    #     p=p_21
    # )
    # S_21.w = w_21


    # STDP или нет
    if use_stdp:
        tau_stdp = 20 * ms
        Aplus = 0.005
        Aminus = 0.005

        stdp_eqs = '''
        dApre/dt = -Apre / tau_stdp : 1 (event-driven)
        dApost/dt = -Apost / tau_stdp : 1 (event-driven)
        w : volt
        '''
        exc_synapses = Synapses(
            neurons, neurons,
            model=stdp_eqs,
            on_pre="""
                v_post += w
                Apre += Aplus
                w = clip(w + Apost, 0, 1)
            """,
            on_post="""
                Apost += Aminus
                w = clip(w + Apre, 0, 1)
            """,
        )
        exc_synapses.w = 1 * mV
    else:
        exc_synapses = Synapses(neurons, neurons,
                                model="w : volt",
                                on_pre="v_post += w",
                                )

        
    inh_synapses = Synapses(neurons, neurons, 
                            model="w : volt", 
                            on_pre="v_post += - w", 
                            )
    
   
    # Генерация источников и целей
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(N / 2)
    # print("all_centrality", all_centrality_1)
    


   # Предполагается, что all_centrality_1 и all_centrality_2 уже определены,
    # а также параметры N_E, N_I и N установлены

    # Разбиение на возбуждающие и тормозные нейроны по мере центральности
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(N / 2)

    ext_neurons1 = sorted(all_centrality_1[:N_E_2])    # Возбуждающие нейроны (первая группа)
    inh_neurons1 = sorted(all_centrality_1[N_E_2:])    # Тормозные нейроны (первая группа)
    ext_neurons2 = sorted(all_centrality_2[:N_E_2])    # Возбуждающие нейроны (вторая группа)
    inh_neurons2 = sorted(all_centrality_2[N_E_2:])    # Тормозные нейроны (вторая группа)

    # Замена генерации непрерывных индексов на использование массивов с центральностью
    rows_exc1 = np.array(ext_neurons1)
    rows_inh1 = np.array(inh_neurons1)
    rows_exc2 = np.array(ext_neurons2)
    rows_inh2 = np.array(inh_neurons2)

    # Формирование масок для выбора строк матрицы соединений согласно выбранным нейронам
    mask_exc1 = np.isin(np.arange(C_total_matrix.shape[0]), rows_exc1)
    mask_inh1 = np.isin(np.arange(C_total_matrix.shape[0]), rows_inh1)
    mask_exc2 = np.isin(np.arange(C_total_matrix.shape[0]), rows_exc2)
    mask_inh2 = np.isin(np.arange(C_total_matrix.shape[0]), rows_inh2)

    # Извлечение индексов источников и целей для каждой группы
    sources_exc1, targets_exc1 = np.where(C_total_matrix[mask_exc1, :] > 0)
    sources_inh1, targets_inh1 = np.where(C_total_matrix[mask_inh1, :] > 0)
    sources_exc2, targets_exc2 = np.where(C_total_matrix[mask_exc2, :] > 0)
    sources_inh2, targets_inh2 = np.where(C_total_matrix[mask_inh2, :] > 0)

    # Преобразование локальных индексов в глобальные номера нейронов
    sources_exc1 = rows_exc1[sources_exc1]
    sources_inh1 = rows_inh1[sources_inh1]
    sources_exc2 = rows_exc2[sources_exc2]
    sources_inh2 = rows_inh2[sources_inh2]

    # Объединение источников и целей для соединения
    sources_exc = np.concatenate((sources_exc1, sources_exc2))
    targets_exc = np.concatenate((targets_exc1, targets_exc2))
    sources_inh = np.concatenate((sources_inh1, sources_inh2))
    targets_inh = np.concatenate((targets_inh1, targets_inh2))

    # Подключение синапсов с использованием корректных индексов
    exc_synapses.connect(i=sources_exc, j=targets_exc)
    inh_synapses.connect(i=sources_inh, j=targets_inh)
    # inh_synapses.connect(i=targets_inh, j=sources_inh)


    for idx in range(len(exc_synapses.i)):
        pre_neuron = exc_synapses.i[idx]
        post_neuron = exc_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            exc_synapses.w[idx] = 1 * mV  # для синапсов внутри первого кластера
        else:
            exc_synapses.w[idx] = 1 * mV  # для остальных связей
    
    for idx in range(len(inh_synapses.i)):
        pre_neuron = inh_synapses.i[idx]
        post_neuron = inh_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            inh_synapses.w[idx] = 1 * mV  # для синапсов внутри первого кластера
        else:
            inh_synapses.w[idx] = 1 * mV  # для остальных связей


    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:] / mV
    W[inh_synapses.i[:], inh_synapses.j[:]] = inh_synapses.w[:] / mV
    
    p_in_measured_syn, p_out_measured_syn = measure_connectivity(W, cluster_sizes)
    print(f"Фактическая p_intra после установки syn: {p_in_measured_syn:.3f}")
    print(f"Фактическая p_inter после установки syn: {p_out_measured_syn:.3f}")

    # Создание новой матрицы для отображения в градациях серого:
    # 0   -> отсутствие связи
    # 0.5 -> наличие связи
    # 1   -> наличие связи, если хотя бы один нейрон является центральным
    M = np.where(W != 0, 0.5, 0.0)


    # Мониторы  
    spike_monitor = SpikeMonitor(neurons)

    # rate_monitor = None
    # rate_monitor2 = None
    # if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
    #     rate_monitor = PopulationRateMonitor(neurons[:int(n_neurons/2)])
    # if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
    #     rate_monitor2 = PopulationRateMonitor(neurons[int(n_neurons/2):])
    rate_monitor = PopulationRateMonitor(neurons[:int(n_neurons/2)])
    rate_monitor2 = PopulationRateMonitor(neurons[int(n_neurons/2):])

   
    trace = StateMonitor(neurons, 'v', record=True)

    def calc_syn_count(N1, p_in, p_out, ext_neurons1, inh_neurons1, ext_neurons2, inh_neurons2):
        # Параметры кластера
        # print("ext_neurons1", ext_neurons1)
        # print("inh_neurons1", inh_neurons1)
        # print("ext_neurons2", ext_neurons2)
        # print("inh_neurons2", inh_neurons2)

        max_count = N1 * (N1-1)
        ext_n_count = max_count*0.8
        inh_n_count = max_count*0.2
        ext_syn_count_intra = ext_n_count * p_in
        ext_syn_count_inter = ext_n_count * p_out
        inh_syn_count_intra = inh_n_count * p_in
        inh_syn_count_inter = inh_n_count * p_out
        ext_syn_in_cluster = ext_syn_count_intra + ext_syn_count_inter
        inh_syn_in_cluster = inh_syn_count_intra + inh_syn_count_inter
        all_ext_syn = ext_syn_in_cluster * 2
        all_inh_syn = inh_syn_in_cluster * 2
        all_syn = all_ext_syn + all_inh_syn
        return all_ext_syn, all_inh_syn

    print(f"Количество нейронов: {N}")
    print(f"Желаемое кол-во возбуж. и тормозн. синапсов: {calc_syn_count(N1, p_in_measured_syn, p_out_measured_syn, ext_neurons1, inh_neurons1, ext_neurons2, inh_neurons2)}")
    print(f"Количество возбуждающих синапсов: {len(exc_synapses.i)}")
    print(f"Количество тормозных синапсов: {len(inh_synapses.i)}")
    avg_exc_synapses = len(exc_synapses.i)
    avg_inh_synapses = len(inh_synapses.i)
    # Запуск
    run(sim_time, profile=True)
    end_time = time.time()
    duration = end_time - start_time
    # print(f"Testing completed in {duration:.2f} seconds.")

    # Анализ спайков
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i

    trace_times = trace.t / ms
    
    mask1 = spike_indices < N1
    mask2 = spike_indices >= N1
    spike_times1 = spike_times[mask1]
    spike_times2 = spike_times[mask2]

    x1 = trace.v[:n_neurons//2, :] / mV  # (форма: n_neurons//2, 1000)
    x2 = trace.v[n_neurons//2:, :] / mV  # (форма: n_neurons//2, 1000)

    trial0 = x1.T  # (1000, n_neurons//2)
    trial1 = x2.T  # (1000, n_neurons//2)

    time_series_v = np.stack((trial0, trial1), axis=-1)

    print("Форма 3D тензора:", time_series_v.shape)

    bins = np.arange(0, int(sim_time/ms) + 1, time_window_size)
    time_window_centers = (bins[:-1] + bins[1:]) / 2

    avg_neuron_spikes_cluster2_list = []
    start_cluster_neuron = n_neurons/2
    end_cluster_neuron = n_neurons
    for i in range(len(bins) - 1):
        start_t = bins[i]
        end_t = bins[i + 1]

        mask = (
            (spike_indices >= start_cluster_neuron) & (spike_indices < end_cluster_neuron) &
            (spike_times > start_t) & (spike_times < end_t)
        )
        filtered_spike_indices = spike_indices[mask]
        group2_spikes = len(filtered_spike_indices)
        avg_spikes = group2_spikes / (end_cluster_neuron - start_cluster_neuron)
        avg_neuron_spikes_cluster2_list.append(avg_spikes)

    # Построение графиков
    if plotting_flags.get('granger', False) and 'ax_granger' in plotting_flags:
        ax_granger = plotting_flags['ax_granger']
        ax_dtf = plotting_flags['ax_dtf']
        ax_pdc = plotting_flags['ax_pdc']
        plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc)

        
    if plotting_flags.get('spikes', False) and 'ax_spikes' in plotting_flags:
        ax_spikes = plotting_flags['ax_spikes']
        plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time,
                    oscillation_frequency, use_stdp, measure_name)
    
    plot_sync(spike_times, spike_indices,
        p_within, p_between, p_input,
        measure_name, test_num)

        # ── после run(...) и перед анализом spike_monitor ───────────────────────────
    rate_file = (
        f"results_ext_test1/rates/rates_"
        f"within{p_within:.2f}_between{p_between:.2f}_"
        f"input{p_input:.2f}_{measure_name}_test{test_num+1}.pickle"
    )
    save_rate_monitors(rate_monitor, rate_monitor2, rate_file)

    if plotting_flags.get('rates', False) and 'ax_rates' in plotting_flags:
        ax_rates = plotting_flags['ax_rates']
        plot_rates(ax_rates, N1, N2, rate_monitor, t_range)

    if plotting_flags.get('rates2', False) and 'ax_rates2' in plotting_flags:
        ax_rates2 = plotting_flags['ax_rates2']
        plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range)

    if plotting_flags.get('psd', False) and 'ax_psd' in plotting_flags:
        ax_psd = plotting_flags['ax_psd']
        plot_psd(rate_monitor, N1, N2, ax_psd)

    if plotting_flags.get('psd2', False) and 'ax_psd2' in plotting_flags:
        ax_psd2 = plotting_flags['ax_psd2']
        plot_psd2(rate_monitor2, N1, N2, ax_psd2)

    if plotting_flags.get('spectrogram', False) and 'ax_spectrogram' in plotting_flags:
        ax_spectrogram = plotting_flags['ax_spectrogram']
        plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram)

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices, centrality, p_in_measured, p_out_measured, max_rate, W, M, percent_central, avg_exc_synapses, avg_inh_synapses

# Флаги для построения графиков
do_plot_granger = True
do_plot_spikes = True
do_plot_rates = True
do_plot_rates2 = True
do_plot_psd = True
do_plot_psd2 = True
do_plot_spectrogram = True

spike_counts_second_cluster = {}
detailed_spike_data_for_3d = {}
spike_counts_second_cluster_for_input = {}
subplot_results = {}  # Ключ: measure_name, значение: кортеж (exc_synapses, inh_synapses, centrality, p_in_measured, p_out_measured, percent_central)
centrality_results = {}


for current_time in simulation_times:
    sim_time = current_time * ms
    t_range = [0, current_time]

    # Подготовим структуры словарей
    for I0_value in I0_values:
        spike_counts_second_cluster[I0_value] = {}
        detailed_spike_data_for_3d[I0_value]  = {}
        spike_counts_second_cluster_for_input[I0_value] = {}

    # Циклы по частоте, I0, STDP
    for oscillation_frequency in oscillation_frequencies:
        for I0_value in I0_values:
            for use_stdp in use_stdp_values:
                print(f"\n### Запуск при I0={I0_value} пА, freq={oscillation_frequency} Гц, "
                        f"STDP={use_stdp}, Time={current_time} ms ###")

                # Создадим ключи верхнего уровня по measure_name И заодно по 'random'
                # (чтобы в дальнейшем вносить результаты для каждого measure_name отдельно)
                for measure_name in measure_names + ['random']:
                    centrality = None
                    spike_counts_second_cluster[I0_value].setdefault(measure_name, {})
                    detailed_spike_data_for_3d[I0_value].setdefault(measure_name, {})
                    spike_counts_second_cluster_for_input[I0_value].setdefault(measure_name, {})
                    # А внутри — под p_within
                    for p_within in p_within_values:
                        p_within_str = f"{p_within:.2f}"
                        spike_counts_second_cluster[I0_value][measure_name].setdefault(p_within_str, {})
                        detailed_spike_data_for_3d[I0_value][measure_name].setdefault(p_within_str, {})
                        spike_counts_second_cluster_for_input[I0_value][measure_name].setdefault(p_within_str, {})

                # --- Цикл по p_within ---
                for p_within in p_within_values:
                    p_within_str = f"{p_within:.2f}"

                    # --- Цикл по p_input ---
                    for p_input in p_input_values:
                        p_input = round(p_input, 2)
                        p_input_str = f"{p_input:.2f}"

                        # --- Цикл по measure_name ---
                        for measure_name in measure_names:
                            # Сначала запускаем симуляции для measure_name
                            centrality = None
                            measure_name_prev = None
                            C_total_prev = None
                            # Здесь будем накапливать GIF-кадры
                            images = []

                            # Список p_between
                            p_between_values = np.arange(0.01, p_within-0.04, 0.03)
                            # p_between_values = np.arange(0.05, p_within - 0.01, 0.05)
                            if len(p_between_values) == 0:
                                continue

                            for p_between in p_between_values:
                                p_between = round(p_between, 3)

                                # Создаём fig и gs, только если планируем что-то рисовать
                                fig = None
                                if any([do_plot_granger, do_plot_spikes, do_plot_rates, 
                                        do_plot_rates2, do_plot_psd, 
                                        do_plot_psd2, do_plot_spectrogram]):
                                    fig = plt.figure(figsize=(14, 12))
                                    
                                    # Настраиваем сетку подграфиков (пример на 3 строки x 6 столбцов)
                                    gs = fig.add_gridspec(ncols=6, nrows=3)

                                # Словарь с флагами для построения графиков
                                plotting_flags = {
                                    'granger': do_plot_granger,
                                    'spikes': do_plot_spikes,
                                    'rates': do_plot_rates,
                                    'rates2': do_plot_rates2,
                                    'psd': do_plot_psd,
                                    'psd2': do_plot_psd2,
                                    'spectrogram': do_plot_spectrogram,
                                }
                                # Если fig не None, создаём оси и пишем их в plotting_flags,
                                # чтобы внутри sim(...) можно было их получить
                                if fig is not None:
                                    

                                    if do_plot_spikes:
                                        ax_spikes = fig.add_subplot(gs[0, :])
                                        plotting_flags['ax_spikes'] = ax_spikes

                                    if do_plot_rates:
                                        ax_rates = fig.add_subplot(gs[1, 0])
                                        plotting_flags['ax_rates'] = ax_rates

                                    if do_plot_rates2:
                                        ax_rates2 = fig.add_subplot(gs[1, 1])
                                        plotting_flags['ax_rates2'] = ax_rates2

                                    if do_plot_psd:
                                        ax_psd = fig.add_subplot(gs[2, 0])
                                        plotting_flags['ax_psd'] = ax_psd

                                    if do_plot_psd2:
                                        ax_psd2 = fig.add_subplot(gs[2, 1])
                                        plotting_flags['ax_psd2'] = ax_psd2

                                    if do_plot_spectrogram:
                                        ax_spectrogram = fig.add_subplot(gs[2, 2])
                                        plotting_flags['ax_spectrogram'] = ax_spectrogram
                                        

                                    if do_plot_granger:
                                        ax_granger_1 = fig.add_subplot(gs[1, 3])
                                        ax_granger_2 = fig.add_subplot(gs[2, 3])
                                        ax_dtf_1 = fig.add_subplot(gs[1, 4])
                                        ax_dtf_2 = fig.add_subplot(gs[2, 4])
                                        ax_pdc_1 = fig.add_subplot(gs[1, 5])
                                        ax_pdc_2 = fig.add_subplot(gs[2, 5])
                                        # Сохраняем оси в словарь; внутри sim(...) вы будете делать:
                                        #   ax_granger = plotting_flags['ax_granger']
                                        #   ax_dtf     = plotting_flags['ax_dtf']
                                        #   ax_pdc     = plotting_flags['ax_pdc']
                                        plotting_flags['ax_granger'] = [ax_granger_1, ax_granger_2]
                                        plotting_flags['ax_dtf'] = [ax_dtf_1, ax_dtf_2]
                                        plotting_flags['ax_pdc'] = [ax_pdc_1, ax_pdc_2]

                                    # Присвоим заголовок для всей фигуры
                                    fig.suptitle(
                                        f'I0={I0_value} pA, p_input={p_input_str}, '
                                        f'p_within={p_within_str}, p_between={p_between}, '
                                        f'measure={measure_name}, Time={current_time} ms'
                                    )

                                # Список для средних спайков
                                avg_window_avg_neuron_spikes_cluster2_tests = []

                                for test_num in range(num_tests):
                                    print(measure_name)
                                    print(f'I0={I0_value} pA, p_within={p_within_str}, '
                                    f'p_input={p_input:.2f}, p_between={p_between}, '
                                    f'tест={test_num + 1}, Time={current_time} ms')
                                    # На последних тестах хотим видеть графики
                                    if test_num < num_tests - 1:
                                        # Отключаем рисование для экономии
                                        for key in [
                                            'granger','spikes','rates','rates2',
                                            'connectivity2','psd','psd2','spectrogram'
                                        ]:
                                            plotting_flags[key] = False
                                    else:
                                        # На последнем тесте всё включаем (или оставляем, как было)
                                        for key in [
                                            'granger','spikes','rates','rates2',
                                            'connectivity2','psd','psd2','spectrogram'
                                        ]:
                                            plotting_flags[key] = True

                                    # Запуск симуляции
                                    (avg_neuron_spikes_cluster2_list,
                                    time_window_centers,
                                    C_total,
                                    spike_indices,
                                    centrality,
                                    p_in_measured,
                                    p_out_measured, 
                                    max_rate, 
                                    W,
                                    M,
                                    percent_central,
                                    avg_exc_syn,
                                    avg_inh_syn
                                    ) = sim(
                                        p_within,
                                        p_between,
                                        refractory_period,
                                        sim_time,
                                        plotting_flags,
                                        rate_tick_step,
                                        t_range,
                                        rate_range,
                                        cluster_labels,
                                        cluster_sizes,
                                        I0_value,
                                        oscillation_frequency,
                                        use_stdp,
                                        time_window_size,
                                        measure_names,
                                        boost_factor_list,
                                        test_num,
                                        num_tests,
                                        C_total_prev,
                                        p_within_prev=p_within,
                                        p_between_prev=p_between,
                                        p_input=p_input,
                                        measure_name=measure_name,
                                        measure_name_prev=measure_name_prev,
                                        centrality=centrality
                                    )
                                    # Усредняем по всем окнам
                                    if avg_neuron_spikes_cluster2_list is not None and len(avg_neuron_spikes_cluster2_list) > 0:
                                        avg_window_val = np.mean(avg_neuron_spikes_cluster2_list)
                                    else:
                                        avg_window_val = 0
                                    avg_window_avg_neuron_spikes_cluster2_tests.append(avg_window_val)

                                    # Если это последний тест И условия по p_input и p_between выполняются, сохраняем результат
                        
                                    if test_num == num_tests - 1 and (p_input == 0.2 and p_between == 0.1):
                                        # Сохраняем необходимые переменные для построения субплотов
                                        subplot_results[measure_name] = W, M, p_in_measured, p_out_measured, percent_central
                                        key = (I0_value, oscillation_frequency, use_stdp, p_within_str, p_input_str, p_between, measure_name)
                                        centrality_results[key] = centrality
                                    # Обновляем «предыдущие» данные (для STDP)
                                    C_total_prev = C_total.copy()
                                    measure_name_prev = measure_name

                                # Запись данных для 3D-графика (Time vs p_between vs AvgSpikes)
                                if time_window_centers is None:
                                    time_window_centers = np.array([])
                                if avg_neuron_spikes_cluster2_list is None:
                                    avg_neuron_spikes_cluster2_list = []

                                detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_between] = {
                                    "time": time_window_centers.copy(),
                                    "spikes_list": avg_neuron_spikes_cluster2_list.copy()
                                }

                                # Считаем среднее по всем тестам
                                avg_window_avg_neuron_spikes_cluster2_tests = np.array(avg_window_avg_neuron_spikes_cluster2_tests)
                                mean_spikes = np.mean(avg_window_avg_neuron_spikes_cluster2_tests)

                                # Сохраняем в spike_counts_second_cluster
                                if p_between not in spike_counts_second_cluster[I0_value][measure_name][p_within_str]:
                                    spike_counts_second_cluster[I0_value][measure_name][p_within_str][p_between] = []
                                spike_counts_second_cluster[I0_value][measure_name][p_within_str][p_between].append(
                                    avg_window_avg_neuron_spikes_cluster2_tests
                                )
                                # Сохраняем в spike_counts_second_cluster_for_input
                                sc_input = spike_counts_second_cluster_for_input[I0_value][measure_name][p_within_str]
                                sc_input.setdefault(p_input_str, {})
                                sc_input[p_input_str][p_between] = {
                                    'mean_spikes': float(mean_spikes),
                                    'avg_exc_synapses': avg_exc_syn,   # добавляем значение возбуждающих синапсов
                                    'avg_inh_synapses': avg_inh_syn    # и значение тормозных синапсов
                                }
                                # Сохраняем кадр в GIF
                                if fig is not None:
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png')
                                    buf.seek(0)
                                    images.append(imageio.imread(buf))
                                    plt.close(fig)


                            # Конец цикла по p_between
                            if images:
                                gif_filename = (
                                    f'{directory_path}/gif_I0_{I0_value}freq_{oscillation_frequency}_'
                                    f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
                                    f'p_input_{p_input_str}_Time_{current_time}ms_{measure_name}.gif'
                                )
                                imageio.mimsave(gif_filename, images, duration=2000.0, loop=0)


                        # Конец цикла по measure_name, включая 'random'
                    # Конец цикла по p_input
                # Конец цикла по p_within
                
                with open('results_ext_test1/detailed_spike_data_for_3d.pickle', 'wb') as f:
                    pickle.dump(detailed_spike_data_for_3d, f)
                with open('results_ext_test1/spike_counts_second_cluster_for_input.pickle', 'wb') as f:
                    pickle.dump(spike_counts_second_cluster_for_input, f)
                with open('results_ext_test1/spike_counts_second_cluster.pickle', 'wb') as f:
                    pickle.dump(spike_counts_second_cluster, f)
                with open('results_ext_test1/subplot_results.pickle', 'wb') as f:
                    pickle.dump(subplot_results, f)

with open('results_ext_test1/centrality_results.pickle', 'wb') as f:
    pickle.dump(centrality_results, f)