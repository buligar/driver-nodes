import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.colors as mcolors
import networkx as nx
import pyspike as spk
from pathlib import Path
import gzip, pickle, glob, re, os
from scipy.interpolate import griddata           # для интерполяции
from numpy.fft import rfft, rfftfreq


def plot_3d_spike_data(
    detailed_spike_data_for_3d,
    measure_names,
    p_within_values,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate,
    directory_path='results_ext_test1',
):
    """
    Для каждого measure_name строится свой 3D-график (при фиксированном p_within).
    """
    os.makedirs(directory_path, exist_ok=True)
    

    if I0_value not in detailed_spike_data_for_3d:
        print(f"Нет данных для I0={I0_value}pA.")
        return
    
    # Перебираем, например, все метрики, включая "random"
    for measure_name in measure_names:
        if measure_name not in detailed_spike_data_for_3d[I0_value]:
            print(f"Нет данных для measure_name={measure_name} при I0={I0_value}. Пропуск.")
            continue
        
        for p_within in p_within_values:
            p_within_str = f"{p_within:.2f}"
            if p_within_str not in detailed_spike_data_for_3d[I0_value][measure_name]:
                print(f"Нет данных для p_within={p_within_str}. Пропуск.")
                continue

            p_between_list = sorted(detailed_spike_data_for_3d[I0_value][measure_name][p_within_str].keys())
            if len(p_between_list) == 0:
                print(f"Нет p_between для p_within={p_within_str}. Пропуск.")
                continue

            sample_p_between = p_between_list[0]
            time_array = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][sample_p_between].get("time", np.array([]))

            if time_array.size == 0:
                print(f"Нет временных данных для measure_name={measure_name}, p_within={p_within_str}, p_between={sample_p_between}.")
                continue

            # Проверяем согласованность временных окон
            consistent_time = True
            for p_btw in p_between_list:
                current_time_array = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_btw].get("time", np.array([]))
                if current_time_array.size == 0 or len(current_time_array) != len(time_array):
                    consistent_time = False
                    print(f"Несоответствие временных окон для measure_name={measure_name}, p_within={p_within_str}, p_between={p_btw}.")
                    break
            if not consistent_time:
                continue

            # Создаём сетки для поверхности
            Time, P_between = np.meshgrid(time_array, p_between_list)
            Z = np.zeros(Time.shape)

            # Заполняем Z
            for i, p_btw in enumerate(p_between_list):
                spikes_arr = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_btw].get("spikes_list", [])
                if not spikes_arr:
                    Z[i, :] = 0
                else:
                    spikes_arr = np.array(spikes_arr)
                    if len(spikes_arr) < Z.shape[1]:
                        spikes_arr = np.pad(spikes_arr, (0, Z.shape[1] - len(spikes_arr)), 'constant')
                    elif len(spikes_arr) > Z.shape[1]:
                        spikes_arr = spikes_arr[:Z.shape[1]]
                    Z[i, :] = spikes_arr
            
            if not np.isfinite(Z).all():
                print(f"Некоторые значения в Z не являются конечными для measure_name={measure_name}, p_within={p_within_str}.")
                continue

            # Построение 3D
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            surf = ax_3d.plot_surface(
                Time,        # X
                P_between,   # Y
                Z,           # Z
                cmap='viridis',
                edgecolor='none'
            )
            ax_3d.set_xlabel('Time [ms]')
            ax_3d.set_ylabel('p_between')
            ax_3d.set_zlabel('Avg Spikes (Hz)')
            ax_3d.set_zlim(0,max_rate)
            ax_3d.set_title(
                f'3D: Time vs p_between vs Avg Spikes\n'
                f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
                f'measure={measure_name}, p_within={p_within_str}, Time={current_time}ms',
                fontsize=14
            )
            fig_3d.colorbar(surf, shrink=0.5, aspect=5)

            fig_filename_3d = os.path.join(
                directory_path,
                f'3D_plot_I0_{I0_value}pA_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_'
                f'measure_{measure_name}_p_within_{p_within_str}_Time_{current_time}ms.png'
            )
            plt.savefig(fig_filename_3d)
            plt.close(fig_3d)


def plot_pinput_between_avg_spikes_with_std(
    spike_counts_second_cluster_for_input,
    spike_counts_second_cluster,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate,
    p_input_values,
    p_between_values,
    directory_path='results_ext_test1',
    measure_names=None,
):
    """
    Для каждой метрики (кроме 'random') строит 3D-график с наложением поверхности для 'random'.
    
    Структура хранения данных:
      spike_counts_second_cluster_for_input[I0_value][measure_name][p_within][p_input][p_between]
          = { 'mean_spikes': <среднее число спайков>,
              'avg_exc_synapses': <среднее число возбуждающих синапсов>,
              'avg_inh_synapses': <среднее число тормозных синапсов> }
      spike_counts_second_cluster[I0_value][measure_name][p_within][p_between] = список,
          где каждая запись — массив средних значений, полученных в различных тестах.
    
    По итогам работы функция сохраняет результаты в CSV‑файл с дополнительными столбцами,
    а также формирует 3D-графики для каждой пары (measure_name, p_within).
    """
    import csv
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Создание директории для сохранения результатов
    os.makedirs(directory_path, exist_ok=True)
    
    # Вывод отладочной информации о структурах данных
    print("spike_counts_second_cluster_for_input:", spike_counts_second_cluster_for_input)
    print("spike_counts_second_cluster:", spike_counts_second_cluster)
    
    # Формирование имени CSV-файла с учётом времени симуляции
    avg_csv_filename = os.path.join(directory_path, f"avg_tests_avg_spikes_{current_time}ms.csv")
    
    # Открытие CSV-файла для записи результатов
    with open(avg_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Заголовок CSV-файла включает дополнительные столбцы для синаптических параметров
        writer.writerow([
            'I0_value', 'measure_name', 'p_within', 'p_input', 'p_between',
            'mean_spikes', 'std_spikes', 'avg_exc_synapses', 'avg_inh_synapses'
        ])
        
        # Проверка наличия данных для данного I0_value
        if I0_value not in spike_counts_second_cluster_for_input:
            print(f"Нет данных spike_counts_second_cluster_for_input для I0={I0_value}. Пропуск.")
            return
        
        # Определение списка метрик для анализа
        measure_names_in_data = list(spike_counts_second_cluster_for_input[I0_value].keys())
        if measure_names is None:
            measure_names = measure_names_in_data

        # Перебор метрик, за исключением 'random' (которая используется для контрольного сравнения)
        for measure_name in measure_names:
            if measure_name == 'random':
                continue
            if measure_name not in measure_names_in_data:
                print(f"measure_name={measure_name} не найден в данных. Пропуск.")
                continue
            if 'random' not in measure_names_in_data:
                print(f"Данные для 'random' не найдены. Пропуск объединённого графика для measure={measure_name}.")
                continue

            # Извлечение данных для текущей метрики и контрольной (random)
            data_measure = spike_counts_second_cluster_for_input[I0_value][measure_name]
            data_random = spike_counts_second_cluster_for_input[I0_value]['random']

            # Перебор всех значений p_within (строковая запись, например "0.15")
            for p_within_str in data_measure.keys():
                if p_within_str not in data_random:
                    print(f"Для p_within={p_within_str} нет данных в 'random'. Пропуск.")
                    continue

                dict_pinput_measure = data_measure[p_within_str]
                dict_pinput_random = data_random[p_within_str]

                # Отсортированные списки значений p_input (ключей словарей)
                sorted_p_inputs_measure = sorted(dict_pinput_measure.keys(), key=float)
                sorted_p_inputs_random = sorted(dict_pinput_random.keys(), key=float)
                common_p_inputs = sorted(set(sorted_p_inputs_measure).intersection(sorted_p_inputs_random), key=float)
                if not common_p_inputs:
                    print(f"Нет пересечения p_input для measure_name={measure_name}, p_within={p_within_str}. Пропуск.")
                    continue

                # Определение пересечения значений p_between
                first_p_input = common_p_inputs[0]
                p_between_list_measure = dict_pinput_measure[first_p_input].keys()
                p_between_list_random = dict_pinput_random[first_p_input].keys()
                common_p_between = sorted(set(p_between_list_measure).intersection(p_between_list_random), key=float)
                if not common_p_between:
                    print(f"Нет пересечения p_between для measure_name={measure_name}, p_within={p_within_str}. Пропуск.")
                    continue

                # Формирование осей для построения матриц
                p_input_list_float = [float(p) for p in common_p_inputs]
                p_between_list_float = [float(pb) for pb in common_p_between]
                Z_measure = np.zeros((len(p_input_list_float), len(p_between_list_float)))
                Z_random = np.zeros_like(Z_measure)
                Z_measure_std = np.zeros_like(Z_measure)
                Z_random_std = np.zeros_like(Z_measure)
                
                # Извлечение списка тестовых данных для вычисления стандартного отклонения
                cluster_data_measure = spike_counts_second_cluster.get(I0_value, {}).get(measure_name, {}).get(p_within_str, {})
                cluster_data_random = spike_counts_second_cluster.get(I0_value, {}).get('random', {}).get(p_within_str, {})

                # Перебор по значениям p_input и p_between для заполнения матриц и записи CSV
                for i, p_inp_str in enumerate(common_p_inputs):
                    p_btw_dict_measure = dict_pinput_measure[p_inp_str]
                    p_btw_dict_random  = dict_pinput_random[p_inp_str]
                    for j, p_btw in enumerate(p_between_list_float):
                        # Формирование строкового представления p_between
                        p_btw_str = f"{p_btw:.2f}"
                        
                        # Извлечение числовых значений среднего числа спайков
                        # Ожидается, что p_btw_dict_* хранит словарь с ключами 'mean_spikes', 'avg_exc_synapses', 'avg_inh_synapses'
                        data_entry_measure = p_btw_dict_measure.get(p_btw, {
                            'mean_spikes': 0.0, 'avg_exc_synapses': 0.0, 'avg_inh_synapses': 0.0
                        })
                        data_entry_random = p_btw_dict_random.get(p_btw, {
                            'mean_spikes': 0.0, 'avg_exc_synapses': 0.0, 'avg_inh_synapses': 0.0
                        })
                        mean_spikes_measure_val = float(data_entry_measure.get('mean_spikes', 0.0))
                        mean_spikes_random_val = float(data_entry_random.get('mean_spikes', 0.0))
                        
                        # Заполнение матриц для построения поверхностей
                        Z_measure[i, j] = mean_spikes_measure_val
                        Z_random[i, j] = mean_spikes_random_val

                        # Вычисление стандартного отклонения для measure (на основе данных тестовых повторов)
                        list_of_arrays_measure = cluster_data_measure.get(p_btw, [])
                        stdev_value_measure = 0.0
                        if list_of_arrays_measure:
                            num_pinputs = len(dict_pinput_measure.keys())
                            n_rep = len(list_of_arrays_measure) // num_pinputs if num_pinputs > 0 else 1
                            index_to_use = (i + 1) * n_rep - 1 if n_rep > 0 else 0
                            try:
                                trial_array = np.array(list_of_arrays_measure[index_to_use], dtype=float)
                                stdev_value_measure = np.std(trial_array)
                            except Exception as e:
                                print(e)
                        Z_measure_std[i, j] = stdev_value_measure

                        # Вычисление стандартного отклонения для контрольной выборки (random)
                        list_of_arrays_random = cluster_data_random.get(p_btw, [])
                        stdev_value_random = 0.0
                        if list_of_arrays_random:
                            num_pinputs = len(dict_pinput_random.keys())
                            n_rep = len(list_of_arrays_random) // num_pinputs if num_pinputs > 0 else 1
                            index_to_use = (i + 1) * n_rep - 1 if n_rep > 0 else 0
                            try:
                                trial_array = np.array(list_of_arrays_random[index_to_use], dtype=float)
                                stdev_value_random = np.std(trial_array)
                            except Exception as e:
                                print(e)
                        Z_random_std[i, j] = stdev_value_random

                        # Извлечение дополнительных синаптических параметров для measure
                        avg_exc_synapses_measure = float(data_entry_measure.get('avg_exc_synapses', 0.0))
                        avg_inh_synapses_measure = float(data_entry_measure.get('avg_inh_synapses', 0.0))
                        avg_exc_synapses_random = float(data_entry_random.get('avg_exc_synapses', 0.0))
                        avg_inh_synapses_random = float(data_entry_random.get('avg_inh_synapses', 0.0))
                        
                        # Запись строк в CSV для measure_name и для random
                        writer.writerow([
                            I0_value, measure_name, p_within_str, p_inp_str, p_btw_str,
                            mean_spikes_measure_val, stdev_value_measure,
                            avg_exc_synapses_measure, avg_inh_synapses_measure
                        ])
                        writer.writerow([
                            I0_value, 'random', p_within_str, p_inp_str, p_btw_str,
                            mean_spikes_random_val, stdev_value_random,
                            avg_exc_synapses_random, avg_inh_synapses_random
                        ])
                
                # Построение 3D-графика для текущей пары (measure_name, p_within)
                p_input_mesh, p_between_mesh = np.meshgrid(p_input_list_float, p_between_list_float, indexing='ij')
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(
                    f'3D: p_input vs p_between vs avg_spikes\n'
                    f'I0={I0_value} pA, freq={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}\n'
                    f'p_within={p_within_str}, Time={current_time} ms, Bin={time_window_size} ms\n'
                    f'Metrics: {measure_name} and random',
                    fontsize=13
                )
                ax.set_zlim(0, max_rate)
                ax.set_xlabel('p_input')
                ax.set_ylabel('p_between')
                ax.set_zlabel('avg_spikes (Hz)')
                ax.set_xticks(p_input_values)
                ax.set_yticks(p_between_values)
                # Поверхность для основной метрики
                surf_measure = ax.plot_surface(
                    p_input_mesh, p_between_mesh, Z_measure,
                    cmap='magma', edgecolor='none', zorder=1, vmin=0, vmax=max_rate, alpha=0.7
                )
                # Поверхность для контрольной выборки ("random")
                surf_random = ax.plot_surface(
                    p_input_mesh, p_between_mesh, Z_random,
                    cmap='magma', edgecolor='none', zorder=1, vmin=0, vmax=max_rate, alpha=0.5
                )
                # Добавление вертикальных линий для error bars по основной метрике
                for irow in range(Z_measure.shape[0]):
                    for jcol in range(Z_measure.shape[1]):
                        x_val = p_input_mesh[irow, jcol]
                        y_val = p_between_mesh[irow, jcol]
                        z_val = Z_measure[irow, jcol]
                        err = Z_measure_std[irow, jcol]
                        ax.plot([x_val, x_val], [y_val, y_val],
                                [z_val - err, z_val + err], c='k', zorder=2, linewidth=1.5)
                # Error bars для контрольной выборки ("random")
                for irow in range(Z_random.shape[0]):
                    for jcol in range(Z_random.shape[1]):
                        x_val = p_input_mesh[irow, jcol]
                        y_val = p_between_mesh[irow, jcol]
                        z_val = Z_random[irow, jcol]
                        err = Z_random_std[irow, jcol]
                        ax.plot([x_val, x_val], [y_val, y_val],
                                [z_val - err, z_val + err], c='k', zorder=2, linewidth=1.5)
                cb1 = fig.colorbar(surf_measure, ax=ax, shrink=0.5, aspect=10)
                
                # Сохранение графика в указанный каталог
                filename = os.path.join(
                    directory_path,
                    f'3D_two_surfaces_{measure_name}_vs_random_I0_{I0_value}_p_within_{p_within_str}_Time_{current_time}ms.png'
                )
                plt.savefig(filename, dpi=150)
                plt.close(fig)
    
    print("Формирование объединённых графиков (основная метрика vs random) завершено.")

def plot_all_measures_vs_random_from_csv(
    csv_filename,
    p_input_values,
    p_between_values,
    max_rate,
    I0_value_filter=1000,
    output_pdf="results_ext_test1/3D_all_measures_vs_random.pdf",
    measure_names=None
):
    """
    Функция читает данные из CSV-файла, содержащего колонки:
      I0_value, measure_name, p_within, p_input, p_between, mean_spikes, std_spikes.
      
    Для 6 метрик (measure_name, кроме 'random') строится объединённый 3D-рисунок. Для каждой метрики 
    выбирается одно значение p_within (первое по сортировке), после чего на соответствующем сабплоте 
    отображаются две поверхности – одна для основной метрики, другая для контрольной (measure_name 'random').
      
    В каждой точке также отрисовываются вертикальные error bar, соответствующие значению std_spikes.
    Для всех сабграфиков используется единая цветовая шкала, а общий colormap отображается через колорбар справа.
    
    Параметры:
      csv_filename : str
          Путь к CSV-файлу с данными.
      max_rate : float
          Максимальное значение для оси Z и нормализации colormap.
      I0_value_filter : int или float, по умолчанию 1000
          Значение I0_value для фильтрации данных.
      output_pdf : str, по умолчанию "3D_all_measures_vs_random.pdf"
          Имя выходного PDF-файла.
      measure_names : list или None
          Список метрик для отображения. Если None, выбираются все метрики, кроме 'random'.
    """
    # Чтение CSV-файла и фильтрация по I0_value
    df = pd.read_csv(csv_filename)
    df = df[df["I0_value"] == I0_value_filter]
    
    # Получаем список всех уникальных метрик и исключаем 'random'
    all_measures = sorted(df["measure_name"].unique())
    available_measures = [m for m in all_measures if m != 'random']
    if measure_names is not None:
        available_measures = [m for m in measure_names if m in available_measures]
    
    # Если найдено не 6 метрик, берем первые 6
    if len(available_measures) != 6:
        print(f"Ожидается 6 метрик, а найдено {len(available_measures)}. Будут использованы первые 6.")
        available_measures = available_measures[:6]
    
    # Настройка фигуры с 6 сабплотами (2 строки x 3 столбца)
    fig = plt.figure(figsize=(16, 10))
    marker_size = 50  # размер маркеров для scatter
    cmap = plt.get_cmap('magma')
    norm = mpl.colors.Normalize(vmin=0, vmax=max_rate)
    cent_rus = ['Степень посредничества', 'Степень близости', 'Степень вершины', 'Степень влиятельности', 'Гармоническая центральность', 'Центральность просачивания', 'Случайные узлы']

    # Перебор метрик для построения отдельных 3D-графиков
    for idx, measure in enumerate(available_measures):
        # Отбор данных для выбранной метрики и для 'random'
        df_measure = df[df["measure_name"] == measure]
        df_random = df[df["measure_name"] == "random"]
        
        # Выбор одного значения p_within (первое по сортировке)
        p_within_vals_measure = sorted(df_measure["p_within"].unique(), key=float)
        if not p_within_vals_measure:
            print(f"Нет значений p_within для {measure}. Пропуск.")
            continue
        p_within_val = p_within_vals_measure[0]
        
        # Фильтрация по выбранному p_within
        df_measure_pw = df_measure[df_measure["p_within"] == p_within_val]
        df_random_pw = df_random[df_random["p_within"] == p_within_val]
        
        # Определение общего множества значений p_input и p_between
        common_p_input = sorted(list(set(df_measure_pw["p_input"].unique()).intersection(
            set(df_random_pw["p_input"].unique()))), key=float)
        common_p_between = sorted(list(set(df_measure_pw["p_between"].unique()).intersection(
            set(df_random_pw["p_between"].unique()))), key=float)
        
        if not common_p_input or not common_p_between:
            print(f"Нет пересечения значений p_input или p_between для {measure} при p_within={p_within_val}. Пропуск.")
            continue
        
        # Используем pivot_table с агрегирующей функцией для устранения дублирования
        pivot_measure_mean = df_measure_pw.pivot_table(index="p_input", columns="p_between", values="mean_spikes", aggfunc=np.mean)
        pivot_random_mean = df_random_pw.pivot_table(index="p_input", columns="p_between", values="mean_spikes", aggfunc=np.mean)
        pivot_measure_std = df_measure_pw.pivot_table(index="p_input", columns="p_between", values="std_spikes", aggfunc=np.mean)
        pivot_random_std = df_random_pw.pivot_table(index="p_input", columns="p_between", values="std_spikes", aggfunc=np.mean)
        
        # Ограничиваем данные пересечением общих значений
        pivot_measure_mean = pivot_measure_mean.loc[common_p_input, common_p_between]
        pivot_random_mean = pivot_random_mean.loc[common_p_input, common_p_between]
        pivot_measure_std = pivot_measure_std.loc[common_p_input, common_p_between]
        pivot_random_std = pivot_random_std.loc[common_p_input, common_p_between]
        
        # Преобразуем списки ключей в массивы и создаем сетку для построения поверхности
        p_input_arr = np.array(common_p_input, dtype=float)
        p_between_arr = np.array(common_p_between, dtype=float)
        p_input_mesh, p_between_mesh = np.meshgrid(p_input_arr, p_between_arr, indexing='ij')
        
        # Извлекаем матрицы значений
        Z_measure = pivot_measure_mean.values.astype(float)
        Z_random = pivot_random_mean.values.astype(float)
        Z_measure_std = pivot_measure_std.values.astype(float)
        Z_random_std = pivot_random_std.values.astype(float)
        
        # Создание саблота для текущей метрики
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        ax.set_title(f"{cent_rus[idx]}\n против случайных узлов / \n {measure} vs random, $p_{{intra}}={p_within_val}$", fontsize=16)
        ax.set_zlim(0, max_rate)
        ax.set_xlabel(r'$p_{input}$', fontsize=14)
        ax.set_ylabel(r'$p_{inter}$', fontsize=14)
        ax.set_zlabel(r'$\overline{spikes}$', fontsize=14)
        ax.set_xticks(p_input_values)   # или какие именно точки хотите
        ax.set_yticks(p_between_values)
        # Увеличение размера подписей отметок осей
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # Построение поверхности для measure
        surf_measure = ax.plot_surface(
            p_input_mesh, p_between_mesh, Z_measure,
            cmap=cmap, edgecolor='none', alpha=0.8, vmin=0, vmax=max_rate
        )
        # Построение поверхности для random
        surf_random = ax.plot_surface(
            p_input_mesh, p_between_mesh, Z_random,
            cmap=cmap, edgecolor='none', alpha=0.6, vmin=0, vmax=max_rate
        )
        
        # Отрисовка error bar для measure
        for i in range(Z_measure.shape[0]):
            for j in range(Z_measure.shape[1]):
                x_val = p_input_mesh[i, j]
                y_val = p_between_mesh[i, j]
                z_val = Z_measure[i, j]
                err = Z_measure_std[i, j]
                ax.plot([x_val, x_val], [y_val, y_val], [z_val - err, z_val + err],
                        color='k', linewidth=1.5)
        
        # Отрисовка error bar для random
        for i in range(Z_random.shape[0]):
            for j in range(Z_random.shape[1]):
                x_val = p_input_mesh[i, j]
                y_val = p_between_mesh[i, j]
                z_val = Z_random[i, j]
                err = Z_random_std[i, j]
                ax.plot([x_val, x_val], [y_val, y_val], [z_val - err, z_val + err],
                        color='k', linewidth=1.5)
    
    # Добавление общего колорбара справа от всех сабграфиков
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\overline{spikes}$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=3, w_pad=0.1)
    plt.savefig(output_pdf)
    plt.show()
    
    print("Построение объединённого графика для 6 метрик vs random завершено.")




import os
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def circular_positions(nodes, center, radius):
    """
    Вычисляет позиции узлов в форме окружности.
    
    Аргументы:
      nodes  – список идентификаторов узлов,
      center – кортеж (x, y), задающий центр окружности,
      radius – радиус окружности.
      
    Возвращает:
      Словарь, где ключ – идентификатор узла, значение – координаты (x, y).
    """
    pos = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        pos[node] = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
    return pos

def plot_networkx_connectivity(W, central_mask, measure, p_in_measured, p_out_measured,
                                              percent_central, highlighted_neurons, directory_path='results_ext_test1',
                                              p_input=0.20, p_between=0.1):
    """
    Визуализация графа с разделением на два кластера (два круга) и раскраской ребер по аналогии с визуализацией матрицы M_up.
    
    Для каждого ребра:
      - Если W[i,j] == 1, базовый цвет – синий,
      - Если W[i,j] == 2, базовый цвет – красный.
      - Если ребро отмечено как центральное (central_mask[i,j] == True), оно разбивается на две части:
        первая половина (от источника до середины) рисуется жёлтым, вторая – базовым цветом.
    
    Узлы разделяются на два кластера на основе их индексов:
      - Кластер 1: узлы с номерами от 0 до k-1,
      - Кластер 2: узлы с номерами от k до n_neurons - 1,
      где k = n_neurons // 2.
      
    Для каждого кластера вычисляются позиции узлов по окружности с центрами (-1, 0) и (1, 0) соответственно.
    
    Результатом работы функции является сохранённый PDF-файл с визуализацией графа, а также возвращаются
    объект графа G и словарь позиций pos.
    """
    n_neurons = W.shape[0]
    
    # Разбиение узлов на два кластера
    k = n_neurons // 2
    cluster1 = list(range(k))
    cluster2 = list(range(k, n_neurons))
    
    # Вычисляем позиции для кластеров:
    # Например, зададим радиус окружности равным 1.0 для обоих кластеров,
    # а центры – (-1,0) для кластера 1 и (1,0) для кластера 2.
    pos_cluster1 = circular_positions(cluster1, center=(-1.5, 0), radius=1.0)
    pos_cluster2 = circular_positions(cluster2, center=(1.5, 0), radius=1.0)
    pos = {**pos_cluster1, **pos_cluster2}
    
    # Создание ориентированного графа
    G = nx.DiGraph()
    G.add_nodes_from(range(n_neurons))
    
    # Заполнение графа ребрами согласно матрице W
    for i in range(n_neurons):
        for j in range(n_neurons):
            if W[i, j] != 0:
                base_color = 'blue' if W[i, j] == 1 else 'red'
                is_central = central_mask[i, j]
                G.add_edge(i, j, weight=W[i,j], base_color=base_color, central=is_central)
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Отрисовка узлов
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    # Рёбра, не помеченные как центральные, отрисовываем стандартно
    non_central_edges = [(u, v) for u, v, data in G.edges(data=True) if not data['central']]
    edge_colors = [G[u][v]['base_color'] for u, v in non_central_edges]
    nx.draw_networkx_edges(G, pos, edgelist=non_central_edges, edge_color=edge_colors, arrows=True, ax=ax)
    
    # Рёбра, отмеченные как центральные, отрисовываем с разделением на две части
    central_edges = [(u, v) for u, v, data in G.edges(data=True) if data['central']]
    for (u, v) in central_edges:
        base_color = G[u][v]['base_color']
        start = np.array(pos[u])
        end = np.array(pos[v])
        mid = (start + end) / 2  # середина отрезка
        
        # Рисуем первую половину (от start до mid) жёлтым
        line1 = plt.Line2D([start[0], mid[0]], [start[1], mid[1]], color='yellow',
                           linewidth=2, solid_capstyle='round')
        ax.add_line(line1)
        # Рисуем вторую половину (от mid до end) базовым цветом
        line2 = plt.Line2D([mid[0], end[0]], [mid[1], end[1]], color=base_color,
                           linewidth=2, solid_capstyle='round')
        ax.add_line(line2)
    
    title = (
        f'Связность внутри кластера: {p_in_measured:.3f},  '
        f'связность между кластерами: {p_out_measured:.3f}\n'
        f'Процент центральных нейронов: {percent_central}%  '
        f'Метрика: {measure}'
    )
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    filename = os.path.join(directory_path, f"networkx_two_clusters_pinput_{p_input:.2f}_pbetween_{p_between:.3f}.pdf")
    plt.savefig(filename)
    plt.close()
    
    return G, pos



def plot_connectivity(subplot_results, centrality, p_input=0.20, p_between=0.1, directory_path='results_ext_test1'):
    """
    Функция визуализации матрицы связей с повышенным разрешением для имитации разделённого цвета одного пикселя.
    
    Для каждой ячейки (пикселя) матрицы:
      - Если значение в матрице W равно 1, то базовый цвет – синий.
      - Если W равно 2, то базовый цвет – красный.
      - Если ячейка участвует в связи центрального нейрона, то представляется блоком субпикселей,
        где левая половина (столбец) окрашивается жёлтым (цвет центральных узлов),
        а правая половина – базовым цветом (синий или красный).
        
    Реализуется через увеличение разрешения (фактор upscale_factor = 2).
    """
    upscale_factor = 2  # Каждый логический пиксель представляется блоком 2x2

    if subplot_results:
        n_measures = len(subplot_results)
        
        # Определение структуры субплотов
        if n_measures > 4:
            nrows = 2
            ncols = math.ceil(n_measures / 2)
        else:
            nrows = 1
            ncols = n_measures
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8 * nrows))
        axes = np.array(axes).flatten() if n_measures > 1 else [axes]
        
        # Определяем дискретную цветовую карту:
        # 0 – фон (чёрный), 1 – синий, 2 – красный, 3 – жёлтый (центральные)
        colors = ['white', 'blue', 'red', 'yellow']
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, len(colors))
        
        for ax, (measure, params) in zip(axes, subplot_results.items()):
            # Распаковка параметров для текущей метрики
            W, M, p_in_measured, p_out_measured, percent_central = params

            # Ключ для поиска центральных нейронов (настройки симуляции)
            key = (1000, 10, False, '0.15', '0.20', 0.1, measure)
            if key in centrality:
                highlighted_neurons = np.array(centrality[key], dtype=int)
            else:
                highlighted_neurons = np.array([], dtype=int)
    
            n_neurons = W.shape[0]
            
            # Создаём базовую матрицу цвета для визуализации:
            # 0 – отсутствие связи, 1 – связь типа 1 (синий), 2 – связь типа 2 (красный)
            M_base = np.zeros_like(W, dtype=int)
            M_base[W == 1] = 1
            M_base[W == 2] = 2

            # Создаём логическую маску центральных связей: 
            # если номер строки или столбца принадлежит highlighted_neurons, считается, что связь центральная.
            central_mask = np.zeros_like(M_base, dtype=bool)
            for neuron in highlighted_neurons:
                # Отмечаем как центральные все связи, связанные с данным нейроном (исходящие и входящие)
                central_mask[neuron, :] = True
                central_mask[:, neuron] = True

            # Создаём матрицу повышенного разрешения
            up_shape = (n_neurons * upscale_factor, n_neurons * upscale_factor)
            M_up = np.zeros(up_shape, dtype=int)
            
            # Обходим каждую ячейку исходной матрицы
            for i in range(n_neurons):
                for j in range(n_neurons):
                    # Определяем базовый цвет для данной ячейки (0, 1 или 2)
                    base_color = M_base[i, j]
                    # Определяем координаты блока в матрице повышенного разрешения
                    i_start, i_end = i * upscale_factor, (i + 1) * upscale_factor
                    j_start, j_end = j * upscale_factor, (j + 1) * upscale_factor
                    
                    # Если связь не установлена, оставляем фон (0)
                    if base_color == 0:
                        M_up[i_start:i_end, j_start:j_end] = 0
                    else:
                        if central_mask[i, j]:
                            # Если связь центральная, формируем эффект "наполовину":
                            # Разбиваем блок вертикально: левая колонка – жёлтый (цвет 3), правая – базовый цвет.
                            # Если upscale_factor > 2, можно масштабировать аналогичным соотношением.
                            mid = upscale_factor // 2
                            # Левая половина
                            M_up[i_start:i_end, j_start:j_start+mid] = 3
                            # Правая половина
                            M_up[i_start:i_end, j_start+mid:j_end] = base_color
                        else:
                            # Если связь не центральная, блок заполняется полностью базовым цветом.
                            M_up[i_start:i_end, j_start:j_end] = base_color
            
            # Визуализируем полученную матрицу повышенного разрешения
            ax.matshow(M_up, cmap=cmap, norm=norm)
            
            # Настройка заголовка и меток осей
            ax.set_title(
                f'Связность внутри кластера: {p_in_measured:.3f},\n'
                f'Связность между кластерами: {p_out_measured:.3f},\n'
                f'Процент центральных нейронов: {percent_central}%,\n'
                f'Метрика: {measure}',
                fontsize=14
            )
            # Отображаем оси с учётом нового размера матрицы
            # new_dim = n_neurons * upscale_factor
            # ticks = [0, new_dim // 2, new_dim - 1]
            # ax.set_xticks(ticks)
            # ax.set_yticks(ticks)
            # ax.tick_params(axis='both', labelsize=12)
        
        # Отключаем неиспользуемые оси, если их число больше числа метрик
        # for ax in axes[len(subplot_results):]:
        ax.axis('off')
    
        plt.tight_layout()
        # Сохранение рисунка с указанными параметрами
        filename = os.path.join(directory_path, f"connectivity_pinput_{p_input:.2f}_pbetween_{p_between:.3f}.pdf")
        plt.savefig(filename)
        plt.close()

        # Вызываем функцию для построения графа с помощью networkx.
        G, pos = plot_networkx_connectivity(
            W, central_mask, measure, p_in_measured, p_out_measured,
            percent_central, highlighted_neurons, directory_path='results_ext_test1',
            p_input=0.20, p_between=0.1
        )


# measure_order = [
#     "betweenness", "closeness", "degree", "random",
#     "eigenvector","harmonic", "percolation", 
# ]

# rx = re.compile(
#     r'^(?P<measure>\w+)_test(?P<test>\d+)_'
#     r'within(?P<pw>[0-9_]+)_between(?P<pb>[0-9_]+)_input(?P<pi>[0-9_]+)_spikes'
# )

# stat: dict[str, list[tuple[float, float, float, float]]] = {m: [] for m in measure_order}
# spike_dir = Path("results_ext_test1/spikes")

# tmp: dict[str, dict[tuple[float, float], list[float]]] = {m: {} for m in measure_order}

# for txt in spike_dir.glob("*.txt"):
#     m = rx.match(txt.stem)
#     if not m:
#         continue

#     meas = m.group("measure")
#     if meas not in measure_order:
#         continue

#     pb = float(m.group("pb").replace("_", "."))
#     pi = float(m.group("pi").replace("_", "."))

#     trains = spk.load_spike_trains_from_txt(txt, edges=(1, 5000))
#     f = spk.spike_sync_profile(trains[250:])          # только 251‑500
#     z = f.avrg() 

#     # x, y = f.get_plottable_data()
#     # plt.figure()
#     # plt.plot(x, y, '-b', alpha=0.7, label="SPIKE-Sync profile")
#     # plt.show()
#     tmp[meas].setdefault((pb, pi), []).append(z)

# # превращаем в «mean ± std»
# for meas, pairs in tmp.items():
#     for (pb, pi), zs in pairs.items():
#         zs = np.asarray(zs)
#         stat[meas].append((
#             pb, pi,
#             zs.mean(),
#             zs.std(ddof=1) if len(zs) > 1 else 0.
#         ))

# --- 0. Русско‑английское соответствие ---------------------------------------
cent_map = {
    "betweenness":  "Степень посредничества",
    "closeness":    "Степень близости",
    "degree":       "Степень вершины",
    "random":       "Случайные узлы",
    "eigenvector":  "Степень влиятельности",
    "harmonic":     "Гармоническая центральность",
    "percolation":  "Центральность просачивания",
}

# # --- 1. функция рисования поверхности ---------------------------------------
# def draw_surface(ax: Axes3D, rows, meas, cmap="viridis"):
#     if not rows:                          # на всякий случай
#         ax.set_title(f"{cent_map[meas]} / {meas}\n(no data)")
#         return

#     rows = np.asarray(rows)               # (N, 4) : pb, pi, mean, std
#     Xu, Yu = np.unique(rows[:, 0]), np.unique(rows[:, 1])
#     X, Y   = np.meshgrid(Xu, Yu)

#     Zm = griddata(rows[:, :2], rows[:, 2], (X, Y), method="linear")
#     Zs = griddata(rows[:, :2], rows[:, 3], (X, Y), method="linear")

#     # заполнение возможных NaN
#     if np.isnan(Zm).any():
#         from scipy.ndimage import generic_filter
#         Zm = generic_filter(Zm, np.nanmean, size=3, mode="nearest")
#         Zs = generic_filter(Zs, np.nanmean, size=3, mode="nearest")

#     ax.plot_surface(X, Y, Zm, cmap=cmap, edgecolor="k",
#                     linewidth=.3, antialiased=True, alpha=.9)

#     # усики ±1 σ
#     for x, y, m, s in rows:
#         ax.plot([x, x], [y, y], [m - s, m + s], color='k', linewidth=.8)

#     ax.set_xlabel("p_between")
#     ax.set_ylabel("p_input")
#     ax.set_zlabel("SPIKE‑distance")
#     ax.set_title(f"{cent_map[meas]} / {meas}", fontsize=11)
#     ax.set_xticks(Xu);  ax.set_yticks(Yu)

# # --- 2. вывод 4+3 субплотов --------------------------------------------------
# fig = plt.figure(figsize=(18, 10))

# # добавляем запас справа и чуть увеличиваем междуосевое расстояние
# fig.subplots_adjust(
#     left=0.1,   right=0.9,     # ←→ общий отступ по краям
#     bottom=0.1, top=0.9,       # ↑↓ общий отступ
#     wspace=0.1, hspace=0.20     # расстояния между субплотами
# )

# pos_map = [1, 2, 3, 4, 5, 6, 7]  # ячейка 8 пустая

# for meas, pos in zip(measure_order, pos_map):
#     ax = fig.add_subplot(2, 4, pos, projection="3d")
#     draw_surface(ax, stat[meas], meas)
#     # подпись‑z с лёгким смещением внутрь, чтобы точно не срезалась
#     ax.set_zlabel("SPIKE‑distance", labelpad=10)

# fig.add_subplot(2, 4, 8).axis("off")      # пустая ячейка
# pdf_name = "results_ext_test1/spike_profiles_7measures_251‑500.pdf"
# fig.savefig(pdf_name, format="pdf")
# print(f"Фигура сохранена в «{pdf_name}»")


with open('results_ext_test1/detailed_spike_data_for_3d.pickle', 'rb') as f:
    detailed_spike_data_for_3d = pickle.load(f)
with open('results_ext_test1/spike_counts_second_cluster_for_input.pickle', 'rb') as f:
    spike_counts_second_cluster_for_input = pickle.load(f)
with open('results_ext_test1/spike_counts_second_cluster.pickle', 'rb') as f:
    spike_counts_second_cluster = pickle.load(f)
with open('results_ext_test1/subplot_results.pickle', 'rb') as f:
    subplot_results = pickle.load(f)
with open('results_ext_test1/centrality_results.pickle', 'rb') as f:
    centrality = pickle.load(f)


p_input_values = np.arange(0.1, 0.21, 0.05)
p_between_values = np.arange(0.01, 0.11, 0.03)
p_within_values = np.arange(0.15, 0.16, 0.05)
measure_names = [
    "degree",  
    "betweenness",
    "closeness",
    "random",
    "eigenvector",
    "percolation",
    "harmonic",
]
I0_value = 1000 # pA
oscillation_frequency = [10]
use_stdp = [False]
current_time = 5000 # in ms
time_window_size = 1000  # in ms
max_rate_on_graph = 10
directory_path = 'results_ext_test1'


plot_3d_spike_data(
    detailed_spike_data_for_3d,
    measure_names + ['random'],  # включаем и random
    p_within_values,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate_on_graph,
    directory_path=directory_path,
)

# Построение 3D-графиков (p_input vs p_between vs avg_spikes) — две поверхности: measure & random
plot_pinput_between_avg_spikes_with_std(
    spike_counts_second_cluster_for_input,
    spike_counts_second_cluster,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate_on_graph,
    p_input_values, 
    p_between_values,
    directory_path=directory_path,
    measure_names=measure_names + ['random'],
)
# Пример вызова функции
plot_all_measures_vs_random_from_csv(
    "results_ext_test1/avg_tests_avg_spikes_5000ms.csv",
    p_input_values,
    p_between_values,
    max_rate=max_rate_on_graph,
)



# ────────────────────── 2. Параметры ──────────────────────────────────────
results_dir   = "results_ext_test1/rates"
p_within      = 0.15          # 0.15 → подстрока 0_15
p_between     = 0.01          # 0.01 → 0_01
p_input       = 0.20          # 0.10 → 0_10
f_max         = 50.0          # Гц, до какой частоты строить PSD

measure_names = [
    "betweenness", "closeness", "degree", "random",
    "eigenvector","harmonic", "percolation", 
]


file_template = (
    f"rates_within{p_within:.2f}_between{p_between:.2f}_input{p_input:.2f}_"
    f"{{measure}}_test1.pickle"
)

file_dict = {}
for m in measure_names:
    print(file_template.format(measure=m))
    candidates = sorted(
        glob.glob(os.path.join(results_dir, file_template.format(measure=m)))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Нет файлов для метрики «{m}» "
            f"(within={p_within}, between={p_between}, input={p_input})"
        )
    file_dict[m] = candidates[-1]        # берём файл с максимальным testN

# ────────────────────── 4. Загрузка и PSD ─────────────────────────────────
def load_group1(fname):
    """Возвращает (t_ms, rate_Hz) первой половины нейронов."""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname, "rb") as f:
        d = pickle.load(f)
    # теперь читаем напрямую из t_group2 и rate_group2
    if "t_group2" not in d or "rate_group1" not in d:
        raise KeyError(f"В файле {fname} нет ожидаемых ключей 't_group2' и 'rate_group2'")
    t_ms   = d["t_group2"]
    rate_hz = d["rate_group2"]
    return t_ms, rate_hz


def calc_psd(t_ms, rate_hz, f_lim=f_max):
    rate = rate_hz - rate_hz.mean()
    N    = len(rate)
    if N < 2:
        return np.array([]), np.array([])
    dt   = (t_ms[1] - t_ms[0]) / 1000.0  # шаг, с
    fs   = 1.0 / dt                     # частота дискр., Гц
    win  = np.hanning(N)
    psd  = np.abs(rfft(rate * win)) / N
    freq = rfftfreq(N, d=1/fs)
    sel  = freq <= f_lim
    return freq[sel], psd[sel]


# ────────────────────── 5. Визуализация с несколькими p_between ───────────────────────────────────
p_between_list = [0.01, 0.04, 0.07, 0.10]  # четыре интересующих значения
colors = ['C0', 'C1', 'C2', 'C3']          # базовые цвета matplotlib

fig, axes = plt.subplots(
    nrows=2, ncols=4,
    figsize=(14, 8),
    sharey=True,
    constrained_layout=True
)
axes_flat = axes.flatten()

for idx, m in enumerate(measure_names):
    ax = axes_flat[idx]
    for j, pb in enumerate(p_between_list):
        # строим имя файла для текущего pb и метрики m
        tpl = (
            f"rates_within{p_within:.2f}_"
            f"between{pb:.2f}_"
            f"input{p_input:.2f}_"
            f"{m}_test1.pickle"
        )
        candidates = sorted(glob.glob(os.path.join(results_dir, tpl)))
        if not candidates:
            raise FileNotFoundError(
                f"Нет файлов для метрики «{m}» "
                f"(within={p_within}, between={pb}, input={p_input})"
            )
        fname = candidates[-1]

        # загрузка и расчет PSD
        t, r = load_group1(fname)
        f, s = calc_psd(t, r, f_lim=f_max)

        # отрисовка
        ax.plot(f, s, label=f"{pb:.2f}", color=colors[j])

    ax.set_title(f"{cent_map[m]} / \n {m}", fontsize=14)
    ax.set_xlim(0, f_max)
    ax.set_xlabel("f, Гц", fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(title="$p_{{inter}}$", fontsize=14, title_fontsize=14)

# отключаем лишнюю восьмую ячейку
axes_flat[-1].axis('off')

# общие подписи
for i in range(2):
    for j in range(4):
        axes[i, j].set_ylabel("СПМ / PSD", fontsize=12)

pdf_name = "results_ext_test1/PSD_7measures_251‑500.pdf"
fig.savefig(pdf_name, format="pdf")
plt.show()


plot_connectivity(subplot_results, centrality)

