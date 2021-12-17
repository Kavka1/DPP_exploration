from typing import List, Tuple, Type, Dict
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.core.fromnumeric import mean, std
import seaborn as sns
import pandas as pd
from seaborn.utils import ci


def data_smooth(data: np.array, sm: int) -> np.array:
    new_data = []
    for d in data:
        y = np.ones(sm) * 1.0 / sm
        d = np.convolve(y, d, 'same')
        new_data.append(d)
    return np.array(new_data, dtype=np.float64)


def load_single_data(exp_path: str, miss_load: bool) -> np.array:
    assert os.path.exists(exp_path)
    
    if miss_load:
        with open(exp_path + 'stats_1.csv', 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)
        data = data.values.tolist()
        model_data = [[], [], [], []]    
        for i, term in enumerate(data):
            index = i % 4
            model_data[index].append(term[-1])
    else:
        model_data = []
        for i in range(4):
            with open(exp_path + f'stats_{i}.csv', 'r', encoding='utf-8') as f:
                data = pd.read_csv(f)
            data = data.values.tolist()
            data = [item[-1] for item in data]
            model_data.append(data)
    
    length = min([len(data) for data in model_data])
    data_array = np.array([single_model_data[:length] for single_model_data in model_data], dtype=np.float64)

    return data_array


def plot_single_compartion(path_a: str, path_b: str, miss_load_a: bool, miss_load_b: bool, title: str, sm: int = 15) -> None:
    data_a = load_single_data(path_a, miss_load_a)
    data_b = load_single_data(path_b, miss_load_b)

    if len(data_a[0]) < len(data_b[0]):
        length = len(data_a[0])
        data_b = [data[:length] for data in data_b]
    else:
        length = len(data_b[0])
        data_a = [data[:length] for data in data_a]

    y_a, y_b = data_smooth(data_a, sm), data_smooth(data_b, sm)
    
    sns.set(style='darkgrid', font_scale=1)
    sns.tsplot(y_a, color='r', condition='TD3_DPP', ci='sd')
    sns.tsplot(y_b, color='b', condition='TD3', ci='sd')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    arguments = {
        'Swimmer': [
            '/home/xukang/GitRepo/DPP_exploration/results/Swimmer_12-15_16-35/',
            '/home/xukang/GitRepo/DPP_exploration/results/Swimmer_12-15_17-35/',
            True,
            True,
            'Swimmer-v2',
            15
        ],
        'Hopper': [
            '/home/xukang/GitRepo/DPP_exploration/results/Hopper-v2_12-16_13-44/',
            '/home/xukang/GitRepo/DPP_exploration/results/Hopper-v2_12-16_00-09/',
            False,
            False,
            'Hopper-v2',
            15
        ],
        'Walker': [
            '/home/xukang/GitRepo/DPP_exploration/results/Walker2d-v2_12-16_00-26/',
            '/home/xukang/GitRepo/DPP_exploration/results/Walker2d-v2_12-16_15-18/',
            False,
            False,
            'Walker-v2',
            15
        ],
        'Ant': [
            '/home/xukang/GitRepo/DPP_exploration/results/Ant-v2_12-16_22-32/',
            '/home/xukang/GitRepo/DPP_exploration/results/Ant-v2_12-16_22-36/',
            False,
            False,
            'Ant-v2',
            15
        ]
    }

    for item in arguments.values():
        plot_single_compartion(
            path_a= item[0],
            path_b= item[1],
            miss_load_a= item[2],
            miss_load_b= item[3],
            title= item[4],
            sm= item[5]
        )

    print('Over')