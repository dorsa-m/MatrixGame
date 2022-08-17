import matplotlib.pyplot as plt
import pickle
import numpy as np

markers = ['*', 'o', 'x', '^']


def show(num):
    with open('all.pckl', 'rb') as file:
        N = pickle.load(file)
        K = pickle.load(file)
        N_types = pickle.load(file)
        avg_expected_Regrets_all = pickle.load(file)
        std_expected_Regrets_all = pickle.load(file)
        avg_expected_Regrets_worst = pickle.load(file)
        std_expected_Regrets_worst = pickle.load(file)
        rate = pickle.load(file)

    T = len(avg_expected_Regrets_all[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 9))
    idx_marker = 0
    for i in range(len(N_types)):
        p = ax1.plot(np.arange(T), avg_expected_Regrets_all[i], marker=markers[idx_marker], markevery=10, markersize=7,
                     label=f'all {N_types[i][0]}')
        color = p[0].get_color()
        ax1.fill_between(range(T), avg_expected_Regrets_all[i] - std_expected_Regrets_all[i],
                         avg_expected_Regrets_all[i] + std_expected_Regrets_all[i], alpha=0.1,
                         color=color)
        ax1.set_title(f'Regret average all players N = {N}, K = {K}')
        p = ax2.plot(np.arange(T), avg_expected_Regrets_worst[i], marker=markers[idx_marker], markevery=10,
                     markersize=7, label=f'all {N_types[i][0]}')
        color = p[0].get_color()
        ax2.fill_between(range(T), avg_expected_Regrets_worst[i] - std_expected_Regrets_worst[i],
                         avg_expected_Regrets_worst[i] + std_expected_Regrets_worst[i], alpha=0.1,
                         color=color)
        ax2.set_title(f'Regret worst player N = {N}, K = {K}')
        idx_marker += 1

    plt.legend()
    plt.xlabel('time')
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.savefig(f'result rate {rate} run {num}.png')
    plt.show()


show(1)
