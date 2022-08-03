import matplotlib.pyplot as plt
import pickle
import numpy as np

markers = ['*', 'o', 'x', '^']


############### AGAINST RANDOM OPPONENT #############
with open('all.pckl', 'rb') as file:
    N_types = pickle.load(file)
    avg_Regrets_all = pickle.load(file)
    std_Regrets_all = pickle.load(file)
    avg_expected_Regrets_all = pickle.load(file)
    std_expected_Regrets_all = pickle.load(file)

T = len(avg_Regrets_all[0])


fig = plt.figure(figsize=(14, 9))
plt.title('Average Expected Regret')
idx_marker = 0
for i in range(len(N_types)):
    p = plt.plot(np.arange(T), avg_expected_Regrets_all[i], marker = markers[idx_marker], markevery = 10, markersize = 7, label = f'all {N_types[i][0]}')
    color = p[0].get_color()
    plt.fill_between(range(T), avg_expected_Regrets_all[i] - std_expected_Regrets_all[i], avg_expected_Regrets_all[i] + std_expected_Regrets_all[i], alpha=0.1,
                     color=color)
    idx_marker += 1


plt.legend()
plt.xlabel('time')
fig.tight_layout()
plt.rcParams.update({'font.size': 14})

plt.show()




