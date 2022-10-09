import numpy as np
import pickle
from tqdm import tqdm
from aux_functions import Assign_payoffs, Player_MWU, Player_GPMW, Player_OPT_MWU, Player_OPT_GPMW, Player_EXP3, Player_OPT_EXP3, joint_dist, get_combinations
from sklearn.gaussian_process.kernels import RBF

N = 4  # number of players
K = 5  # number of actions for each player
T = 500  # time horizon
sigma = 1

" Data to be saved (for post processing/plotting) "


class GameData:
    def __init__(self):
        self.Played_actions = []
        self.Mixed_strategies = []
        self.Obtained_payoffs = []
        self.History_GP_action = []
        self.History_GP_payoff = []

        self.Expected_regret = []
        self.Expected_Obtained_payoffs = []
        self.Expected_Cum_payoffs = []
        self.Expected_payoff_to_update = []

        # test for debugging max var
        self.history = []
        self.history_p = []


def RunGame(N, K, T, A, sigma, types, optimize=True, max_var=False, test=False):
    noises = np.random.normal(0, sigma, size=[N, T])
    Game_data = GameData()
    all_action_profiles = get_combinations(tuple([K] * N))

    Player = list(range(N))  # list of all players
    for i in range(N):
        Game_data.Expected_Cum_payoffs.append(np.zeros(K))
        Game_data.Expected_payoff_to_update.append([np.zeros(K), np.zeros(
            K)])  # for each player stores total expected payoff of each actions for 2 (current and prev) rounds

        # test
        Game_data.history.append([] * N)
        Game_data.history_p.append([] * N)

        min_payoff = A[i].min()
        max_payoff = A[i].max()

        if types[i] == 'MWU':
            Player[i] = Player_MWU(K, T, N, min_payoff, max_payoff)
        if types[i] == 'OPT_MWU':
            Player[i] = Player_OPT_MWU(K, T, N, min_payoff, max_payoff)
        if types[i] == 'GPMW':
            Player[i] = Player_GPMW(K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, A[i], optimize)
        if types[i] == 'OPT_GPMW':
            Player[i] = Player_OPT_GPMW(K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, A[i], optimize)
        if types[i] == 'EXP3':
            Player[i] = Player_EXP3(K, T, min_payoff, max_payoff)
        if types[i] == 'OPT_EXP3':
            Player[i] = Player_OPT_EXP3(K, T, min_payoff, max_payoff)

    " Repated Game "
    for t in tqdm(range(T)):
        " Compute played actions "
        Game_data.Played_actions.append([None] * N)  # initialize
        Game_data.Mixed_strategies.append([None] * N)  # initialize
        Game_data.Expected_regret.append([None] * N)  # initialize
        Game_data.Expected_Obtained_payoffs.append([None] * N)  # initialize
        Game_data.Obtained_payoffs.append([None] * N)  # initialize
        Game_data.History_GP_action.append([None] * N)  # initialize
        Game_data.History_GP_payoff.append([None] * N)  # initialize

        for i in range(N):
            Game_data.Mixed_strategies[t][i] = np.array(Player[i].mixed_strategy())
            Game_data.Played_actions[t][i] = np.random.choice(range(K), p=Game_data.Mixed_strategies[t][i])

        " Assign payoffs and compute regrets"

        for i in range(N):
            Game_data.Obtained_payoffs[t][i] = Assign_payoffs(Game_data.Played_actions[t], A[i])

            others_probabilities = Game_data.Mixed_strategies[t].copy()
            others_probabilities.pop(i)
            joint_dis = joint_dist(others_probabilities, K)

            expected_payoff_all_actions = []  # expected payoff of actions for player i at current time
            predicted_expected_payoff_all_actions = []
            for a in range(K):
                action_expected_payoff = np.sum(np.multiply(joint_dis, np.moveaxis(A[i], i, 0)[a, ...]))
                expected_payoff_all_actions.append(action_expected_payoff)
                Game_data.Expected_Cum_payoffs[i][a] += action_expected_payoff

                if Player[i].type == "GPMW" or Player[i].type == "OPT_GPMW":
                    predicted_action_expected_payoff = np.sum(
                        np.multiply(joint_dis, np.moveaxis(Player[i].UCB_Matrix, i, 0)[a, ...]))
                    predicted_expected_payoff_all_actions.append(predicted_action_expected_payoff)

            Game_data.Expected_payoff_to_update[i].pop(0)
            if Player[i].type == "GPMW" or Player[i].type == "OPT_GPMW":
                Game_data.Expected_payoff_to_update[i].append(np.array(predicted_expected_payoff_all_actions))
            else:
                Game_data.Expected_payoff_to_update[i].append(np.array(expected_payoff_all_actions))
            Game_data.Expected_Obtained_payoffs[t][i] = np.dot(np.array(expected_payoff_all_actions),
                                                               np.array(Game_data.Mixed_strategies[t][i]))

            Game_data.Expected_regret[t][i] = (np.max(Game_data.Expected_Cum_payoffs[i]) - sum(
                [Game_data.Expected_Obtained_payoffs[x][i] for x in range(t + 1)]))

            # Update Max_var data
            if Player[i].type == "GPMW" or Player[i].type == "OPT_GPMW":
                if t == 0:
                    Game_data.History_GP_action[t][i] = Game_data.Played_actions[t]
                    Game_data.History_GP_payoff[t][i] = Game_data.Obtained_payoffs[t][i]
                else:
                    variances = Player[i].var_matrix.reshape([K] * N)
                    max_var_action = list(np.unravel_index(np.argmax(variances, axis=None), variances.shape))
                    Game_data.History_GP_action[t][i] = max_var_action
                    Game_data.History_GP_payoff[t][i] = Assign_payoffs(max_var_action, A[i])

                # test
                Game_data.history[i].append(Game_data.Played_actions[t])
                Game_data.history_p[i].append(Game_data.Obtained_payoffs[t][i] + noises[i][t])
                Game_data.history[i].append(Game_data.History_GP_action[t][i])
                Game_data.history_p[i].append(Game_data.Obtained_payoffs[t][i] + noises[i][t])

        " Update players next mixed strategy "
        for i in range(N):
            if Player[i].type == "MWU":
                Player[i].Update(Game_data.Expected_payoff_to_update[i][1])

            if Player[i].type == "OPT_MWU":
                Player[i].Update(Game_data.Expected_payoff_to_update[i][1],
                                 Game_data.Expected_payoff_to_update[i][0])

            if Player[i].type == "GPMW" or Player[i].type == "OPT_GPMW":
                if max_var:
                    history_actions = np.array([Game_data.History_GP_action[x][i] for x in range(t + 1)])
                    history_payoffs = np.array([Game_data.History_GP_payoff[x][i] + noises[i][x] for x in range(t + 1)])
                else:
                    history_actions = np.array([Game_data.Played_actions[x] for x in range(t + 1)])
                    history_payoffs = np.array([Game_data.Obtained_payoffs[x][i] + noises[i][x] for x in range(t + 1)])

                if Player[i].type == "GPMW":
                    if test:
                        # test
                        # Player[i].GP_update(Game_data.history[i][:-1], Game_data.history_p[i][:-1])
                        Player[i].GP_update(Game_data.history[i], Game_data.history_p[i], False)
                    else:
                        Player[i].GP_update(history_actions, history_payoffs, True)
                    Player[i].Update(Game_data.Expected_payoff_to_update[i][1])

                if Player[i].type == "OPT_GPMW":
                    if test:
                        # test
                        # Player[i].GP_update(Game_data.history[i][:-1], Game_data.history_p[i][:-1])
                        Player[i].GP_update(Game_data.history[i], Game_data.history_p[i], False)
                    else:
                        Player[i].GP_update(history_actions, history_payoffs, True)
                    Player[i].Update(Game_data.Expected_payoff_to_update[i][1],
                                         Game_data.Expected_payoff_to_update[i][0])

            if Player[i].type == "EXP3" or Player[i].type == "OPT_EXP3":
                noisy_payoff = Game_data.Obtained_payoffs[t][i] + np.random.normal(0, sigma, 1)
                Player[i].Update(Game_data.Played_actions[t][i], noisy_payoff)

    return Game_data, Player


def Generate_A(K, N, smooth=True):
    if smooth:
        all_action_profiles = get_combinations(tuple([K] * N))
        Mu = 0 * np.ones(K ** N)
        Cov = RBF.__call__(RBF(), all_action_profiles)
        Realiz = np.random.multivariate_normal(Mu, Cov)

        # Compute Posterior Mean (has bounded RKHS norm w.r.t. Kernel)
        post_mean = np.zeros(K ** N)
        C = Cov + sigma ** 2 * np.eye(K ** N)
        tmp = np.linalg.inv(C).dot(Realiz)
        for i in range(K ** N):
            B = Cov[i, :]
            post_mean[i] = B.dot(tmp)

        Matrix = np.reshape(post_mean, tuple([K] * N))
        return Matrix
    else:
        return np.random.random(size=[K] * N)


" --------------------------------- Begin Simulations --------------------------------- "

simulations = 20
runs = 6

N_types = []
# N_types.append(['MWU'] * N)
# N_types.append(['OPT_MWU'] * N)
N_types.append(['GPMW'] * N)
N_types.append(['OPT_GPMW'] * N)
# N_types.append(['EXP3'] * N)
# N_types.append(['OPT_EXP3'] * N)

avg_expected_Regrets_all = []
std_expected_Regrets_all = []
avg_expected_Regrets_worst = []
std_expected_Regrets_worst = []

# np.random.seed(4)

A_all = []
for sim in range(simulations):
    A = []
    for j in range(N):
        A.append(Generate_A(K, N, True))
    A_all.append(A)

# with open('payoffs.pckl', 'rb') as file:
#     A_all = pickle.load(file)

# np.random.seed(12)

for i in range(len(N_types)):
    e_Regrets_all = []
    e_Regrets_worst = []
    for sim in range(simulations):
        for run in range(runs):
            Games_data, Player = RunGame(N, K, T, A_all[sim], sigma, N_types[i], max_var=False, optimize=False, test=False)

            # finding player with max regret
            ind_worst_e = 0
            s_e = 0
            for ind in range(N):
                if Games_data.Expected_regret[T - 1][ind] > s_e:
                    ind_worst_e = ind
                    s_e = Games_data.Expected_regret[T - 1][ind]
            e_Regrets_all.append(
                np.array([np.mean([Games_data.Expected_regret[x][i] for i in range(N)]) for x in range(T)]))
            e_Regrets_worst.append(np.array([Games_data.Expected_regret[x][ind_worst_e] for x in range(T)]))
            print(f'sim{sim}, run {run}')

    avg_expected_Regrets_all.append(np.mean(e_Regrets_all, 0))
    std_expected_Regrets_all.append(np.std(e_Regrets_all, 0))
    avg_expected_Regrets_worst.append(np.mean(e_Regrets_worst, 0))
    std_expected_Regrets_worst.append(np.std(e_Regrets_worst, 0))
    # rate = Player[0].gamma_t

with open('all.pckl', 'wb') as file:
    pickle.dump(N, file)
    pickle.dump(K, file)
    pickle.dump(N_types, file)
    pickle.dump(avg_expected_Regrets_all, file)
    pickle.dump(std_expected_Regrets_all, file)
    pickle.dump(avg_expected_Regrets_worst, file)
    pickle.dump(std_expected_Regrets_worst, file)
    # pickle.dump(rate, file)

with open('payoffs.pckl', 'wb') as file:
    pickle.dump(A_all, file)
