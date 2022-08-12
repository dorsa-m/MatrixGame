import numpy as np
import pickle
from tqdm import tqdm
from aux_functions import Assign_payoffs, Player_MWU, Player_GPMW, Player_OPT_MWU, joint_dist

N = 3  # number of players
K = 3  # number of actions for each player
T = 100  # time horizon
sigma = 1

" Data to be saved (for post processing/plotting) "


class GameData:
    def __init__(self):
        self.Played_actions = []
        self.Mixed_strategies = []
        self.Obtained_payoffs = []

        self.Expected_regret = []
        self.Expected_Obtained_payoffs = []
        self.Expected_Cum_payoffs = []
        self.Expected_payoff_single_actions = []


def RunGame(N, K, T, A, sigma, types):
    noises = np.random.normal(0, sigma, T)

    Game_data = GameData()

    Player = list(range(N))  # list of all players
    for i in range(N):
        Game_data.Expected_Cum_payoffs.append(np.zeros(K))
        Game_data.Expected_payoff_single_actions.append([np.zeros(K), np.zeros(
            K)])  # for each player stores total expected payoff of each actions for (2) current and prev rounds

        if types[i] == 'MWU':
            Player[i] = Player_MWU(K, T)
        if types[i] == 'OPT_MWU':
            Player[i] = Player_OPT_MWU(K, T)
        if types[i] == 'GPMW':
            Player[i] = Player_GPMW(K, T, sigma)

    " Repated Game "
    for t in tqdm(range(T)):
        " Compute played actions "
        Game_data.Played_actions.append([None] * N)  # initialize
        Game_data.Mixed_strategies.append([None] * N)  # initialize
        Game_data.Expected_regret.append([None] * N)  # initialize
        Game_data.Expected_Obtained_payoffs.append([None] * N)  # initialize
        Game_data.Obtained_payoffs.append([None] * N)  # initialize

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
            for a in range(K):
                action_expected_payoff = np.sum(np.multiply(joint_dis, np.moveaxis(A[i], i, 0)[a, ...]))
                expected_payoff_all_actions.append(action_expected_payoff)
                Game_data.Expected_Cum_payoffs[i][a] += action_expected_payoff

            Game_data.Expected_payoff_single_actions[i].pop(0)
            Game_data.Expected_payoff_single_actions[i].append(np.array(expected_payoff_all_actions))

            Game_data.Expected_Obtained_payoffs[t][i] = np.dot(np.array(expected_payoff_all_actions),
                                                               np.array(Game_data.Mixed_strategies[t][i]))

            Game_data.Expected_regret[t][i] = (np.max(Game_data.Expected_Cum_payoffs[i]) - sum(
                [Game_data.Expected_Obtained_payoffs[x][i] for x in range(t + 1)]))

        " Update players next mixed strategy "
        for i in range(N):
            if Player[i].type == "MWU":
                Player[i].Update(Game_data.Expected_payoff_single_actions[i][1], t)

            if Player[i].type == "OPT_MWU":
                Player[i].Update(Game_data.Expected_payoff_single_actions[i][1],
                                 Game_data.Expected_payoff_single_actions[i][0], t, N)

            if Player[i].type == "GPMW":
                history_actions = np.array([Game_data.Played_actions[x] for x in range(t + 1)])
                history_payoffs = np.array([Game_data.Obtained_payoffs[x][i] + noises[x] for x in range(t + 1)])
                '''remark! for expected payoffs you should do regression for all K^N points and calculate the expected
                result and pass it to the update function. Find an efficient way to do that'''
                all_possible_profiles = []
                for a in range(K):
                    modified_outcome = np.array(Game_data.Played_actions[t])
                    modified_outcome[i] = a
                    all_possible_profiles.append(modified_outcome)
                all_possible_profiles = np.array(all_possible_profiles)
                Player[i].GP_update(history_actions, history_payoffs, all_possible_profiles)
                payoffs = Player[i].UCB
                Player[i].Update(payoffs, t)

    return Game_data, Player


" --------------------------------- Begin Simulations --------------------------------- "

Runs = 9
# np.random.seed(10)

# N_types = [['MWU'] * N, ['OPT_MWU'] * N, ['GPMW'] * N]
N_types = [['GPMW'] * N]
avg_expected_Regrets_all = []
std_expected_Regrets_all = []
avg_expected_Regrets_worst = []
std_expected_Regrets_worst = []

A_all = []
for run in range(Runs):
    if run % 3 == 0:
        A = []
        for j in range(N):
            A.append(np.random.random(size=[K] * N))
    A_all.append(A)

for i in range(len(N_types)):
    e_Regrets_all = [None] * Runs
    e_Regrets_worst = [None] * Runs
    for run in range(Runs):
        Games_data, Player = RunGame(N, K, T, A_all[run], sigma, N_types[i])

        # finding player with max regret
        ind_worst_e = 0
        s_e = 0
        for ind in range(N):
            if Games_data.Expected_regret[T - 1][ind] > s_e:
                ind_worst_e = ind
                s_e = Games_data.Expected_regret[T - 1][ind]
        e_Regrets_all[run] = np.array([np.mean([Games_data.Expected_regret[x][i] for i in range(N)]) for x in range(T)])
        e_Regrets_worst[run] = np.array([Games_data.Expected_regret[x][ind_worst_e] for x in range(T)])

    avg_expected_Regrets_all.append(np.mean(e_Regrets_all, 0))
    std_expected_Regrets_all.append(np.std(e_Regrets_all, 0))
    avg_expected_Regrets_worst.append(np.mean(e_Regrets_worst, 0))
    std_expected_Regrets_worst.append(np.std(e_Regrets_worst, 0))
    rate = Player[0].gamma_t

with open('all.pckl', 'wb') as file:
    pickle.dump(N_types, file)
    pickle.dump(avg_expected_Regrets_all, file)
    pickle.dump(std_expected_Regrets_all, file)
    pickle.dump(avg_expected_Regrets_worst, file)
    pickle.dump(std_expected_Regrets_worst, file)
    pickle.dump(rate, file)
