import numpy as np
from aux_functions import Assign_payoffs, Player_MWU, Player_OPT_MWU, joint_dist
import pickle
from tqdm import tqdm

N = 4  # number of players
K = 6  # number of actions for each player
T = 1000  # time horizon,   should be at least K*log(K) to have a meaningfull EXP3.P algorithm

" Data to be saved (for post processing/plotting) "


class GameData:
    def __init__(self):
        self.Played_actions = []
        self.Mixed_strategies = []
        self.Obtained_payoffs = []
        self.Cum_payoffs = []
        self.Expected_regret = []
        self.Obtained_expected_payoffs = []
        self.Expected_Cum_payoffs = []
        self.expected_payoff_single_actions = []
        self.Regrets = []


def RunGame(N, K, T, A, types):
    Game_data = GameData()

    Player = list(range(N))  # list of all players
    min_payoff = []
    payoffs_range = []
    for i in range(N):
        min_payoff.append(np.array(A[i].min()))
        payoffs_range.append(np.array(A[i].max() - A[i].min()))
        Game_data.Cum_payoffs.append(np.zeros(K))
        Game_data.Expected_Cum_payoffs.append(np.zeros(K))
        Game_data.expected_payoff_single_actions.append([np.zeros(K), np.zeros(K)])

        if types[i] == 'MWU':
            Player[i] = Player_MWU(K, T, min_payoff[i], payoffs_range[i])
        if types[i] == 'OPT_MWU':
            Player[i] = Player_OPT_MWU(K, T, min_payoff[i], payoffs_range[i])

    " Repated Game "
    for t in tqdm(range(T)):
        " Compute played actions "
        Game_data.Played_actions.append([None] * N)  # initialize
        Game_data.Mixed_strategies.append([None] * N)  # initialize
        Game_data.Regrets.append([None] * N)  # initialize
        Game_data.Expected_regret.append([None] * N)  # initialize
        Game_data.Obtained_payoffs.append([None] * N)  # initialize
        Game_data.Obtained_expected_payoffs.append([None] * N)  # initialize
        Game_data.expected_payoff_single_actions.append([None] * N)  # initialize


        for i in range(N):
            Game_data.Mixed_strategies[t][i] = np.array(Player[i].mixed_strategy())
            Game_data.Played_actions[t][i] = np.random.choice(range(K), p=Game_data.Mixed_strategies[t][i])

        " Assign payoffs and compute regrets"

        for i in range(N):

            others_probabilities = Game_data.Mixed_strategies[t].copy()
            others_probabilities.pop(i)
            joint_dis = joint_dist(others_probabilities, K)

            expected_payoff_single_actions = []
            for a in range(K):
                modified_outcome = np.array(Game_data.Played_actions[t])
                modified_outcome[i] = a
                Game_data.Cum_payoffs[i][a] = np.array(
                    Game_data.Cum_payoffs[i][a] + Assign_payoffs(modified_outcome, A[i]))

                action_expected_payoff = np.sum(np.multiply(joint_dis, np.moveaxis(A[i], i, 0)[a, ...]))
                expected_payoff_single_actions.append(action_expected_payoff)
                Game_data.Expected_Cum_payoffs[i][a] += action_expected_payoff

            Game_data.expected_payoff_single_actions[i].pop(0)
            Game_data.expected_payoff_single_actions[i].append(np.array(expected_payoff_single_actions))



            Game_data.Obtained_payoffs[t][i] = Assign_payoffs(Game_data.Played_actions[t], A[i])
            Game_data.Obtained_expected_payoffs[t][i] = np.dot(np.array(expected_payoff_single_actions),
                                                                    np.array(Game_data.Mixed_strategies[t][i]))


            Game_data.Regrets[t][i] = (np.max(Game_data.Cum_payoffs[i]) - sum(
                [Game_data.Obtained_payoffs[x][i] for x in range(t + 1)])) / (t + 1)
            Game_data.Expected_regret[t][i] = (np.max(Game_data.Expected_Cum_payoffs[i]) - sum(
                [Game_data.Obtained_expected_payoffs[x][i] for x in range(t + 1)])) / (t + 1)

        " Update players next mixed strategy "
        for i in range(N):
            if Player[i].type == "MWU":
                tmp_idx = Game_data.Played_actions[t].copy()
                tmp_idx.pop(i)
                Player[i].Update(Game_data.expected_payoff_single_actions[i][1])

            if Player[i].type == "OPT_MWU":
                Player[i].Update(Game_data.expected_payoff_single_actions[i][1], Game_data.expected_payoff_single_actions[i][0])

        Game_data.A = A
    return Game_data, Player


" --------------------------------- Begin Simulations --------------------------------- "

Runs = 5

N_types = [['MWU'] * N, ['OPT_MWU'] * N]

avg_Regrets_all = []
std_Regrets_all = []
avg_expected_Regrets_all = []
std_expected_Regrets_all = []


A_sample = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
# A_sample = np.random.random(size=[K] * N)

for i in range(len(N_types)):
    Regrets_all = [None] * Runs
    e_Regrets_all = [None] * Runs
    for run in range(Runs):
        # A = []
        # A.append(A_sample)
        # A.append(-A_sample)
        if run % 3 == 0:
            A = []
            for j in range(N):
                A.append(np.random.random(size=[K] * N))

        Games_data, Player = RunGame(N, K, T, A, N_types[i])

        ind_worst = 0
        ind_worst_e = 0
        s = 0
        s_e = 0
        for ind in range(N):
            tmp = sum([Games_data.Regrets[t][ind_worst] for t in range(T)])
            tmp_e = sum([Games_data.Expected_regret[t][ind_worst] for t in range(T)])
            if tmp > s:
                ind_worst = ind
            if tmp_e > s_e:
                ind_worst_e = ind
        # Regrets_all[run] = np.array([np.mean([Games_data.Regrets[x][i] for i in range(N)]) for x in range(T)])
        # e_Regrets_all[run] = np.array([np.mean([Games_data.Expected_regret[x][i] for i in range(N)]) for x in range(T)])
        Regrets_all[run] = np.array([Games_data.Regrets[x][i] for x in range(T)])
        e_Regrets_all[run] = np.array([Games_data.Expected_regret[x][i] for x in range(T)])
    avg_Regrets_all.append(np.mean(Regrets_all, 0))
    std_Regrets_all.append(np.std(Regrets_all, 0))
    avg_expected_Regrets_all.append(np.mean(e_Regrets_all, 0))
    std_expected_Regrets_all.append(np.std(e_Regrets_all, 0))


with open('all.pckl', 'wb') as file:
    pickle.dump(N_types, file)
    pickle.dump(avg_Regrets_all, file)
    pickle.dump(std_Regrets_all, file)
    pickle.dump(avg_expected_Regrets_all, file)
    pickle.dump(std_expected_Regrets_all, file)


