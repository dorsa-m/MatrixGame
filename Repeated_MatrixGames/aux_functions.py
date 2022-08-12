import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class Player_MWU:  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self, K, T):
        self.type = "MWU"
        self.K = K
        self.weights = np.ones(K)
        # self.weights = np.random.randint(10, size = K)
        self.T = T
        self.gamma_t = 1

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def Update(self, payoffs, t):
        # self.gamma_t = 1/np.sqrt(t+1)
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)  # To avoid numerical errors when the weights become too small


class Player_OPT_MWU:  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self, K, T):
        self.type = "OPT_MWU"
        self.K = K
        self.weights = np.ones(K)
        # self.weights = np.random.randint(10, size = K)
        self.T = T
        self.gamma_t = 1

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def Update(self, payoffs_t, payoffs_t_1, t, N):
        # self.gamma_t = 1 / (N * np.log(t + 2) ** 4)
        loss_t = np.ones(self.K) - payoffs_t
        loss_t_1 = np.ones(self.K) - payoffs_t_1
        losses = 2 * loss_t - loss_t_1
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)  # To avoid numerical errors when the weights become too small


class Player_GPMW:
    def __init__(self, K, T, sigma):
        self.type = "GPMW"
        self.K = K
        self.T = T
        self.weights = np.ones(K)
        self.UCB = np.zeros(K)
        self.gamma_t = 1
        self.beta_t = 2.0
        self.kernel = RBF()
        self.gpr = GaussianProcessRegressor(kernel = self.kernel, n_restarts_optimizer=10, alpha=sigma**2)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def GP_update(self, history_actions, history_payoffs, X_predicts):
        self.gpr.fit(history_actions, history_payoffs)
        mean_prediction, std_prediction = self.gpr.predict(X_predicts, return_std=True)
        self.UCB = mean_prediction + self.beta_t*std_prediction

    def Update(self, payoffs, t):
        # self.gamma_t = np.sqrt(8*np.log(K)/t)
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)


def Assign_payoffs(outcome, payoff_matrix):
    return np.squeeze(payoff_matrix[tuple(outcome)])


def joint_dist(Mixed_strategies, K):
    if len(Mixed_strategies) == 1:
        return Mixed_strategies[0]
    x = Mixed_strategies.pop(0)
    joint_dis = []
    tmp_joint = joint_dist(Mixed_strategies, K)
    for j in range(K):
        joint_dis.append(x[j] * tmp_joint)
    return np.array(joint_dis)
