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
    def __init__(self, K, T, N, sigma, all_action_profiles):
        self.type = "GPMW"
        self.K = K
        self.T = T
        self.N = N
        self.weights = np.ones(K)
        self.UCB_Matrix = np.zeros([K] * N)
        self.gamma_t = 1
        self.beta_t = 2.0
        self.sigma = sigma
        self.kernel = RBF()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=sigma ** 2)
        self.mean_matrix = 0 * np.ones(K ** N)
        self.var_matrix = np.zeros(K ** N)
        self.all_action_profiles = all_action_profiles
        self.kernel_matrix = RBF.__call__(RBF(), all_action_profiles)
        for idx in range(K**N):
            self.var_matrix[idx] = np.array(self.kernel_matrix[idx,idx])
        self.std_matrix = np.sqrt(self.var_matrix)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def GP_update(self, history_actions, history_payoffs, t):
        if 0:
            self.gpr.fit(history_actions, history_payoffs)
            # params = self.gpr.kernel_.get_params()
            # k1 = params.get('k1__constant_value')
            # k2 = params.get('k2__length_scale')
            mean_prediction, std_prediction = self.gpr.predict(self.all_action_profiles, return_std=True)
            mean_prediction = np.array(mean_prediction).reshape([self.K] * self.N)
            std_prediction = np.array(std_prediction).reshape([self.K] * self.N)
            self.UCB_Matrix = mean_prediction + self.beta_t * std_prediction
        if 1:
            self.UCB_Matrix = np.ndarray.flatten(self.UCB_Matrix)
            mean_matrix_prev = np.array(self.mean_matrix)
            var_matrix_prev = np.array(self.var_matrix)
            kernel_matrix_prev = np.array(self.kernel_matrix)

            mean_matrix = np.zeros(self.K ** self.N)
            var_matrix = np.zeros(self.K ** self.N)

            idx_t = np.squeeze(np.where(np.all(self.all_action_profiles == history_actions[-1], axis=1)))
            for idx in range(self.all_action_profiles.shape[0]):
                mean_matrix[idx] = mean_matrix_prev[idx] + (kernel_matrix_prev[idx, idx_t] /
                                                            (self.sigma ** 2 + var_matrix_prev[idx_t])) * (
                                               history_payoffs[-1] - mean_matrix_prev[idx_t])
                var_matrix[idx] = var_matrix_prev[idx] - (kernel_matrix_prev[idx, idx_t] ** 2) / (
                        self.sigma ** 2 + var_matrix_prev[idx_t])

                self.UCB_Matrix[idx] = mean_matrix[idx] + self.beta_t * np.sqrt(var_matrix[idx])

            kernel_matrix = kernel_matrix_prev - np.outer(kernel_matrix_prev[:, idx_t], kernel_matrix_prev[idx_t, :]) * (
                    1 / (self.sigma ** 2 + var_matrix_prev[idx_t]))

            self.var_matrix = var_matrix
            self.std_matrix = np.sqrt(var_matrix)
            self.mean_matrix = mean_matrix
            self.kernel_matrix = kernel_matrix
            self.UCB_Matrix = self.UCB_Matrix.reshape([self.K] * self.N)

    def Update(self, payoffs, t):
        # self.gamma_t = np.sqrt(8*np.log(K)/t)
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)


class Player_OPT_GPMW:
    def __init__(self, K, T, N, sigma, all_action_profiles):
        self.type = "OPT_GPMW"
        self.K = K
        self.T = T
        self.N = N
        self.weights = np.ones(K)
        self.UCB_Matrix = np.zeros([K] * N)
        self.gamma_t = 1
        self.beta_t = 2.0
        self.sigma = sigma
        self.kernel = RBF()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=sigma ** 2)
        self.mean_matrix = 0 * np.ones(K ** N)
        self.var_matrix = np.zeros(K ** N)
        self.all_action_profiles = all_action_profiles
        self.kernel_matrix = RBF.__call__(RBF(), all_action_profiles)
        for idx in range(K**N):
            self.var_matrix[idx] = np.array(self.kernel_matrix[idx,idx])
        self.std_matrix = np.sqrt(self.var_matrix)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def GP_update(self, history_actions, history_payoffs):
        if 0:
            self.gpr.fit(history_actions, history_payoffs)
            mean_prediction, std_prediction = self.gpr.predict(self.all_action_profiles, return_std=True)
            mean_prediction = np.array(mean_prediction).reshape([self.K] * self.N)
            std_prediction = np.array(std_prediction).reshape([self.K] * self.N)
            self.UCB_Matrix = mean_prediction + self.beta_t * std_prediction
        if 1:
            self.UCB_Matrix = np.ndarray.flatten(self.UCB_Matrix)
            mean_matrix_prev = np.array(self.mean_matrix)
            var_matrix_prev = np.array(self.var_matrix)
            kernel_matrix_prev = np.array(self.kernel_matrix)

            mean_matrix = np.zeros(self.K ** self.N)
            var_matrix = np.zeros(self.K ** self.N)

            idx_t = np.squeeze(np.where(np.all(self.all_action_profiles == history_actions[-1], axis=1)))
            for idx in range(self.all_action_profiles.shape[0]):
                mean_matrix[idx] = mean_matrix_prev[idx] + (kernel_matrix_prev[idx, idx_t] /
                                                            (self.sigma ** 2 + var_matrix_prev[idx_t])) * (
                                               history_payoffs[-1] - mean_matrix_prev[idx_t])
                var_matrix[idx] = var_matrix_prev[idx] - (kernel_matrix_prev[idx, idx_t] ** 2) / (
                        self.sigma ** 2 + var_matrix_prev[idx_t])

                self.UCB_Matrix[idx] = mean_matrix[idx] + self.beta_t * np.sqrt(var_matrix[idx])

            kernel_matrix = kernel_matrix_prev - np.outer(kernel_matrix_prev[:, idx_t], kernel_matrix_prev[idx_t, :]) * (
                    1 / (self.sigma ** 2 + var_matrix_prev[idx_t]))

            self.var_matrix = var_matrix
            self.std_matrix = np.sqrt(var_matrix)
            self.mean_matrix = mean_matrix
            self.kernel_matrix = kernel_matrix
            self.UCB_Matrix = self.UCB_Matrix.reshape([self.K] * self.N)

    def Update(self, payoffs_t, payoffs_t_1, t, N):
        # self.gamma_t = np.sqrt(8*np.log(K)/t)
        loss_t = np.ones(self.K) - payoffs_t
        loss_t_1 = np.ones(self.K) - payoffs_t_1
        losses = 2 * loss_t - loss_t_1
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)  # To avoid numerical errors when the weights become too small


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


def get_combinations(params):
    all_indices = np.indices(params)
    return np.moveaxis(all_indices, 0, -1).reshape(-1, len(params))


