import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class Parent_MWU:
    def __init__(self, K, T, N, min_payoff, max_payoff):
        self.K = K
        self.weights = np.ones(K)
        self.T = T
        self.N = N
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        # learning rate from Fast Convergence of Regularized Learning in Games [Syrgkanis et al. 2015]
        # self.gamma_t = 0.1
        # This learning rate works better in practice
        self.gamma_t = 1

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def semi_Update(self, losses):
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)  # To avoid numerical errors when the weights become too small


class Player_MWU(Parent_MWU):  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self, K, T, N, min_payoff, max_payoff):
        super().__init__(K, T, N, min_payoff, max_payoff)
        self.type = "MWU"
        # adversarial hedge learning rate
        # self.gamma_t = np.sqrt(np.log(self.K) / self.T)

    def Update(self, payoffs):
        payoffs = normalize(payoffs, self.min_payoff, self.max_payoff)
        losses = np.ones(self.K) - np.array(payoffs)
        # print('vanilla', losses)
        super().semi_Update(losses)


class Player_OPT_MWU(Parent_MWU):  # Optimistic Hedge algorithm
    def __init__(self, K, T, N, min_payoff, max_payoff):
        super().__init__(K, T, N, min_payoff, max_payoff)
        self.type = "OPT_MWU"
        # learning rate from Near-Optimal No-Regret Learning in General Games [Daskalakis et al. 2021]
        # self.gamma_t = 1 / (2 * self.N * np.log(self.T + 2) ** 4)

    def Update(self, payoffs_t, payoffs_t_1):
        payoffs_t = normalize(payoffs_t, self.min_payoff, self.max_payoff)
        payoffs_t_1 = normalize(payoffs_t_1, self.min_payoff, self.max_payoff)
        loss_t = np.ones(self.K) - payoffs_t
        loss_t_1 = np.ones(self.K) - payoffs_t_1
        losses = 2 * loss_t - loss_t_1
        # print('OPT', losses)
        super().semi_Update(losses)


class Parent_GPMW:
    def __init__(self, K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, payoff_matrix,
                 kernel_optimization=False):
        self.K = K
        self.T = T
        self.N = N
        self.weights = np.ones(K)
        self.UCB_Matrix = np.zeros([K] * N)
        self.gamma_t = 0.1
        # beta from No-Regret Learning in Unknown Games with Correlated Payoffs [Sessa et al. 2019]
        self.beta_t = 2.0
        self.sigma = sigma
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff

        self.kernel = RBF()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=sigma ** 2)
        if kernel_optimization:
            self.gpr.fit(all_action_profiles, np.ndarray.flatten(payoff_matrix))
            self.kernel = self.gpr.kernel_
        else:
            self.gpr.optimizer = None

        self.kernel_matrix = RBF.__call__(self.kernel, all_action_profiles)
        self.mean_matrix = 0 * np.ones(K ** N)
        self.var_matrix = np.zeros(K ** N)
        self.all_action_profiles = all_action_profiles
        for idx in range(K ** N):
            self.var_matrix[idx] = np.array(self.kernel_matrix[idx, idx])
        self.std_matrix = np.sqrt(self.var_matrix)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def GP_update(self, history_actions, history_payoffs, recursive_update=True):
        if not recursive_update:
            self.gpr.fit(history_actions, history_payoffs)
            # params = self.gpr.kernel_.get_params()
            # k1 = params.get('k1__constant_value')
            # k2 = params.get('k2__length_scale')
            mean_prediction, std_prediction = self.gpr.predict(self.all_action_profiles, return_std=True)
            mean_prediction = np.array(mean_prediction).reshape([self.K] * self.N)
            std_prediction = np.array(std_prediction).reshape([self.K] * self.N)
            self.UCB_Matrix = mean_prediction + self.beta_t * std_prediction

        else:
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

            kernel_matrix = kernel_matrix_prev - np.outer(kernel_matrix_prev[:, idx_t],
                                                          kernel_matrix_prev[idx_t, :]) * (
                                    1 / (self.sigma ** 2 + var_matrix_prev[idx_t]))

            self.var_matrix = var_matrix
            self.std_matrix = np.sqrt(var_matrix)
            self.mean_matrix = mean_matrix
            self.kernel_matrix = kernel_matrix
            self.UCB_Matrix = self.UCB_Matrix.reshape([self.K] * self.N)

    def semi_Update(self, losses):
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)


class Player_GPMW(Parent_GPMW):
    def __init__(self, K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, payoff_matrix, kernel_optimization=False):
        super().__init__(K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, payoff_matrix, kernel_optimization)
        self.type = "GPMW"
        # beta from No-Regret Learning in Unknown Games with Correlated Payoffs [Sessa et al. 2019]
        self.gamma_t = np.sqrt(8*np.log(self.K)/self.T)

    def Update(self, payoffs):
        payoffs = normalize(payoffs, self.min_payoff, self.max_payoff)
        losses = np.ones(self.K) - np.array(payoffs)
        # print('Vanilla', losses)
        super().semi_Update(losses)


class Player_OPT_GPMW(Parent_GPMW):
    def __init__(self, K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, payoff_matrix, kernel_optimization=False):
        super().__init__(K, T, N, min_payoff, max_payoff, sigma, all_action_profiles, payoff_matrix, kernel_optimization)
        self.type = "OPT_GPMW"
        # beta from No-Regret Learning in Unknown Games with Correlated Payoffs [Sessa et al. 2019]
        self.gamma_t = np.sqrt(8*np.log(self.K)/self.T)

    def Update(self, payoffs_t, payoffs_t_1):
        payoffs_t = normalize(payoffs_t, self.min_payoff, self.max_payoff)
        payoffs_t_1 = normalize(payoffs_t_1, self.min_payoff, self.max_payoff)
        loss_t = np.ones(self.K) - payoffs_t
        loss_t_1 = np.ones(self.K) - payoffs_t_1
        losses = 2 * loss_t - loss_t_1
        # print('OPT', losses)
        super().semi_Update(losses)

# EXP3.P algorithm (Auer et al. 2002)
class Parent_EXP3:
    def __init__(self, K, T, min_payoff, max_payoff):
        self.K = K
        self.T = T
        self.weights = np.ones(K)
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff
        self.rewards_est = np.zeros(K)
        # params from Theorem 3.2 of Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems
        # [Bubeck and Cesa-Bianchi 2012]
        self.gamma = 1.05 * np.sqrt(np.log(self.K) * self.K / self.T)
        delta = 0.01
        self.beta = np.sqrt(np.log(self.K * (1 / delta)) / (self.T * self.K))
        self.eta = 0.95 * np.sqrt(np.log(self.K) / (self.T * self.K))
        assert self.beta > 0 and self.beta < 1 and self.gamma > 0 and self.gamma < 1

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def semi_Update(self, played_a, payoff, recency_bias):
        payoff = normalize(payoff, self.min_payoff, self.max_payoff)
        prob = self.weights[played_a] / np.sum(self.weights)
        self.rewards_est = self.rewards_est + recency_bias * self.beta * np.divide(np.ones(self.K), self.weights / np.sum(self.weights))
        self.rewards_est[played_a] += recency_bias*(payoff / prob)
        self.weights = np.exp(np.multiply(self.eta, self.rewards_est))
        self.weights = self.weights / np.sum(self.weights)
        self.weights = (1 - self.gamma) * self.weights + self.gamma / self.K * np.ones(self.K)


class Player_EXP3(Parent_EXP3):
    def __init__(self, K, T, min_payoff, max_payoff):
        super().__init__(K, T, min_payoff, max_payoff)
        self.type = 'EXP3'

    def Update(self, played_a, payoff):
        super().semi_Update(played_a, payoff, 1)



class Player_OPT_EXP3(Parent_EXP3):
    def __init__(self, K, T, min_payoff, max_payoff):
        super().__init__(K, T, min_payoff, max_payoff)
        self.type = 'OPT_EXP3'

    def Update(self, played_a, payoff):
        super().semi_Update(played_a, payoff, 2)




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


def normalize_util(payoffs, min_payoff, max_payoff):
    if min_payoff == max_payoff:
        return payoffs
    payoff_range = max_payoff - min_payoff
    payoffs = np.maximum(payoffs, min_payoff)
    payoffs = np.minimum(payoffs, max_payoff)
    payoffs_scaled = (payoffs - min_payoff) / payoff_range
    return payoffs_scaled


normalize = np.vectorize(normalize_util)
