
import numpy as np

        
class Player_MWU:  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self,K,T,min_payoff,payoffs_range):
        self.type = "MWU"
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.T  = T
        # self.gamma_t = np.sqrt(8*np.log(K)/T)
        self.gamma_t = 0.01
        
    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def Update(self,payoffs):
        # payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        # payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        # payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply( self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights/np.sum(self.weights) # To avoid numerical errors when the weights become too small


class Player_OPT_MWU:  # Hedge algorithm (Freund and Schapire. 1997)
    def __init__(self, K, T, min_payoff, payoffs_range):
        self.type = "OPT_MWU"
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.T = T
        # self.gamma_t = 1/(2*2* np.log(T)**4)
        self.gamma_t = 0.01

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def normalize(self, payoffs):
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range) * np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / self.payoffs_range)
        return payoffs_scaled

    def Update(self, payoffs_t, payoffs_t_1):
        loss_t = np.ones(self.K)- payoffs_t
        loss_t_1 = np.ones(self.K)- payoffs_t_1
        losses = 2*loss_t - loss_t_1
        #payoffs_next = self.normalize(payoffs_next)
        #losses = np.ones(self.K) - np.array(payoffs_next)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.gamma_t, -losses)))
        self.weights = self.weights / np.sum(
            self.weights)  # To avoid numerical errors when the weights become too small



def Assign_payoffs(outcome , payoff_matrix):
    return payoff_matrix[tuple(outcome)]

def joint_dist(Mixed_strategies, K):
    if len(Mixed_strategies) == 1:
        return Mixed_strategies[0]
    x = Mixed_strategies.pop(0)
    joint_dis = []
    tmp_joint = joint_dist(Mixed_strategies, K)
    for j in range(K):
        joint_dis.append(x[j] * tmp_joint)
    return np.array(joint_dis)




