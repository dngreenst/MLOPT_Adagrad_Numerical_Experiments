import numpy as np


class Evaluator:
    def __init__(self, regret_of_best_in_hindsight: float, iteration_num):
        self.iteration_num = iteration_num
        self.running_loss = np.zeros(self.iteration_num)
        self.running_loss_sum = np.zeros_like(self.running_loss)

    def evaluate(self, curr_iter, curr_loss):
        self.running_loss[curr_iter] = curr_loss
        if curr_iter > 0:
            self.running_loss_sum[curr_iter] += self.running_loss_sum[curr_iter - 1]
        self.running_loss_sum[curr_iter] += curr_loss
