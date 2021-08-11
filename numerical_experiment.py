import os

import numpy as np

from calculate_regret_in_hindsight import calculate_regret_in_hindsight
from data_reader import DataReader
from display import display_cumulative_losses_and_regret
from evaluator import Evaluator
from loss_function_generator import generate_loss_functions
from scale_features import scale_features
from simulation import simulate_adagrad, simulate_online_gradient_descent, simulate_follow_the_regularized_leader


def main():
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    wpbc = os.path.join(curr_directory, 'wpbc.data')

    raw_data = DataReader.read(wpbc, delimiter=',')

    labels = np.array([raw_data[i][1] for i in range(len(raw_data))])

    new_data_list = []

    for data_entry in raw_data:
        tmp = data_entry[3:]
        new_data_list.append(tmp)

    feature_vectors = np.array(new_data_list)
    feature_vectors = scale_features(feature_vectors)

    loss_functions = generate_loss_functions(labels=labels, features=feature_vectors)

    regret_in_hindsight = calculate_regret_in_hindsight(loss_functions)

    evaluator_adagrad = Evaluator(regret_of_best_in_hindsight=regret_in_hindsight, iteration_num=len(feature_vectors))
    evaluator_ogd = Evaluator(regret_of_best_in_hindsight=regret_in_hindsight, iteration_num=len(feature_vectors))
    evaluator_ftrl = Evaluator(regret_of_best_in_hindsight=regret_in_hindsight, iteration_num=len(feature_vectors))

    simulate_adagrad(loss_functions, dimension=len(feature_vectors[0]), evaluator=evaluator_adagrad)
    simulate_online_gradient_descent(loss_functions, dimension=len(feature_vectors[0]), evaluator=evaluator_ogd)
    simulate_follow_the_regularized_leader(loss_functions, dimension=len(feature_vectors[0]), evaluator=evaluator_ftrl)

    display_cumulative_losses_and_regret(adagard=evaluator_adagrad, ogd=evaluator_ogd, ftrl=evaluator_ftrl)

if __name__ == "__main__":
    # execute only if run as a script
    main()
