import matplotlib.pyplot as plt

from evaluator import Evaluator


def display_cumulative_losses_and_regret(adagard: Evaluator,
                                         ogd: Evaluator,
                                         ftrl: Evaluator,
                                         regret_in_hindsight: float):
    plt.plot(adagard.running_loss_sum, label='AdaGrad')
    plt.plot(ogd.running_loss_sum, label='Online Gradient Descent')
    plt.plot(ftrl.running_loss_sum, label='Follow the Regularized Leader')
    plt.title(f'Cumulative Loss Per Iteration')
    plt.xlabel('iterations')
    plt.ylabel('accumulated loss')
    plt.legend()
    plt.show()

    regret_scores = [adagard.running_loss_sum[-1] - regret_in_hindsight,
                     ogd.running_loss_sum[-1] - regret_in_hindsight,
                     ftrl.running_loss_sum[-1] - regret_in_hindsight]
    labels = ['AdaGrad',
              'Online Gradient Descent',
              'Follow the Regularized Leader']

    plt.bar(labels, regret_scores)
    plt.xlabel("Algorithm")
    plt.ylabel("Regret")
    plt.title("Regret in Hindsight")
    plt.show()
