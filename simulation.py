import numpy as np
from typing import List

import Params
from evaluator import Evaluator
from hinge_loss_function import HingeLossFunction


def _projection_onto_L2_ball(radius: float, x: np.array) -> np.array:
    norm_x = np.linalg.norm(x)
    if norm_x > radius:
        return (x / norm_x) * radius
    return x


def simulate_adagrad(loss_functions: List[HingeLossFunction], dimension: int, evaluator: Evaluator):
    curr_x = np.zeros(dimension)
    epsilon = 1e-3
    s = np.zeros(dimension)
    learning_rate = 100.0
    radius = Params.RADIUS

    for iteration, loss_function in enumerate(loss_functions):
        gradient = loss_function.gradient(curr_x)
        s += np.square(gradient)
        curr_x = _projection_onto_L2_ball(radius=radius,
                                          x=curr_x - learning_rate * (gradient / (np.sqrt(s) + np.full_like(s, epsilon))))
        evaluator.evaluate(curr_iter=iteration, curr_loss=loss_function.loss(curr_x))


def simulate_online_gradient_descent(loss_functions: List[HingeLossFunction], dimension: int, evaluator: Evaluator):
    curr_x = np.zeros(dimension)
    G = np.sqrt(dimension)  # all the features are normalized to [0, 1]
    radius = Params.RADIUS
    D = 2 * radius
    t = 1

    for iteration, loss_function in enumerate(loss_functions):
        gradient = loss_function.gradient(curr_x)
        step_size = D / (G * np.sqrt(t))
        t += 1
        curr_x = curr_x - step_size * gradient
        evaluator.evaluate(curr_iter=iteration, curr_loss=loss_function.loss(curr_x))


def _solve_ftrl_sub_problem(accumulated_subgradients: np.array, eta: float, radius) -> np.array:
    # For the L2 regularization, the sub problem is strongly convex,
    # and its stationary point can be calculated analytically
    solution_without_projection = (- eta / 2) * accumulated_subgradients
    if np.linalg.norm(solution_without_projection) <= radius:
        return solution_without_projection
    return (- radius / np.linalg.norm(accumulated_subgradients)) * accumulated_subgradients


def simulate_follow_the_regularized_leader(loss_functions: List[HingeLossFunction], dimension: int, evaluator: Evaluator):
    # regularization is L2 norm
    curr_x = np.zeros(dimension)

    accumulated_subgradients = np.zeros(dimension)
    G = np.sqrt(dimension)  # all the features are normalized to [0, 1]
    radius = Params.RADIUS
    D = 2 * radius
    t = 1

    for iteration, loss_function in enumerate(loss_functions):
        gradient = loss_function.gradient(curr_x)
        accumulated_subgradients += gradient
        eta = D / (G * np.sqrt(2*t))
        t += 1
        curr_x = _solve_ftrl_sub_problem(accumulated_subgradients=accumulated_subgradients, eta=eta, radius=radius)
        evaluator.evaluate(curr_iter=iteration, curr_loss=loss_function.loss(curr_x))
