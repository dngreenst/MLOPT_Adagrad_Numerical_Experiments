from typing import List

from hinge_loss_function import HingeLossFunction

import scipy.optimize
import numpy as np

# def hinge_loss_gradient_descent(loss_functions: List[HingeLossFunction]):
#     base_step_size = 0.01
#     empirical_G = 300000000
#     empirical_D = 2
#     epsilon = 1e-5
#     curr_x = np.zeros(31)
#     t = 1
#
#     s = np.zeros(31)
#
#     curr_gradient_size = np.infty
#     while curr_gradient_size > epsilon:
#         gradient = (lambda x: np.sum(hinge_function.gradient(x) for hinge_function in loss_functions))(curr_x)
#         curr_gradient_size = np.linalg.norm(gradient)
#         step_size = base_step_size / (np.sqrt(t))
#         t += 1
#         s[:] += np.square(gradient)
#         curr_x = curr_x - base_step_size *( gradient / (np.sqrt(s) + 1e-5))
#         if t % 1000 == 0:
#             print(f't={t}, curr_gradient_size={curr_gradient_size}')
#
#     return curr_x

def calculate_regret_in_hindsight(loss_functions: List[HingeLossFunction]):

    cumulative_loss = lambda x: sum(hinge_function.loss(x) for hinge_function in loss_functions)

    gradient = lambda x: np.sum(hinge_function.gradient(x) for hinge_function in loss_functions)

    options = {"disp": False, "maxiter": 50000}

    # res = scipy.optimize.minimize(fun=cumulative_loss, x0=np.zeros(31), method='BFGS', jac=gradient)
    res = scipy.optimize.minimize(fun=lambda x: sum(hinge_function.loss(x) for hinge_function in loss_functions),
                                  x0=np.zeros(31), method='Nelder-Mead', options=options)

    print(f'Regret of best in hindsight: {res.fun}')
    return res.fun


