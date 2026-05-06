"""Frame matching and error metrics shared by experiment scripts."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_cost_matrix(U, V):
    """Pairwise sign-invariant squared distances between rows of U and V."""
    n_u, n_columns = U.shape
    n_v, _ = V.shape
    if n_columns != V.shape[1]:
        raise ValueError('U and V must have the same number of columns.')

    cost_matrix = np.zeros((n_u, n_v))
    for i in range(n_u):
        for j in range(n_v):
            cost_matrix[i, j] = min(
                np.linalg.norm(U[i] - V[j]),
                np.linalg.norm(U[i] + V[j]),
            ) ** 2
    return cost_matrix


def approx_metric_matching(estimated_matrix, true_matrix_axis):
    cost = compute_cost_matrix(true_matrix_axis, estimated_matrix)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()


def normalized_frame_error(estimated_matrix, true_matrix_axis):
    assignment_error = approx_metric_matching(estimated_matrix, true_matrix_axis)
    normalizer = np.linalg.norm(true_matrix_axis, ord='fro') ** 2
    return assignment_error, assignment_error / normalizer
