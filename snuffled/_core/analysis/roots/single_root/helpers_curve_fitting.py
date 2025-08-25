import numpy as np

from snuffled._core.compatibility import numba


# =================================================================================================
#  Optimization
# =================================================================================================
@numba.njit
def fit_curve_brute_force(
    x: np.ndarray,
    fx: np.ndarray,
    range_c_exp: tuple[float, float],
    range_c_step: tuple[float, float],
    exp_sign: float,
    n_grid: int,
    c_reg: float,
    tol_c0: float,
    tol_c1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a curve of the following form to a list of (x, fx)-tuples:

        f(x) = c_step + (1-c_step)*(x**c_exp)

    by minimizing the L1 cost (mean of absolute deviations) over a rectangular grid of (c_exp, c_step)-values.
    The grid is (n_grid x n_grid) values and is linear in c_step and logarithmic in c_exp.

    This function does not return a single optimal (c_exp, c_step)-value, but returns all tuples in the search grid
    whose cost is 'close enough' (as determined by tol_c0, tol_c1) to the optimal cost.

    :param x: (n,)-sized numpy array containing x values, with each x>0
    :param fx: (n,)-sized numpy array containing corresponding f(x) values
    :param range_c_exp:  (c_exp_min, c_exp_max)    range of c_exp values  (>0)
    :param range_c_step: (c_step_min, c_step_max)  range of c_step values
    :param exp_sign: (float) sign of c_exp, i.e. if exp_sign=-1, we will search over range [-c_exp_max, -c_exp_min].
    :param n_grid: (int) number of grid values along each dimension
    :param c_reg: (float) regularization coefficient that helps favour c_exp=1.0, c_step=0.0  (e.g. 1e-3)
    :param tol_c0: (float) Only those values whose cost <= cost_threshold are returned, with
    :param tol_c1: (float)   cost_threshold = tol_c0 + tol_c1*min(cost)
    :return: (c_exp_values, c_step_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (c_exp[i], c_step[i]) having cost[i]
    """

    # --- init --------------------------------------------
    c_exp_min, c_exp_max = range_c_exp
    c_step_min, c_step_max = range_c_step

    c_exp_values = exp_sign * np.exp(np.linspace(np.log(c_exp_min), np.log(c_exp_max), n_grid))
    c_step_values = np.linspace(c_step_min, c_step_max, n_grid)

    # --- grid search -------------------------------------
    cost_arr = np.zeros(shape=(n_grid, n_grid))
    for i_e, c_exp in enumerate(c_exp_values):
        for i_step, c_step in enumerate(c_step_values):
            cost_arr[i_e, i_step] = fitting_cost(x, fx, c_step, c_exp, c_reg)

    # --- return good enough results ----------------------
    cost_min = np.min(cost_arr)
    cost_threshold = tol_c0 + (tol_c1 * cost_min)

    c_exp_lst = []
    c_step_lst = []
    cost_lst = []
    for i_e, c_exp in enumerate(c_exp_values):
        for i_step, c_step in enumerate(c_step_values):
            cost = cost_arr[i_e, i_step]
            if cost <= cost_threshold:
                c_exp_lst.append(c_exp)
                c_step_lst.append(c_step)
                cost_lst.append(cost)

    return np.array(c_exp_lst), np.array(c_step_lst), np.array(cost_lst)


# =================================================================================================
#  Fitting curve & cost
# =================================================================================================
@numba.njit(inline="always")
def fitting_cost(x: np.ndarray, fx: np.ndarray, c_step: float, c_exp: float, c_reg: float) -> float:
    """
    L1-cost of fitting fitting_curve to (x,fx)-values, where all x>0.
    A regularization term is added that should help keep c_exp=1.0 and c_step=0.0 unless the data convincingly says
    otherwise.
    """
    fx_pred = fitting_curve(x, c_step, c_exp)
    reg_term = c_reg * (abs(c_step) + abs(np.log10(c_exp)))
    return float(np.mean(np.abs(fx - fx_pred)) + reg_term)


@numba.njit(inline="always")
def fitting_curve(x: np.ndarray, c_step: float, c_exp: float) -> np.ndarray:
    """curve f(x) = c_step + (1-c_step)*(x**c_exp), assuming x>0"""
    return c_step + (1 - c_step) * (np.pow(x, c_exp))
