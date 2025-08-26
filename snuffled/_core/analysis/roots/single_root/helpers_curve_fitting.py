import math
import random

import numba
import numpy as np
from numba.core import types
from numba.typed.typeddict import Dict


# =================================================================================================
#
# =================================================================================================
@numba.njit
def compute_x_deltas(dx: float, k: int, seed: int = 42) -> np.ndarray:
    """
    Computes x_delta-array, containing values that can be used to sample function values around
    a root:    f(root ± x_delta).

    x_delta values are constructed such that...
        - we will have 3*(1 + 2*k) values
             - 1 group around   dx
             - 1 group around 2*dx
             - 1 group around 4*dx
        - with each group consisting of...
             - 1 sample at the reference point
             - k samples below the reference point
             - k samples symmetrically (geomean) above the reference point
        - each of the 3 groups of samples will span a 2x range from 1/sqrt(2) -> sqrt(2) relative around the ref. value
           - hence different groups do not overlap
           - but they span a consecutive 8x range [dx/sqrt(2), 4*dx*sqrt(2)]

    This will result in delta_x values with the following properties:
        - median(x_deltas < dx*sqrt(2))    =  geomean(x_deltas < dx*sqrt(2))    =    dx
        - median(x_deltas)                 =  geomean(x_deltas)                 =  2*dx
        - median(x_deltas > 2*sqrt(2)*dx)  =  geomean(x_deltas > 2*sqrt(2)*dx)  =  4*dx

    :param dx: (float) reference distance dx > 0
    :param k: (int) sampling count parameter
    :param seed: (int) seed for random number generator
    :return: np.ndarray with 3*(1+2k) x_delta values, sorted in increasing order
    """

    # initialize
    random.seed(seed)
    x_deltas = np.zeros(3 + 6 * k)
    x_deltas[0] = dx
    x_deltas[1] = 2 * dx
    x_deltas[2] = 4 * dx

    # 6*k additional randomized samples
    for j in range(k):
        # set up iteration j
        r_outer = 1 + ((math.sqrt(2) - 1) * random.random())
        r_inner = 1 + ((math.sqrt(2) - 1) * random.random())
        i_start = 3 + (6 * j)

        # two values geo-symmetrically around dx
        x_deltas[i_start] = dx * r_outer
        x_deltas[i_start + 1] = dx / r_outer

        # two values geo-symmetrically around 2*dx
        x_deltas[i_start + 2] = 2 * dx * r_inner
        x_deltas[i_start + 3] = 2 * dx / r_inner

        # two values geo-symmetrically around 4*dx
        x_deltas[i_start + 4] = 4 * dx * r_outer
        x_deltas[i_start + 5] = 4 * dx / r_outer

    return np.sort(x_deltas)


# =================================================================================================
#  Optimization
# =================================================================================================
@numba.njit
def fit_curve_brute_force(
    x: np.ndarray,
    fx: np.ndarray,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    c_sign: float,
    n_grid: int,
    reg: float,
    tol_c0: float,
    tol_c1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a curve of the following form to a list of (x, fx)-tuples:

        g(x) = a * (b + (1-b)*(x^c))

    by minimizing the L1 cost (mean of absolute deviations) over (a,b,c) search space:
      - first we determine an appropriate value for 'a'
      - then we optimize by brute force grid search over a rectangular grid of (b, c)-values.
    The grid is (n_grid x n_grid) values and is linear in b and logarithmic in c.

    This function does not return a single optimal (a, b, c)-value, but returns all tuples in the search grid
    whose cost is 'close enough' (as determined by tol_c0, tol_c1) to the optimal cost.

    NOTE:
      - we assume median(x) == 1

    :param x: (n,)-sized numpy array containing x values, with each x>0
    :param fx: (n,)-sized numpy array containing corresponding f(x) values
    :param range_c:  (c_min, c_max)    range of c values  (>0)
    :param range_b: (b_min, b_max)  range of b values
    :param c_sign: (float) sign of c, i.e. if _sign=-1, we will search over range [-c_max, -c_min].
    :param n_grid: (int) number of grid values along each dimension
    :param reg: (float) regularization coefficient that helps favour c=1.0, b=0.0  (e.g. 1e-3)
    :param tol_c0: (float) Only those values whose cost <= cost_threshold are returned, with
    :param tol_c1: (float)   cost_threshold = tol_c0 + tol_c1*min(cost)
    :return: (a_values, b_values, c_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (a[i], b[i], c[i]) having cost[i]
    """

    # --- determine a -------------------------------------
    a = float(np.median(fx))  # should be a reasonable value for 'a' given that median(x) == 1.0 and g(1)==a

    # --- init --------------------------------------------
    b_min, b_max = range_b
    c_min, c_max = range_c

    b_values = np.linspace(b_min, b_max, n_grid)
    c_values = c_sign * np.exp(np.linspace(np.log(c_min), np.log(c_max), n_grid))

    # --- grid search -------------------------------------
    cost_arr = np.zeros(shape=(n_grid, n_grid))
    for i_b, b in enumerate(b_values):
        for i_c, c in enumerate(c_values):
            cost_arr[i_b, i_c] = fitting_cost(x, fx, a, b, c, reg)

    # --- return good enough results ----------------------
    cost_min = np.min(cost_arr)
    cost_threshold = tol_c0 + (tol_c1 * cost_min)

    a_lst, b_lst, c_lst = [], [], []
    cost_lst = []
    for i_b, b in enumerate(b_values):
        for i_c, c in enumerate(c_values):
            cost = cost_arr[i_b, i_c]
            if cost <= cost_threshold:
                a_lst.append(a)
                b_lst.append(b)
                c_lst.append(c)
                cost_lst.append(cost)

    return np.array(a_lst), np.array(b_lst), np.array(c_lst), np.array(cost_lst)


@numba.njit
def fit_curve_random_walk(
    x: np.ndarray,
    fx: np.ndarray,
    range_c_exp: tuple[float, float],
    range_c_step: tuple[float, float],
    exp_sign: float,
    n_iters: int,
    t_init: float,
    t_final: float,
    step_size_init: float,
    step_size_final: float,
    c_reg: float,
    tol_c0: float,
    tol_c1: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closely resembles 'fit_curve_brute_force' in terms of the problem it solves, the arguments and the return values.

    The main difference is that this method does not use brute force 2d grid search, but rather uses a
    Simulated Annealing style of approach.

    Random steps of 'step_size' are taken and new evaluations are accepted with a probability that depends on the
    function value & temperate.  Relative function deltas will always be used.  We will consider 2 deltas:
      - delta wrt optimal cost
      - delta wrt threshold cost (below which we return results).

    Temperature is decreased exponentially from t_init to t_final; same for step size.  Step sizes are to be interpreted
    wrt to a normalized 2d search space representing a unit box [0,1] x [0,1].

    Reasonable tuning variables are probably...
        n_iters         = 1000
        t_init          = 1.0
        t_final         = 0.1
        step_size_init  = 0.1
        step_size_final = 0.001

    """

    # --- initialize grid ---------------------------------
    # MAP (z0, z1) in [0,1]x[0,1] -> [c_exp_min, c_exp_max]x[c_step_min,c_step_max]
    # c_exp = np.exp2(c_exp_c0 + c_exp_c1 * z0)
    c_exp_c0 = np.log2(range_c_exp[0])
    c_exp_c1 = np.log2(range_c_exp[1] / range_c_exp[0])
    # c_step = c_step_c0 + c_step_c1 * z1
    c_step_c0 = range_c_step[0]
    c_step_c1 = range_c_step[1] - range_c_step[0]

    # this is the candidate that we will update each iteration
    z = np.array([0.5, 0.5], dtype=np.float64)
    z_cand = z

    # --- initialize annealing ----------------------------
    t = t_init
    t_factor = (t_final / t_init) ** (1 / (n_iters - 1))
    step_size = step_size_init
    step_size_factor = (step_size_final / step_size_init) ** (1 / (n_iters - 1))

    # --- initialize data structures ----------------------
    cost_min = 1e3
    cost_threshold = tol_c0 + (tol_c1 * cost_min)
    cost_dict = Dict.empty(
        key_type=types.UniTuple(types.float64, 2),
        value_type=types.float64,
    )  #  map (c_exp, c_step) -> cost

    # --- simulated annealing -----------------------------
    random.seed(seed)
    for i in range(n_iters):
        # generate new candidate
        if i == 0:
            # if i==0, we still need to evaluate the starting point z
            z_cand = z
        else:
            # otherwise, we can start perturbing
            z_cand[0] = np.clip(z[0] + (step_size * (1 - 2 * random.random())), 0.0, 1.0)
            z_cand[1] = np.clip(z[1] + (step_size * (1 - 2 * random.random())), 0.0, 1.0)

        # evaluate current z_cand
        c_exp = np.exp2(c_exp_c0 + c_exp_c1 * z[0])
        c_step = c_step_c0 + c_step_c1 * z[1]
        cost = fitting_cost(x, fx, c_step, c_exp, c_reg)

        # see if we should remember this one
        if cost <= cost_threshold:
            cost_dict[(c_exp, c_step)] = cost

        # acceptance logic
        p = 0.0
        p += 0.5 * (1.0 if cost < cost_min else np.exp(-(cost - cost_min) / (t * cost_min)))
        p += 0.5 * (1.0 if cost < cost_threshold else np.exp(-(cost - cost_threshold) / (t * cost_threshold)))
        if random.random() <= p:
            z = z_cand

        # update cost_min
        if cost < cost_min:
            cost_min = cost
            cost_threshold = tol_c0 + (tol_c1 * cost_min)

        # update t, step_size
        t *= t_factor
        step_size *= step_size_factor

    # --- return good enough results ----------------------
    c_exp_lst = []
    c_step_lst = []
    cost_lst = []
    for (c_exp, c_step), cost in cost_dict.items():
        if cost <= cost_threshold:
            c_exp_lst.append(c_exp)
            c_step_lst.append(c_step)
            cost_lst.append(cost)

    return np.array(c_exp_lst), np.array(c_step_lst), np.array(cost_lst)


# =================================================================================================
#  Fitting curve & cost
# =================================================================================================
@numba.njit(inline="always")
def fitting_cost(x: np.ndarray, fx: np.ndarray, a: float, b: float, c: float, reg: float) -> float:
    """
    L1-cost of fitting fitting_curve to (x,fx)-values, where all x>0.
    A regularization term is added that should help keep c=1.0 and b=0.0 unless the data convincingly says
    otherwise.
    """
    fx_pred = fitting_curve(x, a, b, c)
    reg_term = reg * (abs(b) + abs(np.log10(c)))
    return float(np.mean(np.abs(fx - fx_pred)) + reg_term)


@numba.njit(inline="always")
def fitting_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """curve g(x) = a * (b + (1-b)**(x^c), assuming x>0"""
    return a * (b + (1 - b) * (np.pow(x, c)))
