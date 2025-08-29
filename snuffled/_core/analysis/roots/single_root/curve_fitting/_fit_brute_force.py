import numpy as np

from snuffled._core.compatibility import numba

from ._curves_and_costs import compute_threshold_cost, fitting_cost


# =================================================================================================
#  Find solution WITH uncertainty
# =================================================================================================
@numba.njit
def fit_curve_with_uncertainty_brute_force(
    x: np.ndarray,
    fx: np.ndarray,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    c_sign: float,
    n_grid: int,
    reg: float,
    rel_uncertainty_size: float = 1.0,
    fixed_a_value: float | None = None,
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
    :param rel_uncertainty_size: (float, default=1.0) factor to influence size of uncertainty region; this parameter
                                   maps to the 'relative_margin' parameter of the compute_threshold_cost function
    :param fixed_a_value: (float | None, default=None) Parameter to set a fixed value for 'a' instead of estimating it
                                   from the data. If None, the 'a' parameter will be estimated based on the median
                                   of the fx values.
    :return: (a_values, b_values, c_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (a[i], b[i], c[i]) having cost[i]
    """

    # --- init --------------------------------------------
    b_min, b_max = range_b
    c_min, c_max = range_c

    b_values = np.linspace(b_min, b_max, n_grid)
    c_values = c_sign * np.exp(np.linspace(np.log(c_min), np.log(c_max), n_grid))

    fx_q25, fx_q50, fx_q75 = np.quantile(fx, [0.25, 0.5, 0.75])

    # --- determine a -------------------------------------
    if fixed_a_value is not None:
        a = fixed_a_value
    else:
        a = float(fx_q50)  # should be a reasonable value for 'a' given that median(x) == 1.0 and g(1)==a

    # --- grid search -------------------------------------
    cost_arr = np.zeros(shape=(n_grid, n_grid))
    for i_b, b in enumerate(b_values):
        for i_c, c in enumerate(c_values):
            cost_arr[i_b, i_c] = fitting_cost(x, fx, a, b, c, reg)

    # --- return good enough results ----------------------
    cost_min = np.min(cost_arr)
    cost_threshold = compute_threshold_cost(
        relative_margin=rel_uncertainty_size,
        optimal_cost=cost_min,
        fx_q25=fx_q25,
        fx_q50=fx_q50,
        fx_q75=fx_q75,
    )

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
