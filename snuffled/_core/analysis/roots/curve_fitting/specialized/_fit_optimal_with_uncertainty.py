import math

import numpy as np

from ..shared import compute_threshold_cost
from ._explore_uncertainty import explore_uncertainty
from ._fit_optimal import fit_curve


# =================================================================================================
#  Find solution WITH uncertainty
# =================================================================================================
def fit_curve_with_uncertainty(
    x: np.ndarray,
    fx: np.ndarray,
    range_a: tuple[float, float],
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    reg: float,
    include_negative_c: bool = False,
    n_iters: int = 15,
    uncertainty_size: float = 1.0,
    uncertainty_tol: float = 1e-3,
    debug_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Method for estimating optimal parameters (a,b,c) for curve fitting as well as uncertainties, similar to
    fit_curve_with_uncertainty_brute_force.

    The returned method is different in that we use an approach specifically tailored to the problem structure at hand
      - first compute the optimum (a_opt, b_opt, c_opt) using fit_curve_tailored
      - starting from the optimum find the edges of the uncertainty region (using bisection) using the same
          parameter search directions

    Results are returned as arrays of a,b,c,cost values, containing all parameter sets that were encountered during
      the uncertainty exploration to be within the cost_threshold, including the actual optimum.

    NOTE: I corner cases, it can happen that we report results that are slightly outside the uncertainty range.
          This can happen when we serendipitously find more optimal results during uncertainty exploration.  In this
          case we won't update the cost_threshold, but continue with the original cost_threshold which is based on the
          result of fit_curve_tailored(...).

    :param x: (n,)-sized numpy array containing x values, with each x>0
    :param fx: (n,)-sized numpy array containing corresponding f(x) values
    :param range_a: (a_min, a_max)  range of a values  (>0)
    :param range_b: (b_min, b_max)  range of b values
    :param range_c: (c_min, c_max)  range of c values  (>0)
    :param reg: (float) regularization coefficient that helps favour c=1.0, b=0.0  (e.g. 1e-3)
    :param include_negative_c: (bool, default=False) if True, also check for fit with c-range (-c_max, -c_min)
    :param n_iters: (int, default=15) number of iterations (both for optimum finding as uncertainty exploration)
    :param uncertainty_size: (float, default=1.0) factor to influence size of uncertainty region; this parameter
                                   maps to the 'relative_margin' parameter of the compute_threshold_cost function
    :param uncertainty_tol: (float, default=1e-3) accuracy with which uncertainty region needs to be determined
    :param debug_flag: (bool, default=False) if True, stdout output is generated to debug algorithm flow.
    :return: (a_values, b_values, c_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (a[i], b[i], c[i]) having cost[i]
    """

    # --- init --------------------------------------------
    if include_negative_c:
        c_ranges = [range_c, (-range_c[1], -range_c[0])]
    else:
        c_ranges = [range_c]

    # --- get optimal solutions ---------------------------
    abc_opt_lst: list[tuple[float, float, float]] = []
    cost_opt_lst: list[float] = []
    for range_c in c_ranges:
        # go over each range_c-value and find optimal fit
        a_opt, b_opt, c_opt, cost_opt = fit_curve(x, fx, range_a, range_b, range_c, reg, n_iters, debug_flag)
        abc_opt_lst.append((a_opt, b_opt, c_opt))
        cost_opt_lst.append(cost_opt)

    # --- determine threshold cost for uncertainty --------
    fx_q25, fx_q50, fx_q75 = np.quantile(fx, [0.25, 0.5, 0.75])
    cost_threshold = compute_threshold_cost(
        relative_margin=uncertainty_size,
        optimal_cost=min(cost_opt_lst),  # overall optimal cost
        fx_q25=fx_q25,
        fx_q50=fx_q50,
        fx_q75=fx_q75,
    )

    # --- get uncertainty bounds --------------------------
    a_lst: list[np.ndarray] = []
    b_lst: list[np.ndarray] = []
    c_lst: list[np.ndarray] = []
    cost_lst: list[np.ndarray] = []
    for (a_opt, b_opt, c_opt), cost_opt, range_c in zip(abc_opt_lst, cost_opt_lst, c_ranges):
        if cost_opt <= cost_threshold:
            # there is at least 1 solution in this c-range that falls within the cost_threshold
            a_values, b_values, c_values, cost_values = explore_uncertainty(
                x,
                fx,
                a_opt,
                b_opt,
                c_opt,
                cost_opt,
                cost_threshold,
                range_a,
                range_b,
                range_c,
                reg,
                n_iters,
                uncertainty_tol,
                debug_flag,
            )
            a_lst.append(a_values)
            b_lst.append(b_values)
            c_lst.append(c_values)
            cost_lst.append(cost_values)

    # --- aggregate & return ------------------------------
    return (
        np.array([a for a_arr in a_lst for a in a_arr]),
        np.array([b for b_arr in b_lst for b in b_arr]),
        np.array([c for c_arr in c_lst for c in c_arr]),
        np.array([cost for cost_arr in cost_lst for cost in cost_arr]),
    )
