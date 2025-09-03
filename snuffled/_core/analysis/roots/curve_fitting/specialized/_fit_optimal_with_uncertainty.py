import numpy as np

from ..shared import compute_threshold_cost, fitting_cost
from ._fit_optimal import fit_curve
from ._helpers import param_step


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
    n_iters: int = 15,
    rel_uncertainty_size: float = 1.0,
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
    :param n_iters: (int, default=15) number of iterations (both for optimum finding as uncertainty exploration)
    :param rel_uncertainty_size: (float, default=1.0) factor to influence size of uncertainty region; this parameter
                                   maps to the 'relative_margin' parameter of the compute_threshold_cost function
    :param debug_flag: (bool, default=False) if True, stdout output is generated to debug algorithm flow.
    :return: (a_values, b_values, c_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (a[i], b[i], c[i]) having cost[i]
    """

    # --- init --------------------------------------------
    fx_q25, fx_q50, fx_q75 = np.quantile(fx, [0.25, 0.5, 0.75])

    # placeholder data structures for return values
    a_lst, b_lst, c_lst, cost_lst = [], [], [], []

    # --- get optimal solution ----------------------------
    a_opt, b_opt, c_opt, cost_opt = fit_curve(x, fx, range_a, range_b, range_c, reg, n_iters, debug_flag)

    # remember this solution
    a_lst.append(a_opt)
    b_lst.append(b_opt)
    c_lst.append(c_opt)
    cost_lst.append(cost_opt)

    # --- determine uncertainty bounds --------------------
    cost_threshold = compute_threshold_cost(
        relative_margin=rel_uncertainty_size,
        optimal_cost=cost_opt,
        fx_q25=fx_q25,
        fx_q50=fx_q50,
        fx_q75=fx_q75,
    )

    # Find edge points of uncertainty region by performing bisection over the step_size parameter in ±[0,1]
    # until we find the edge.  We know that for step_size==0.0 we are strictly below the threshold_cost.
    # If for step_size==1.0 we are still below, then we consider this an edge point; otherwise we can perform bisection.
    if debug_flag:
        print("=== UNCERTAINTY EXPLORATION ================================")
        print(f"cost_opt       =", cost_opt)
        print(f"cost_threshold =", cost_threshold)
        print("============================================================")
    for step_method in ["a", "b", "c", "ac", "ba", "bc"]:
        for step_dir in [-1.0, 1.0]:
            # initialize bisection
            step_size_min = 0.0
            step_size_max = 1.0
            cand_step_size = 1.0

            for i in range(n_iters):
                # evaluate cand_step_size
                a_cand, b_cand, c_cand = param_step(
                    a_opt, b_opt, c_opt, step_method, step_dir * cand_step_size, range_a, range_b, range_c
                )
                cost_cand = fitting_cost(x, fx, a_cand, b_cand, c_cand, reg)

                if debug_flag:
                    print(f"direction '{step_method}', step_size=", step_dir * cand_step_size, " cost=", cost_cand)

                if cost_cand <= cost_threshold:
                    # remember this solution
                    a_lst.append(a_cand)
                    b_lst.append(b_cand)
                    c_lst.append(c_cand)
                    cost_lst.append(cost_cand)

                # determine what to do with this solution and how to proceed
                if i == 0:
                    # first iteration, which means cand_step_size was 1.0
                    #  --> if cost_cand <= cost_threshold, we don't need to do bisection
                    if cost_cand <= cost_threshold:
                        break
                elif 0.999 * cost_threshold <= cost_cand <= cost_threshold:
                    # whenever we get close enough to the threshold cost (from below, so we can remember & return
                    # this solution), we can stop.
                    break
                else:
                    if cost_cand > cost_threshold:
                        step_size_max = cand_step_size
                    else:
                        step_size_min = cand_step_size

                # prepare next iteration
                cand_step_size = 0.5 * (step_size_min + step_size_max)

    # return all results
    return np.array(a_lst), np.array(b_lst), np.array(c_lst), np.array(cost_lst)
