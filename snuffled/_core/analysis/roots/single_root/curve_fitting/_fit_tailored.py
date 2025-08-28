import math

import numba
import numpy as np

from snuffled._core.utils.constants import EPS
from snuffled._core.utils.numba import clip_scalar, geomean

from ._curves_and_costs import compute_threshold_cost, fitting_cost, fitting_curve

# pre-computed constant, to avoid unnecessary re-computation and to improve readability
__LN_R = math.log(2 * math.sqrt(2))  # ln(r) with r = 2*math.sqrt(2),   used in param_step()


# =================================================================================================
#  Find solution WITH uncertainty
# =================================================================================================
def fit_curve_with_uncertainty_tailored(
    x: np.ndarray,
    fx: np.ndarray,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    reg: float,
    n_iters: int = 15,
    rel_uncertainty_size: float = 1.0,
):
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
    :param range_c:  (c_min, c_max)    range of c values  (>0)
    :param range_b: (b_min, b_max)  range of b values
    :param reg: (float) regularization coefficient that helps favour c=1.0, b=0.0  (e.g. 1e-3)
    :param n_iters: (int, default=15) number of iterations (both for optimum finding as uncertainty exploration)
    :param rel_uncertainty_size: (float, default=1.0) factor to influence size of uncertainty region; this parameter
                                   maps to the 'relative_margin' parameter of the compute_threshold_cost function
    :return: (a_values, b_values, c_values, cost_values)-tuples, each of which is a (k,)-sized numpy array with k>=1
               i-th elements of these arrays should be interpreted as tuples (a[i], b[i], c[i]) having cost[i]
    """

    # --- init --------------------------------------------
    fx_q25, fx_q50, fx_q75 = np.quantile(fx, [0.25, 0.5, 0.75])

    # placeholder data structures for return values
    a_lst, b_lst, c_lst, cost_lst = [], [], [], []

    # --- get optimal solution ----------------------------
    a_opt, b_opt, c_opt = fit_curve_tailored(x, fx, range_b, range_c, reg, n_iters)
    cost_opt = fitting_cost(x, fx, a_opt, b_opt, c_opt, reg)

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
    print("=== UNCERTAINTY EXPLORATION ================================")
    print(f"cost_opt       =", cost_opt)
    print(f"cost_threshold =", cost_threshold)
    for step_method in ["a", "b", "c", "ac", "ba", "bc"]:
        for step_dir in [-1.0, 1.0]:
            # initialize bisection
            step_size_min = 0.0
            step_size_max = 1.0
            cand_step_size = 1.0

            for i in range(n_iters):
                # evaluate cand_step_size
                a_cand, b_cand, c_cand = param_step(
                    a_opt, b_opt, c_opt, step_method, step_dir * cand_step_size, range_b, range_c
                )
                cost_cand = fitting_cost(x, fx, a_cand, b_cand, c_cand, reg)

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


# =================================================================================================
#  Find solution WITHOUT uncertainty - OPTIMAL FIT
# =================================================================================================
@numba.njit
def fit_curve_tailored(
    x: np.ndarray,
    fx: np.ndarray,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    reg: float,
    n_iters: int = 15,
) -> tuple[float, float, float]:
    """
    Tailored approach to find approximately optimal initial parameters (according to 'fitting_cost' method)
    with provided bounds of b and c parameters.

    We assume that (x, fx) have been generated using the compute_x_deltas method and that x-values have been
    rescaled such that median(x) == 1.0, i.e. by dividing by 2*dx.

    :return: (a,b,c)-tuple of parameters that are considered optimal, with b & c in requested ranges.
    """

    # --- init --------------------------------------------
    n = len(fx)
    fx_group_1 = fx[: n // 3]
    fx_group_2 = fx[n // 3 : 2 * n // 3]
    fx_group_3 = fx[2 * n // 3 :]

    # --- initial point -----------------------------------
    a_opt, b_opt, c_opt = 1.0, 0.0, 1.0
    optimal_cost = math.inf
    if np.min(fx) > 0.0:
        # if all fx are positive, we can use geomean-based approaches as well
        methods = ["median_local", "median_global", "geomean_local", "geomean_global"]
    else:
        # cannot take the geomean of negative values
        methods = ["median_local", "median_global"]

    for method in methods:
        # We potentially use 4 different methods for generating the initial point,
        #                                   differing in how we generate the 3 reference values.
        #
        # We use two aggregation methods:
        #    - median  : Most robust method that should theoretically work under all circumstances (any a,b,c-values),
        #                  but might be slightly less accurate because it doesn't truly average out multiple points.
        #    - geomean : Only expected to be accurate if b_true=0 (so no discontinuity).  But could be more accurate
        #                  under slightly noisy conditions, since it averages out data points in each group.
        #
        # We use two methods for generating fx_1:
        #    - local  : Only use the middle group of data points.
        #    - global : Use all data points.  Might be more accurate because more data points are taken into account,
        #                 but might be more subject to skewing effects under very noisy conditions.
        match method:
            case "median_local":
                fx_05 = np.median(fx_group_1)
                fx_1 = np.median(fx_group_2)
                fx_2 = np.median(fx_group_3)
            case "median_global":
                fx_05 = np.median(fx_group_1)
                fx_1 = np.median(fx)  # = global
                fx_2 = np.median(fx_group_3)
            case "geomean_local":
                fx_05 = geomean(fx_group_1)
                fx_1 = geomean(fx_group_2)
                fx_2 = geomean(fx_group_3)
            case "geomean_global":
                fx_05 = geomean(fx_group_1)
                fx_1 = geomean(fx)  # =global
                fx_2 = geomean(fx_group_3)
            case _:
                raise ValueError(f"Unknown method for initial point generation: {method}")

        a_est, b_est, c_est = fit_curve_exact_three_points(
            fx_05=fx_05,
            fx_1=fx_1,
            fx_2=fx_2,
            range_b=range_b,
            range_c=range_c,
        )
        current_cost = fitting_cost(x, fx, a_est, b_est, c_est, reg)

        print(f"Method '{method}'  --> cost=", current_cost)

        if current_cost < optimal_cost:
            a_opt, b_opt, c_opt = a_est, b_est, c_est
            optimal_cost = current_cost

    # --- refine - step 2 - grid --------------------------
    step_size = 1.0
    for i in range(n_iters):
        # start with step_size=1 and reduce with factor 2x each iteration
        if i > 0:
            step_size *= 0.5

        # take discrete steps
        for step_method in ["a", "b", "c", "ac", "ba", "bc"]:
            for step_dir in [-1.0, 1.0]:
                # compute candidate (a,b,c) values by taking a step from (a_opt, b_opt, c_opt)
                a_cand, b_cand, c_cand = param_step(
                    a=a_opt,
                    b=b_opt,
                    c=c_opt,
                    method=step_method,
                    step_size=step_dir * step_size,
                    range_b=range_b,
                    range_c=range_c,
                )

                # evaluate
                if (a_cand != a_opt) or (b_cand != b_opt) or (c_cand != c_opt):
                    # only evaluate if this is an actually new set of parameters
                    # (can happen that there's no change if we're already at the boundary of the search space)
                    current_cost = fitting_cost(x, fx, a_cand, b_cand, c_cand, reg)
                    if current_cost < optimal_cost:
                        print(
                            "Improved along ",
                            step_size * step_dir,
                            f"x {step_method}: ",
                            optimal_cost,
                            " --> ",
                            current_cost,
                        )
                        a_opt = a_cand
                        b_opt = b_cand
                        c_opt = c_cand
                        optimal_cost = current_cost
                        break  # no need to explore the other step direction (back to where we came from)

    # --- return ------------------------------------------
    return a_opt, b_opt, c_opt


@numba.njit
def param_step(
    a: float,
    b: float,
    c: float,
    method: str,
    step_size: float,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Take a step of size 'step_size' using method 'method' starting from current parameter values (a,b,c)
    and return (a_new, b_new, c_new).
    """

    # --- init --------------------------------------------
    a_new, b_new, c_new = a, b, c
    b_min, b_max = range_b
    c_min, c_max = range_c

    # --- take step ---------------------------------------
    if step_size != 0.0:
        match method:
            case "a" | "ac":
                # -------------------------------
                # These steps first modify parameter 'a' in a certain way and then optionally modify 'c'
                # to satisfy an invariant
                # -------------------------------
                # STEP 1: modify 'a' with a factor in [0.5, 2.0]
                a_new *= np.exp2(step_size)
                # STEP 2: modify 'c' if needed
                match method:
                    case "a":
                        # don't modify 'c', in this mode we only modify 'b'
                        pass
                    case "ac":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting c
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep b constant, this means we need to adjust c such that...
                        #    c = asinh(  (a'/a) * sinh(ln(r)*c') ) / ln(r)
                        ratio = a / a_new
                        c_new = math.asinh(ratio * math.sinh(__LN_R * c)) / __LN_R
            case "b" | "ba" | "bc":
                # -------------------------------
                # These steps first modify parameter 'b' in a certain way and then optionally modify 'c'
                # to satisfy an invariant
                # -------------------------------
                # STEP 1: modify 'b' in [b_min, b_max] with step_size=-1 -> b_min and step_size=+1 -> b_max  (LIN scale)
                if step_size < 0:
                    b_new = b + step_size * (b - b_min)
                else:
                    b_new = b + step_size * (b_max - b)
                # STEP 2: modify 'c' if needed
                match method:
                    case "b":
                        # don't modify 'c', in this mode we only modify 'b'
                        pass
                    case "ba":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting a
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep c constant, this means we need to adjust a such that...
                        #     a =  a'*(1-b')/(1-b)
                        ratio = (1 - b) / max(EPS, (1 - b_new))
                        a_new = a * ratio
                    case "bc":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting c
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep a constant, this means we need to adjust c such that...
                        #    c = asinh(  ((1-b')/(1-b)) * sinh(ln(r)*c') ) / ln(r)
                        ratio = (1 - b) / max(EPS, (1 - b_new))
                        c_new = math.asinh(ratio * math.sinh(__LN_R * c)) / __LN_R
            case "c":
                # modify 'c' with a factor in [0.5, 2.0]
                c_new *= np.exp2(step_size)
            case _:
                raise ValueError(f"Unknown step method: {method}")

    # --- return clipped updates --------------------------
    return (
        clip_scalar(float(a_new), 0.001 * a, 1000 * a),  # we don't have explicit ranges for a, we only want a>0
        clip_scalar(float(b_new), b_min, b_max),
        clip_scalar(float(c_new), c_min, c_max),
    )


# =================================================================================================
#  Find solution WITHOUT uncertainty - EXACT
# =================================================================================================
@numba.njit
def fit_curve_exact_three_points(
    fx_05: float,
    fx_1: float,
    fx_2: float,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Compute a,b,c exactly based on 3 data points  (0.5, fx_05), (1.0, fx_1), (2.0, fx_2),
    with solution being guaranteed to lie inside imposed b & c ranges.

    Reminder, we're trying to fit this function:

        g(x) = a*(b + (1-b)*(x^c))

    Solution is returned as tuple (a,b,c).
    """

    # initialize
    b_min, b_max = range_b
    c_min, c_max = range_c

    # computing a is straightforward
    a = fx_1
    if a == 0.0:
        # we cannot infer anything about the shape -> assume linear (c=1) without discontinuity (b=0)
        return a, 0.0, 1.0

    # Computing c (the exponent) essentially looks at the ratio of (fx_2-fx_1)/(fx_1-fx_05), which is
    # expected to be 2.0 if the function is linear, larger if c>1 and smaller of c<1.
    # Considering this ratio of differences also conveniently rids us of the influence of parameter b,
    # which we haven't determined yet.
    if fx_1 == fx_05:
        # ill conditioned, might be a sign of c being very close to 0, which squishes all f(x) value together
        c = c_min
    else:
        ratio = (fx_2 - fx_1) / (fx_1 - fx_05)
        if ratio <= 0:
            # this might again be a sign of poor conditioning due to all f(x) being squished together
            c = c_min
        else:
            c = clip_scalar(np.log2(ratio), c_min, c_max)

    # Now we can compute b by simply filling in a, c and solving for b
    b = 1 - (fx_2 - fx_05) / (fx_1 * (np.exp2(c) - np.exp2(-c)))
    b = clip_scalar(b, b_min, b_max)

    # return results as a tuple
    return a, b, c
