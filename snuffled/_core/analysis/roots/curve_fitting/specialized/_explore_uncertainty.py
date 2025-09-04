import numpy as np

from snuffled._core.compatibility import numba

from ..shared import fitting_cost
from ._helpers import param_step


@numba.njit
def explore_uncertainty(
    x: np.ndarray,
    fx: np.ndarray,
    a_opt: float,
    b_opt: float,
    c_opt: float,
    cost_opt: float,
    cost_threshold: float,
    range_a: tuple[float, float],
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    reg: float,
    n_iters: int = 20,
    tol: float = 1e-3,
    debug_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # --- shortcuts ---------------------------------------
    if cost_opt > cost_threshold:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    # --- init --------------------------------------------
    a_lst, b_lst, c_lst, cost_lst = [a_opt], [b_opt], [c_opt], [cost_opt]

    # --- uncertainty exploration -------------------------
    if cost_threshold > cost_opt:
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
                    elif (1 - tol) * cost_threshold <= cost_cand <= cost_threshold:
                        # whenever we get within 'tol' of the threshold cost (from below, so we can remember & return
                        # this solution), we can stop.
                        break
                    else:
                        if cost_cand > cost_threshold:
                            step_size_max = cand_step_size
                        else:
                            step_size_min = cand_step_size

                    # prepare next iteration
                    cand_step_size = 0.5 * (step_size_min + step_size_max)

    # --- return all results ------------------------------
    return (
        np.array(a_lst, dtype=np.float64),
        np.array(b_lst, dtype=np.float64),
        np.array(c_lst, dtype=np.float64),
        np.array(cost_lst, dtype=np.float64),
    )
