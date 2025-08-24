import numpy as np

from snuffled._core.compatibility import numba


@numba.njit
def estimate_sign_flip_frequency(
    w: np.ndarray,
    p: np.ndarray,
    lambda_min: float,
    lambda_max: float,
    n_iters: int = 5,
) -> float:
    """
    ---------------------
     PROBLEM DESCRIPTION
    ---------------------

    We consider some function g(x) that is assumed to exhibit 'sign flips' with some regularity.
    A sign flip means the sign of g(x) flips from - to + or + to -.

    We assume the sign flipping can be modelled as a Poisson process (over 'time' axis 'x') with 'time constant' tau,
    meaning on average every 'tau' time, 1 sign flip occurs.  Or, in other words, over any unit interval, we can
    expect lambda = 1/tau sign flips to occur.

    The goal is to estimate the parameter 'lambda' from a given set of observations.  The observations consist
    of tuples (w_i, p_i):
       w_i  :  width of interval i
       p_i  :  -> 1.0 if we observe a sign flip based on the function values g(.) at the interval endpoints
               -> 0.0 if we do not observe such a sign flip.
               -> a value in between represents a probability or mean from multiple such intervals

    ----------
     ANALYSIS
    ----------

        Observation 1

           If we assume 'lambda' is the sign flip frequency over a unit interval, we can assume
           lambda_i = w_i * lambda represents the expected number of occurrences over interval i of width w_i

        Observation 2

           Given the nature of how we observe sign flips, we need to correctly interpret observations:
               -> 1.0 means an  ODD number of sign flips occurred over the interval
                        -> p( ODD @ w_i) = 1/2 * (1 - exp(-2*lambda_i))
                                         = 1/2 * (1 - exp(-2*lambda*w_i))
               -> 0.0 means an EVEN number of sign flips occurred over the interval
                        -> p(EVEN @ w_i) = 1/2 * (1 + exp(-2*lambda_i))
                                         = 1/2 * (1 + exp(-2*lambda*w_i))

    ----------
     APPROACH
    ----------

    We can optimize the value of lambda using a CROSS-ENTROPY loss, which we can write down as...

      L = -1/N sum_i L_i

      L_i = p_i log(p( ODD @ w_i ))                  + (1 - p_i) log(p( EVEN @ w_i ))
          = p_i log(1/2 * (1 - exp(-2*lambda*w_i)))  + (1 - p_i) log(1/2 * (1 + exp(-2*lambda*w_i)))
          = p_i log(1 - exp(-2*lambda*w_i))          + (1 - p_i) log(1 + exp(-2*lambda*w_i))          + log(1/2)

    Optimization:
      - we'll perform an initial grid search over a log-equidistant lambda-grid
      - then we'll iteratively refine the grid around the current optimum to get a more accurate estimate
      - parameters
         - n_iters = 5   (default)
         - n_per_iter = 20  -> will result in a 10x refinement factor
      - this should give a relative accuracy of ~1e-5 of the total search range with 100 computations of the loss

    :param w, (N,)-sized array containing w_i values of (w_i, p_i) observation tuples
    :param p: (N,)-sized array containing p_i values of (w_i, p_i) observation tuples
    :param lambda_min: (float) left-most edge of [lambda_min, lambda_max] interval
    :param lambda_max: (float) right-most edge of [lambda_min, lambda_max] interval
    :param n_iters: (int) number of iteratively-refined grid search iterations
    :return: (float) optimal lambda value
    """

    # --- init --------------------------------------------
    n_per_iter = 20
    n_per_iter_per_side = int(n_per_iter / 2)
    log_lambda_min = np.log(lambda_min)
    log_lambda_max = np.log(lambda_max)

    results: dict[float, float] = dict()  # log_lambda -> loss

    # --- main loop ---------------------------------------
    for i in range(n_iters):
        # generate candidate log(lambda)-values to evaluate
        if i == 0:
            # first iteration -> initial grid
            log_lambda_cands = list(np.linspace(log_lambda_min, log_lambda_max, n_per_iter))
        else:
            # later iteration -> refine grid
            result_tuples = sorted([(loss, log_lambda) for log_lambda, loss in results.items()])
            log_lambda_values = sorted(results.keys())
            log_lambda_opt = result_tuples[0][1]
            log_lambda_cands = []
            if min(log_lambda_values) < log_lambda_opt:
                # there is a next-smaller value <log_lambda_opt
                log_lambda_smaller = max([log_l for log_l in log_lambda_values if log_l < log_lambda_opt])
                cands_smaller = list(np.linspace(log_lambda_smaller, log_lambda_opt, n_per_iter_per_side + 2))
                log_lambda_cands.extend(cands_smaller[1:-1])  # omit first and last, since we already have those

            if max(log_lambda_values) > log_lambda_opt:
                # there is a next-larger value >log_lambda_opt
                log_lambda_larger = min([log_l for log_l in log_lambda_values if log_l > log_lambda_opt])
                cands_larger = list(np.linspace(log_lambda_opt, log_lambda_larger, n_per_iter_per_side + 2))
                log_lambda_cands.extend(cands_larger[1:-1])  # omit first and last, since we already have those

        # evaluate log_lambda_cands
        for log_lambda in log_lambda_cands:
            if log_lambda not in results:
                loss = cross_entropy_loss(w, p, np.exp(log_lambda))
                results[log_lambda] = loss

    # determine & return final optimum
    result_tuples = sorted([(loss, log_lambda) for log_lambda, loss in results.items()])
    log_lambda_opt = result_tuples[0][1]
    return float(np.exp(log_lambda_opt))


@numba.njit(inline="always")
def cross_entropy_loss(w: np.ndarray, p: np.ndarray, lamb: float) -> float:
    exp_minus_two_lamb_w = np.exp(-2 * lamb * w)
    return -1 * np.sum((p * np.log(1 - exp_minus_two_lamb_w)) + ((1 - p) * np.log(1 + exp_minus_two_lamb_w)))
