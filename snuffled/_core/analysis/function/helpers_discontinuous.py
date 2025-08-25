import math

import numba
import numpy as np

from snuffled._core.analysis import FunctionSampler


# =================================================================================================
#  Main computation
# =================================================================================================
def discontinuity_score(function_sampler: FunctionSampler, n_samples: int) -> float:
    """
    Compute discontinuity score (value in [0,1]) for the given function indicating to what extent it suffers
    from discontinuities, where...
      - a purely linear function          --> score 0.0
      - a pure step function              --> score 1.0
      - a linear combination of the above --> score >0.0 and <1.0

    We only examine the function f(x) in interval [x_min, x_max] and zoom in up to zoom level dx_min, i.e.
    we do not consider sub-intervals [x_l, x_r] with x_r - x_l < dx_min.

    ----------------------
     A. SCORING MECHANISM
    ----------------------

    It is not possible to always detect discontinuities purely numerically in a fail-safe way, with a finite function
    sampling budget (finite # of function evaluations), so some heuristics will need to be used:
      - consider we have sampled the function at x-values x_i, i=0...n   (with x_i sorted)
      - this means we have n intervals [x_i, x_(i+1)],
                                  with dx_i   = x_(i+1) - x_i
                                       dfx_i  = | f(x_(i+1)) - f(x_i) |
                                       dfdx_i = dfx_i / dx_i              (we ignore the sign of derivatives)
      - define dfx_total = sum_i dfx_i     (=total up-down movement of the function)
      - This brings us to 2 'reference derivatives'
          - dfdx_mean = dfx_total / (x_max - x_min)     (average absolute derivative of the function)
          - dfdx_max  = dfx_total / dx_min              (max theoretical derivative of an interval, given dfx_total)
      - given all these definitions, we will compute the score as...

          -----------------------------------------------------------------------------------------------------------
           DISCONTINUITY SCORE

             --- DEFINITION ---
             Fraction of the up-down movement of the function that happens with a sufficiently steep derivative
             that it can be considered a 'step', with 'sufficiently steep' meaning dfdx_i >> dfdx_threshold,
             with dfdx_threshold = sqrt(dfdx_mean * dfdx_max).

             --- PRACTICAL COMPUTATION ---
             For the score to behave smoothly for smoothly varying circumstances, we will assign a weight w_i in [0,1]
             to each interval, such that...
                  w_i ~= 0.0    if   dfdx_i << dfdx_threshold
                  w_i ~= 0.5    if   dfdx_i ~= dfdx_threshold
                  w_i ~= 1.0    if   dfdx_i >> dfdx_threshold

             The total score S can then be computed as...

                  S = sum_i s_i      with     s_i = w_i * (dfx_i / dfdx_total)

          -----------------------------------------------------------------------------------------------------------

    -----------------------
     B. SAMPLING ALGORITHM
    -----------------------

        The total score we obtain using the aforementioned heuristics, clearly depends on how we have chosen the
        samples x_i.  E.g. if a function exhibits one step at x=x_s, it will be clearly beneficial to have identified
        a dx_min-sized interval containing x_s, such that we can isolate the vertical step and score that interval
        accurately in isolation.

        Overall flow
          1) start with x_i, fx_i values     (reusing the multiscale sampling used for other properties)
          2) iterate until stopping criterion is reached (max iters, max f evals, ...) ...
              a) select k intervals with largest dfx_i and dx_i > dx_min as candidates to be further investigated
              b) for the midpoints of the selected intervals, compute additional samples f(x_j) and add those to
                   the existing set
              c) repeat from step a)

    :param function_sampler: (FunctionSampler) used to get sampling settings + obtain addition f(x_j) samples
    :param n_samples: (int) number of additional samples we can take to zoom in on suspected discontinuities.
    :return score: (int) value in [0,1]
    """

    # --- init --------------------------------------------

    # extract sampling settings
    dx_min = function_sampler.dx
    x_min, x_max = function_sampler.x_min, function_sampler.x_max
    dx_total = x_max - x_min

    # determine iteration count, etc...
    log2_dx_min_max_ratio = math.log2(dx_total / dx_min)  # number of bisection steps to get from dx_total to dx_min
    n_iters = math.ceil(
        max(
            2 * log2_dx_min_max_ratio,  # margin 2x, so also later detected suspicious intervals can be fully bisected
            math.sqrt(n_samples),
        )
    )
    n_samples_remaining = n_samples

    # --- iterative sampling ------------------------------
    for i in range(n_iters):
        # init
        n_samples_i = math.ceil(n_samples_remaining / (n_iters - i))

        # analyse current intervals & give scores
        all_samples = sorted(function_sampler.function_cache())
        intervals = [
            (x0, x1, x1 - x0, abs(fx1 - fx0))  # convert to (x_left, x_right, dx, dfx)-tuples
            for (x0, fx0), (x1, fx1) in zip(all_samples[:-1], all_samples[1:])
        ]
        dfx_total = sum(dfx for _, _, _, dfx in intervals)

        # We use dfx*deriv + dx^2 as heuristic resampling score
        # Rationale:
        #    dfx*deriv is the main driving heuristic
        #       --> if we split a mostly linear interval in 2 --> the score halves and is split acros both intervals
        #       --> if we split an interval with a step       --> the score of the step-containing sub-interval doubles
        #    This way we focus primarily first on those intervals that look most 'suspicious' until
        #
        #    In order to avoid long seemingly constant intervals stay under the radar indefinitely, we add a small dx^2
        #     component to also focus on long flat intervals, if we don't have any other suspicious looking ones.
        #
        #    Note that we normalize both dx & dfx to be scaling-independent
        def heuristic(_dx: float, _dfx: float) -> float:
            if _dx > dx_min:
                _dx_rel = _dx / dx_total
                _dfx_rel = _dfx / dfx_total
                return (_dfx_rel * _dfx_rel / _dx_rel) + (_dx_rel * _dx_rel)
            else:
                return 0.0

        scores = [heuristic(dx, dfx) for _, _, dx, dfx in intervals]

        # sort by score and generate new candidate x-values for sampling
        scored_intervals = sorted(list(zip(scores, intervals)), reverse=True)
        new_x_values = [
            0.5 * (x_left + x_right)
            for (score, (x_left, x_right, dx, dfx)) in scored_intervals[:n_samples_i]
            if score > 0
        ]

        # sample function
        _ = function_sampler.f(new_x_values)  # no need to keep track of result, it's stored in the sampler cache
        n_samples_remaining -= len(new_x_values)

        if (len(new_x_values) == 0) or (n_samples_remaining == 0):
            break  # stop prematurely

    # --- compute final score -----------------------------
    samples = sorted(function_sampler.function_cache())
    x_values = np.array([x for x, fx in samples], dtype=np.float64)
    fx_values = np.array([fx for x, fx in samples], dtype=np.float64)
    return compute_discontinuity_score_from_intervals(
        x_values=x_values,
        fx_values=fx_values,
        dx_min=dx_min,
    )


# =================================================================================================
#  Helper functions
# =================================================================================================
@numba.njit
def compute_discontinuity_score_from_intervals(x_values: np.ndarray, fx_values: np.ndarray, dx_min: float) -> float:
    """
    Computes final discontinuity score (value in [0,1]) based on the provided (x,fx)-samples
    and minimum interval width dx_min.

    The provided x_values are assumed to be sorted and to span the entire interval of interest, but
    we do not assume interval widths are >= dx_min.

    :param x_values: (n,)-sized np.ndarray with x-values (sorted in ascending order and without duplicates)
    :param fx_values: (n,)-sized np.ndarray with corresponding f(x)-values
    :param dx_min: (float > 0) smallest relevant interval size.
    :return: score in [0,1]
    """

    # --- init --------------------------------------------
    dx_values = np.diff(x_values)
    dfx_values = np.abs(np.diff(fx_values))
    dx_total = x_values[-1] - x_values[0]
    dfx_total = np.sum(dfx_values)

    if dfx_total == 0.0:
        return 0.0  # constant function cannot have discontinuities

    # calibrate interval weighting function
    #    such that     deriv_i = deriv_mean       -->  w_i = 0.0
    #                  deriv_i = deriv_threshold  -->  w_i = 0.5
    #                  deriv_i = deriv_max        -->  w_i = 1.0
    deriv_mean = dfx_total / dx_total  # avg. absolute derivative over entire interval
    deriv_max = dfx_total / dx_min  # if the entire dfx_total would occur in 1 interval of size dx_min
    deriv_threshold = math.sqrt(deriv_mean * deriv_max)
    # we now compute 'c' such that deriv_i = deriv_mean        -->  w_i = 0.0
    #                              deriv_i = deriv_threshold   -->  w_i = 0.5
    #                              deriv_i = deriv_max         -->  w_i = 1.0
    # when using the formula:
    #   w_i_unscaled = 1 / (1 + (deriv_threshold/deriv_i))
    #   w_i          = 0.5 * c*(w_i_unscaled - 0.5)
    #
    # NOTE: we choose this type of scaling (in the w_i-direction rather than the deriv-direction),
    #       since we don't want the transition around the threshold derivative to become sharper for increasing
    #       values of dx_min.  The result will be that for smaller dx_min values there will be 'wider' regions
    #       where w_i is close to 0.0 or 1.0, which then correctly reflects the fact that we can more easily
    #       distinguish 'step' vs 'non-step' intervals if dx_min is smaller.
    c = 0.5 / ((1 / (1 + (deriv_threshold / deriv_max))) - 0.5)

    # --- actual computations -------------------
    score = 0.0
    for i in range(len(dx_values)):
        dfx_i, dx_i = dfx_values[i], dx_values[i]
        if dfx_i > 0:
            deriv_i = dfx_i / dx_i
            w_i_unscaled = 1 / (1 + (deriv_threshold / deriv_i))
            w_i = 0.5 + c * (w_i_unscaled - 0.5)
            w_i = min(max(w_i, 0.0), 1.0)
            score += w_i * (dfx_i / dfx_total)

    return score
