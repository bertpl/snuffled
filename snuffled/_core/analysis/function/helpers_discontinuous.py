import math

import numba
import numpy as np

from snuffled._core.analysis import FunctionSampler


# ==================================================================================================
#  Main computation
# ==================================================================================================
def discontinuity_score(function_sampler: FunctionSampler, n_samples: int) -> float:
    """Compute discontinuity score (value in [0,1]) for the given function.

    The score indicates to what extent the function suffers from discontinuities, where...
      - a purely linear function          --> score 0.0
      - a pure step function              --> score 1.0
      - a linear combination of the above --> score >0.0 and <1.0.

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
    :return score: (float) value in [0,1]
    """
    # --- init -----------------------------------
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

    # --- iterative sampling ---------------------
    # The samples stay sorted by x across iterations: each iteration merges its new samples into them, so the
    # function cache is sorted only once.
    samples = sorted(function_sampler.function_cache())
    x_values = np.array([x for x, fx in samples], dtype=np.float64)
    fx_values = np.array([fx for x, fx in samples], dtype=np.float64)
    for i in range(n_iters):
        n_samples_i = math.ceil(n_samples_remaining / (n_iters - i))

        # sample the function at the midpoints of the intervals that look most like a step
        new_x_values = select_midpoints_to_sample(x_values, fx_values, dx_min, dx_total, n_samples_i)
        new_fx_values = np.array(function_sampler.f(new_x_values.tolist()), dtype=np.float64)
        x_values, fx_values = insert_sorted_samples(x_values, fx_values, new_x_values, new_fx_values)
        n_samples_remaining -= len(new_x_values)

        if (len(new_x_values) == 0) or (n_samples_remaining == 0):
            break  # stop prematurely

    # --- compute final score --------------------
    return compute_discontinuity_score_from_intervals(
        x_values=x_values,
        fx_values=fx_values,
        dx_min=dx_min,
    )


# ==================================================================================================
#  Helper functions
# ==================================================================================================
@numba.njit
def compute_discontinuity_score_from_intervals(x_values: np.ndarray, fx_values: np.ndarray, dx_min: float) -> float:
    """Computes final discontinuity score (value in [0,1]) from the provided samples.

    Based on the provided (x,fx)-samples and minimum interval width dx_min.

    The provided x_values are assumed to be sorted and to span the entire interval of interest, but
    we do not assume interval widths are >= dx_min.

    :param x_values: (n,)-sized np.ndarray with x-values (sorted in ascending order and without duplicates)
    :param fx_values: (n,)-sized np.ndarray with corresponding f(x)-values
    :param dx_min: (float > 0) smallest relevant interval size.
    :return: score in [0,1]
    """
    # --- init -----------------------------------
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
    #   w_i          = 0.5 + c*(w_i_unscaled - 0.5)
    #
    # NOTE: we choose this type of scaling (in the w_i-direction rather than the deriv-direction),
    #       since we don't want the transition around the threshold derivative to become sharper for increasing
    #       values of dx_min.  The result will be that for smaller dx_min values there will be 'wider' regions
    #       where w_i is close to 0.0 or 1.0, which then correctly reflects the fact that we can more easily
    #       distinguish 'step' vs 'non-step' intervals if dx_min is smaller.
    c = 0.5 / ((1 / (1 + (deriv_threshold / deriv_max))) - 0.5)

    # --- actual computations --------------------
    score = 0.0
    for i in range(len(dx_values)):
        dfx_i, dx_i = dfx_values[i], dx_values[i]
        if dfx_i > 0:
            deriv_i = dfx_i / dx_i
            w_i_unscaled = 1 / (1 + (deriv_threshold / deriv_i))
            w_i = 0.5 + c * (w_i_unscaled - 0.5)
            w_i = min(max(w_i, 0.0), 1.0)
            score += w_i * (dfx_i / dfx_total)

    # score is a weighted average of w_i in [0, 1] (weights dfx_i/dfx_total sum to 1), so it is
    # mathematically in [0, 1]; clamp to absorb floating-point rounding at the 1.0 boundary.
    return min(max(score, 0.0), 1.0)


@numba.njit
def select_midpoints_to_sample(
    x_values: np.ndarray, fx_values: np.ndarray, dx_min: float, dx_total: float, n_max: int
) -> np.ndarray:
    """Return the midpoints of the at most n_max intervals with the highest positive resampling score.

    The midpoints are returned by decreasing score, and of 2 equal scores the interval further right comes first.
    compute_resampling_scores defines the score.

    Args:
        x_values: (n,)-sized array of x-values, sorted in ascending order and without duplicates.
        fx_values: (n,)-sized array of the corresponding f(x)-values.
        dx_min: Smallest interval width worth splitting.
        dx_total: Width of the whole interval, which normalizes the interval widths.
        n_max: Largest number of midpoints to return.
    """
    # --- select the highest scores --------------
    scores = compute_resampling_scores(x_values, fx_values, dx_min, dx_total)
    n_intervals = len(scores)
    n_selected = min(n_max, n_intervals)
    if n_selected <= 0:
        return np.zeros(0, dtype=np.float64)
    # every score above the n_selected-th largest is selected; of the scores equal to it, those furthest right
    threshold = np.partition(scores, n_intervals - n_selected)[n_intervals - n_selected]
    is_selected = scores > threshold
    n_equal_wanted = n_selected - np.sum(is_selected)
    i = n_intervals - 1
    while n_equal_wanted > 0:
        if scores[i] == threshold:
            is_selected[i] = True
            n_equal_wanted -= 1
        i -= 1
    selected = np.flatnonzero(is_selected & (scores > 0.0))

    # --- order by decreasing score --------------
    # a stable ascending sort keeps equal scores in ascending x; reversing it puts equal scores in descending x
    order = np.argsort(scores[selected], kind="mergesort")[::-1]
    selected = selected[order]
    return 0.5 * (x_values[selected] + x_values[selected + 1])


@numba.njit
def compute_resampling_scores(
    x_values: np.ndarray, fx_values: np.ndarray, dx_min: float, dx_total: float
) -> np.ndarray:
    """Return the resampling score of each interval between 2 consecutive samples.

    The score of an interval is dfx_rel^2 / dx_rel + dx_rel^2, with dx and dfx normalized by their totals, and 0 for
    an interval no wider than dx_min:
       - dfx_rel^2 / dx_rel is the main term: splitting a mostly linear interval in 2 halves its score and divides
         it over both halves, while splitting an interval with a step doubles the score of the half with the step;
       - dx_rel^2 makes long, seemingly flat intervals get resampled once no interval looks more like a step.
    When every interval is flat (dfx_total == 0, e.g. a constant function), dx_rel^2 alone drives the resampling.

    The heights are summed the way Python's built-in sum() totals floats, so the scores are bit-identical to those
    of a plain Python computation.

    Args:
        x_values: (n,)-sized array of x-values, sorted in ascending order and without duplicates.
        fx_values: (n,)-sized array of the corresponding f(x)-values.
        dx_min: Smallest interval width worth splitting.
        dx_total: Width of the whole interval, which normalizes the interval widths.

    Returns:
        (n-1,)-sized array of scores, one per interval.
    """
    # --- total height ---------------------------
    # Neumaier's compensated sum, in the form that Python's built-in sum() uses for floats since Python 3.12
    n_intervals = len(x_values) - 1
    dfx_total = 0.0
    compensation = 0.0
    for i in range(n_intervals):
        dfx = abs(fx_values[i + 1] - fx_values[i])
        t = dfx_total + dfx
        if abs(dfx_total) >= abs(dfx):
            compensation += (dfx_total - t) + dfx
        else:
            compensation += (dfx - t) + dfx_total
        dfx_total = t
    if (compensation != 0.0) and math.isfinite(compensation):
        dfx_total += compensation

    # --- interval scores ------------------------

    scores = np.zeros(n_intervals, dtype=np.float64)
    for i in range(n_intervals):
        dx = x_values[i + 1] - x_values[i]
        if dx > dx_min:
            dx_rel = dx / dx_total
            if dfx_total > 0:
                dfx_rel = abs(fx_values[i + 1] - fx_values[i]) / dfx_total
            else:
                dfx_rel = 0.0
            scores[i] = (dfx_rel * dfx_rel / dx_rel) + (dx_rel * dx_rel)
    return scores


@numba.njit
def insert_sorted_samples(
    x_values: np.ndarray, fx_values: np.ndarray, new_x_values: np.ndarray, new_fx_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the samples with the new samples merged in, sorted by x, keeping 1 sample per x.

    Args:
        x_values: (n,)-sized array of x-values, sorted in ascending order and without duplicates.
        fx_values: (n,)-sized array of the corresponding f(x)-values.
        new_x_values: (m,)-sized array of new x-values, in any order.
        new_fx_values: (m,)-sized array of the corresponding f(x)-values.

    Returns:
        (x_values, fx_values)-tuple of the merged arrays. A new x-value that is already present is dropped, which
        matches the function cache, since it holds 1 value per x.
    """
    order = np.argsort(new_x_values, kind="mergesort")
    new_x_sorted = new_x_values[order]
    new_fx_sorted = new_fx_values[order]

    n, m = len(x_values), len(new_x_sorted)
    merged_x = np.empty(n + m, dtype=np.float64)
    merged_fx = np.empty(n + m, dtype=np.float64)
    i, j, k = 0, 0, 0
    while (i < n) or (j < m):
        if (j == m) or ((i < n) and (x_values[i] <= new_x_sorted[j])):
            x, fx = x_values[i], fx_values[i]
            i += 1
        else:
            x, fx = new_x_sorted[j], new_fx_sorted[j]
            j += 1
        if (k == 0) or (merged_x[k - 1] != x):
            merged_x[k] = x
            merged_fx[k] = fx
            k += 1
    return merged_x[:k], merged_fx[:k]
