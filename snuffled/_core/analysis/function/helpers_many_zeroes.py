import math


def compute_many_zeroes_score(n: int, n_max: int) -> float:
    """
    The MANY_ZEROES score indicates if we're 'suffering' from a large number of zeroes,
    and is calibrated on a log-like scale as follows:

        # of zeroes                    score

         1                              0.0
         3                             ~0.1
        n_max                           1.0

    score =  1 / ( log2(3)/(0.1*log2(n)) + c )

    with c = 1.0 - log2(3)/(0.1*log2(n_max))      with n_max = (x_max - x_min)/dx

    :return: (float) score in [0,1]
    """
    if n == 1:
        return 0.0
    elif n >= n_max:
        return 1.0
    else:

        def tmp(_n: int) -> float:
            # function mapping _n=1 --> 0.0
            #                  _n=3 --> 0.1
            # and logarithmic in _n
            return math.log2(_n) * (0.1 / math.log2(3))

        return 1 / (1 + (1 / tmp(n)) - (1 / tmp(n_max)))
