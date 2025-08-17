import numpy as np


def robust_sign_estimate(values: list[float] | np.ndarray) -> int:
    """
    Makes robust estimate of dominant sign of list of values, in presence of noise, underflow, ...

    We use sign(q25+q75) as a robust estimate.
      -> In presence of noise, some values might have opposite sign as the underlying data, but we should still
          see the sign of the underlying data as a bias in q25 & q75 values
      -> This metric is robust to up to 74% of 0s in the data due to numerical underflow
      -> We're robust wrt to noise spikes, due to using more central quantiles, instead of min & max.

    :param values: list or 1d numpy array of floats/ints.
    """
    return int(np.sign(np.quantile(values, 0.75) + np.quantile(values, 0.25)))
