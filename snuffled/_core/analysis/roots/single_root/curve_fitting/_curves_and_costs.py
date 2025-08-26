import numba
import numpy as np


@numba.njit(inline="always")
def fitting_cost(x: np.ndarray, fx: np.ndarray, a: float, b: float, c: float, reg: float) -> float:
    """
    L1-cost of fitting fitting_curve to (x,fx)-values, where all x>0.
    A regularization term is added that should help keep c=1.0 and b=0.0 unless the data convincingly says
    otherwise.
    """
    fx_pred = fitting_curve(x, a, b, c)
    reg_term = reg * (abs(b) + abs(np.log10(c)))
    return float(np.mean(np.abs(fx - fx_pred)) + reg_term)


@numba.njit(inline="always")
def fitting_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """curve g(x) = a * (b + (1-b)**(x^c), assuming x>0"""
    return a * (b + (1 - b) * (np.pow(x, c)))
