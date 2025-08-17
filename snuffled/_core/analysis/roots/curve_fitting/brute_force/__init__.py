"""
This package contains functionality for brute-force finding the optimum + uncertainty of a curve fitting problem,
focusing on parameters b & c.  This can be used for debugging / didactical purposes to compare results of the more
specialized (faster but approximate) methods.
"""

from ._fit_optimal_with_uncertainty import fit_curve_with_uncertainty_brute_force
