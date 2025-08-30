from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Root:
    """
     Class to describe a root of a function and a few elementary properties.

     (x_min, x_max)  Describes an interval that encloses the root (see also fx_min, fx_max below).
                     Except during the initial stages of root finding, this interval can be assumed to be the tightest
                     enclosing interval for the root we were able to determine:
                       - either because x_min, x_max are two subsequent floats with none in between
                       - or because intermediate x values have f(x)==0

    (fx_min, fx_max) Represent f(x_min) != 0.0 and f(x_max) != 0.0.  If both signs are different this root is an 'odd'
                     root, otherwise it is 'even' and is for all practical cases dismissed (as this package is not
                     intended to analyze roots with even multiplicity)

    x                Value in [x_min, x_max] that is the closest estimate of (the middle of) the true root.

    fx               Represents f(x).  Often this will be 0.0, except when we have a very sharply defined root
                     Where f switches sign without going through 0.0 exactly.  Note that fx!=0.0 can only happen
                     in case sign(fx_min) != sign(fx_max).
    """

    # primary properties
    x_min: float
    x: float
    x_max: float

    fx_min: float
    fx: float
    fx_max: float

    def __post_init__(self):
        if self.x_min > self.x:
            raise ValueError(f"x_min<=x expected; here {self.x_min}>{self.x}")
        if self.x > self.x_max:
            raise ValueError(f"x<=x_max expected; here {self.x}>{self.x_max}")
        if (self.x == self.x_min) and (self.fx != self.fx_min):
            raise ValueError(
                f"if x==x_min, we expect fx==fx_min; here {self.x}=={self.x_min} but {self.fx}!={self.fx_min}"
            )
        if (self.x == self.x_max) and (self.fx != self.fx_max):
            raise ValueError(
                f"if x==x_max, we expect fx==fx_max; here {self.x}=={self.x_max} but {self.fx}!={self.fx_max}"
            )
        if (self.deriv_sign == 0) and (self.fx != 0.0):
            raise ValueError(
                "If sign(fx_min)==sign(fx_max), we expect fx==0.0; "
                + f"here sign(fx_min)==sign(fx_max)=={int(np.sign(self.fx_min))}, but fx=={self.fx}."
            )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @cached_property
    def deriv_sign(self) -> int:
        # +1 if fx_min<0 and fx_max>0
        # -1 if fx_min>0 and fx_max<0
        #  0 otherwise
        return int(np.sign(int(np.sign(self.fx_max)) - int(np.sign(self.fx_min))))
