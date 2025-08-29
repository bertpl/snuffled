from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Root:
    """
    Class to describe a root of a function and a few elementary properties.
    Can only model a root with odd multiplicity, i.e. we should have a sign-switch at the root.

    (x_min, x_max) describe location & extent (width) of the root, with 2 cases existing:

      - f(x_min) == f(x_max) == 0.0   --> x_min & x_max represent the outermost values for which f(.)==0.0

      - f(x_min)*f(x_max) < 0         --> the root does not go exactly through 0, but switches sign immediately;
                                      --> x_min, x_max represent the nearest distinct x-values for which we see this
                                          sign switch
    """

    # primary properties
    x_min: float
    x_max: float
    deriv_sign: int  # +1 if df/dx > 0, -1 if df/dx < 0   (0.0 not allowed)

    def __post_init__(self):
        # some simple validation
        if abs(self.deriv_sign) != 1.0:
            raise ValueError(f"deriv_sign should be either -1 or +1; here {self.deriv_sign}")
        if self.x_min > self.x_max:
            raise ValueError(f"x_min<=x_max expected; here {self.x_min}>{self.x_max}")

    @cached_property
    def width(self) -> float:
        return self.x_max - self.x_min
