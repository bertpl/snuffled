import sys

# Re-exported via this package's __init__; __all__ marks it as a public re-export
# so ruff does not treat the conditional import as unused (F401).
__all__ = ["StrEnum"]

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
