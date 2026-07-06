# Re-exported via this package's __init__; __all__ marks it as a public re-export
# so ruff does not treat the import as unused (F401). StrEnum is stdlib from 3.11
# (the package's minimum), so no backport is needed.
from enum import StrEnum

__all__ = ["StrEnum"]
