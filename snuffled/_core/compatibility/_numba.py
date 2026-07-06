from collections.abc import Callable

try:
    import numba  # ty: ignore[unresolved-import] -- optional extra; absent in the numba-off env


except ImportError:
    # dummy decorator that will replace numba.jit and numba.njit
    def dummy_decorator(*args: object, **kwargs: object) -> Callable:
        # dummy decorator that does nothing and can be used with or without arguments
        if len(args) == 1 and isinstance(args[0], Callable):
            # decorator used without arguments
            return args[0]

        # decorator used with arguments
        def decorator(func: Callable) -> Callable:
            return func

        return decorator

    # create a dummy numba object with numba.jit and numba.njit dummy decorators
    class Numba:
        jit = dummy_decorator
        njit = dummy_decorator

    numba = Numba  # ty: ignore[invalid-assignment] -- fallback shim replaces the real numba module
