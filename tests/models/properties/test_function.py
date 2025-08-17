from snuffled._core.models import FunctionProperty, NamedArray, SnuffledFunctionProperties


def test_snuffled_function_properties():
    # --- arrange -----------------------------------------
    expected_names = list(FunctionProperty)

    # --- act ---------------------------------------------
    sfp = SnuffledFunctionProperties()

    # --- assert ------------------------------------------
    assert sfp.names() == expected_names
    assert isinstance(sfp, NamedArray)
