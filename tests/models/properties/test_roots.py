from snuffled._core.models import NamedArray, RootProperty, SnuffledRootProperties


def test_snuffled_function_properties():
    # --- arrange -----------------------------------------
    expected_names = list(RootProperty)

    # --- act ---------------------------------------------
    srp = SnuffledRootProperties()

    # --- assert ------------------------------------------
    assert srp.names() == expected_names
    assert isinstance(srp, NamedArray)
