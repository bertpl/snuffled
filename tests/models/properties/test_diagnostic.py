from snuffled._core.models import Diagnostic, NamedArray, SnuffledDiagnostics


def test_snuffled_function_properties():
    # --- arrange -----------------------------------------
    expected_names = list(Diagnostic)

    # --- act ---------------------------------------------
    sd = SnuffledDiagnostics()

    # --- assert ------------------------------------------
    assert sd.names() == expected_names
    assert isinstance(sd, NamedArray)
