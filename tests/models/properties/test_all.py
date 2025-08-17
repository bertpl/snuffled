from snuffled._core.models.properties import (
    Diagnostic,
    FunctionProperty,
    RootProperty,
    SnuffledDiagnostics,
    SnuffledFunctionProperties,
    SnuffledProperties,
    SnuffledRootProperties,
)


def test_snuffled_properties_construction_extraction_getters():
    # --- arrange -----------------------------------------
    sfp = SnuffledFunctionProperties([1, 2, 3, 4, 5])
    srp = SnuffledRootProperties([11, 12, 13, 14, 15])
    sdp = SnuffledDiagnostics([21, 22, 23])

    # --- act ---------------------------------------------
    sp = SnuffledProperties(
        function_props=sfp,
        root_props=srp,
        diagnostics=sdp,
    )

    # --- assert ------------------------------------------
    assert sp.function_props == sfp
    assert sp.root_props == srp
    assert sp.diagnostics == sdp

    for fp in FunctionProperty:
        assert sp.function_props[fp] == sfp[fp]

    for rp in RootProperty:
        assert sp.root_props[rp] == srp[rp]

    for d in Diagnostic:
        assert sp.diagnostics[d] == sdp[d]
