from snuffled._core.models.base import NamedArray

from ._diagnostic import Diagnostic, SnuffledDiagnostics
from ._function import FunctionProperty, SnuffledFunctionProperties
from ._root import RootProperty, SnuffledRootProperties


class SnuffledProperties(NamedArray):
    """
    Class merging the root, function & diagnostic properties
      - SnuffledRootProperties
      - SnuffledFunctionProperties
      - SnuffledDiagnostics
    """

    def __init__(
        self,
        function_props: SnuffledFunctionProperties | None = None,
        root_props: SnuffledRootProperties | None = None,
        diagnostics: SnuffledDiagnostics | None = None,
    ):
        super().__init__(names=list(FunctionProperty) + list(RootProperty) + list(Diagnostic))
        if function_props:
            for key, value in function_props.as_dict().items():
                self[key] = value
        if root_props:
            for key, value in root_props.as_dict().items():
                self[key] = value
        if diagnostics:
            for key, value in diagnostics.as_dict().items():
                self[key] = value

    @property
    def function_props(self) -> SnuffledFunctionProperties:
        """
        Returns the function properties as a SnuffledFunctionProperties object.
        """
        props = SnuffledFunctionProperties()
        for name in FunctionProperty:
            props[name] = self[name]
        return props

    @property
    def root_props(self) -> SnuffledRootProperties:
        """
        Returns the root properties as a SnuffledRootProperties object.
        """
        props = SnuffledRootProperties()
        for name in RootProperty:
            props[name] = self[name]
        return props

    @property
    def diagnostics(self) -> SnuffledDiagnostics:
        """
        Returns the diagnostics as a SnuffledDiagnostics object.
        """
        props = SnuffledDiagnostics()
        for name in Diagnostic:
            props[name] = self[name]
        return props
