==========
Exceptions
==========
.. currentmodule:: polars.exceptions

Errors
~~~~~~

.. autosummary::
    :toctree: api/
    :nosignatures:

    PolarsError
    ColumnNotFoundError
    ComputeError
    DuplicateError
    InvalidOperationError
    ModuleUpgradeRequiredError
    NoDataError
    NoRowsReturnedError
    OutOfBoundsError
    ParameterCollisionError
    RowsError
    SQLInterfaceError
    SQLSyntaxError
    SchemaError
    SchemaFieldNotFoundError
    ShapeError
    StringCacheMismatchError
    StructFieldNotFoundError
    TooManyRowsReturnedError
    UnsuitableSQLError

Warnings
~~~~~~~~

.. autosummary::
    :toctree: api/
    :nosignatures:

    PolarsWarning
    CategoricalRemappingWarning
    ChronoFormatWarning
    CustomUFuncWarning
    DataOrientationWarning
    MapWithoutReturnDtypeWarning
    PerformanceWarning
    PolarsInefficientMapWarning
    UnstableWarning

Panic
~~~~~

.. autosummary::
    :toctree: api/
    :nosignatures:

    PanicException
