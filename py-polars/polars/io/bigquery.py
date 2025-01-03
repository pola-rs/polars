from __future__ import annotations

import functools
import io
import json
from typing import TYPE_CHECKING

import polars._reexport as pl
import polars.expr.meta
import polars.functions
import polars.io.ipc
from polars._utils import polars_version
from polars.datatypes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Field,
    Float64,
    Int64,
    List,
    String,
    Struct,
    Time,
)
from polars.dependencies import bigquery, bigquery_storage_v1

if TYPE_CHECKING:
    from typing import Any

    import google.api_core.client_info  # type: ignore[import-untyped]
    import google.api_core.gapic_v1.client_info  # type: ignore[import-untyped]
    import google.auth.credentials

    from polars import LazyFrame
    from polars.datatypes import DataType


def scan_bigquery(
    source: str,
    *,
    credentials: google.auth.credentials.Credentials | None = None,
    billing_project_id: str | None = None,
) -> LazyFrame:
    """
    Lazily read from a BigQuery table.

    Parameters
    ----------
    source
        A BigQuery table, in the form 'project.dataset.table' or
        'dataset.table' (if ``billing_project_id`` is set).
    credentials
        A google.auth Credentials object to use to authenticate to the
        BigQuery APIs.
    billing_project_id
        The Google Cloud project ID to use for billing quotas.

        Note: this doesn't have to be the same project as the one where the
        table resides, allowing public datasets to be read while billing the
        BigQuery Storage API read session to your personal billing account.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for a whole BigQuery table.

    >>> table_path = "bigquery-public-data.usa_names.usa_1910_2013"
    >>> billing_project_id = "my-project"
    >>> pl.scan_bigquery(
    ...     table_path,
    ...     billing_project_id=billing_project_id,
    ... ).collect()  # doctest: +SKIP

    Many filters can be pushed down using the BigQuery Storage Read API.

    >>> table_path = "bigquery-public-data.usa_names.usa_1910_2013"
    >>> billing_project_id = "my-project"
    >>> pl.scan_bigquery(
    ...     table_path,
    ...     billing_project_id=billing_project_id,
    ... ).filter(pl.col("year") == 2015).collect()  # doctest: +SKIP

    Injestion time partitioned tables can be filtered by the _PARTITIONDATE
    psuedocolumn.

    >>> table_path = "my-project.my_dataset.injestion_time_partitioned_table"
    >>> pl.scan_bigquery(table_path).filter(
    ...     pl.col("_PARTITIONDATE) > datetime.date(2025, 1, 1)
    ... ).collect()  # doctest: +SKIP
    """
    bq_client = bigquery.Client(
        project=billing_project_id,
        credentials=credentials,
        client_info=_create_client_info(),
    )
    bqstorage_client = bigquery_storage_v1.BigQueryReadClient(
        credentials=credentials, client_info=_create_client_info_gapic()
    )

    # Theoretically, we could avoid the REST API because the first request to
    # the BigQuery Storage Read API provides an Arrow schema, but we want to
    # delay starting a read session until after we know which columns and row
    # filters we're using.
    table = bq_client.get_table(source)
    pl_schema = _bigquery_to_polars_types(table)

    table_path, billing_project_id = _source_to_table_path_and_billing_project(
        source, default_project_id=billing_project_id
    )
    func = functools.partial(
        _scan_bigquery_impl,
        bqstorage_client,
        list(pl_schema.keys()),
        table_path,
        billing_project_id,
    )
    return pl.LazyFrame._scan_python_function(pl_schema, func, pyarrow=False)


def _scan_bigquery_impl(
    bqstorage_client: bigquery_storage_v1.BigQueryReadClient,
    original_columns: list[str],
    table_path: str,
    billing_project_id: str,
    with_columns: list[str] | None,
    predicate: str | None,
    n_rows: int | None,
    batch_size: int | None,
    **kwargs: dict[str, Any],
) -> pl.DataFrame:
    """
    Generator function that creates the source.

    This function will be registered as IO source.

    ``n_rows`` and ``batch_size`` are not supported by the BigQuery Storage
    Read API. These parameters are ignored.
    """
    read_request = _to_read_request(
        original_columns, table_path, billing_project_id, with_columns, predicate
    )
    session = bqstorage_client.create_read_session(read_request)
    stream = io.BytesIO()
    arrow_schema = session.arrow_schema.serialized_schema
    stream.write(arrow_schema)

    if len(session.streams) != 0:
        reader = bqstorage_client.read_rows(session.streams[0].name)
        for message in reader:
            stream.write(message.arrow_record_batch.serialized_record_batch)

    stream.seek(0)
    return polars.io.ipc.read_ipc_stream(stream)


_BINARY_OPS = {
    "Or": "OR",
    "And": "AND",
    "Eq": "=",
    "Gt": ">",
    "GtEq": ">=",
    "Lt": "<",
    "LtEq": "<=",
}


def _create_user_agent() -> str:
    """
    Polars version information to include in the user-agent header.

    Including something in the user-agent header indicating requests to BigQuery
    originated from polars can help the BigQuery team prioritize improvements to
    the Polars + BigQuery connector.
    """
    return f"polars/{polars_version.get_polars_version()}"


def _create_client_info() -> google.api_core.client_info.ClientInfo:
    """User-agent for REST API clients."""
    from google.api_core.client_info import ClientInfo

    return ClientInfo(
        user_agent=_create_user_agent(),
    )


def _create_client_info_gapic() -> google.api_core.gapic_v1.client_info.ClientInfo:
    """User-agent for gRPC API clients."""
    from google.api_core.gapic_v1.client_info import ClientInfo

    return ClientInfo(
        user_agent=_create_user_agent(),
    )


def _bigquery_to_polars_type(field: bigquery.SchemaField) -> DataType:
    """Convert a BigQuery type into a polars type.

    Note: the REST API uses the names from the Legacy SQL data types, but if
    user-entered it may include the newer aliases.
    (https://cloud.google.com/bigquery/docs/data-types).
    """
    # Check for BQ ARRAY (polars List) type first because it's not returned as
    # a separate type in the BQ REST API. Instead, it uses the 'mode' field to
    # indicate an ARRAY type.
    if field.mode.casefold() == "repeated":
        inner_type = _bigquery_to_polars_type(
            bigquery.SchemaField(
                field.name,
                field.field_type,
                fields=field.fields,
                mode="NULLABLE",
            ),
        )
        return List(inner_type)

    type_ = field.field_type.casefold()

    if type_ in ("bool", "boolean"):
        return Boolean()
    if type_ == "bytes":
        return Binary()
    if type_ == "date":
        return Date()
    if type_ == "datetime":
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        # https://stackoverflow.com/a/47724366/101923
        return Datetime(time_unit="us")
    if type_ == "geography":
        # TODO: support geopolars data types, if available
        return String()
    if type_ in ("float", "float64"):
        return Float64()
    if type_ in ("integer", "int64"):
        return Int64()
    if type_ in ("numeric", "decimal"):
        # BigQuery NUMERIC type has precision 38 and scale 9.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
        return Decimal(precision=38, scale=9)
    if type_ in ("record", "struct"):
        polars_fields = [
            Field(field.name, _bigquery_to_polars_type(field)) for field in field.fields
        ]
        return Struct(polars_fields)
    if type_ == "string":
        return String()
    if type_ == "time":
        return Time()
    if type_ == "timestamp":
        return Datetime(time_unit="us", time_zone="utc")

    message = f"got unexpected BigQuery type: {type_}"
    raise TypeError(message)


def _bigquery_to_polars_types(table: bigquery.Table) -> dict[str, DataType]:
    """Convert a BigQuery table into a Polars schema."""
    pl_schema = {}
    for field in table.schema:
        pl_schema[field.name] = _bigquery_to_polars_type(field)

    # If table is ingestion time partitioned, add pseudocolumn for _PARTITIONDATE
    # to allow for partition filters. See:
    # https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time
    if (
        time_partitioning := table.time_partitioning
    ) is not None and time_partitioning.field is None:
        # TODO: also include _PARTITIONTIME when a Datetime(time_unit="us",
        # time_zone="utc") predicate can be pushed down.
        pl_schema["_PARTITIONDATE"] = Date()

    return pl_schema


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a polars expression JSON into SQL-like."""
    if "Dyn" in literal_json:
        return _json_literal_to_sql(literal_json["Dyn"])

    if "Scalar" in literal_json:
        return _json_literal_to_sql(literal_json["Scalar"])

    if "dtype" in literal_json and "value" in literal_json:
        return _json_literal_to_sql(literal_json["value"])

    polars_type, value = literal_json.popitem()

    # TODO: support more polars types.
    if polars_type == "DateTime":
        # TODO: check timezone, too
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        # https://stackoverflow.com/a/47724366/101923
        ticks = value[0]
        units = value[1]
        if units == "Microseconds":
            return f"TIMESTAMP_MICROS({ticks})"
        return None
    if polars_type == "Date":
        return f"DATE(TIMESTAMP_SECONDS({value} * 86400))"

    if polars_type in ("String", "Int", "Float"):
        return repr(value)

    return None


def _json_function_to_sql(function_json: dict[str, Any]) -> str | None:
    """Converts a polars function call into the equivalent BigQuery syntax."""
    # So far, only boolean output functions are supported.
    boolean_function_name = function_json.get("function", {}).get("Boolean", None)
    if boolean_function_name is None:
        return None

    if boolean_function_name == "IsNull":
        input_ = _json_expr_to_row_restriction(function_json["input"][0])
        if input_ is None:
            return None
        return f"({input_} IS NULL)"

    if boolean_function_name == "IsNotNull":
        input_ = _json_expr_to_row_restriction(function_json["input"][0])
        if input_ is None:
            return None
        return f"({input_} IS NOT NULL)"

    return None


def _json_expr_to_row_restriction(expr_json: dict[str, Any]) -> str | None:
    """Create a row restriction to filter rows.

    Returns None if unknown operators are found and can't guarantee a superset of rows.
    """
    # TODO: Use iterative compilation to support deeper trees. Python 3.12+
    # has a pretty strict 1000 depth limit. See:
    # https://github.com/python/cpython/issues/112282
    if "BinaryExpr" in expr_json:
        binary_expr = expr_json["BinaryExpr"]
        left = _json_expr_to_row_restriction(binary_expr["left"])
        right = _json_expr_to_row_restriction(binary_expr["right"])

        polars_op = binary_expr.get("op", None)
        if polars_op is None:
            return None

        # TODO: lookup table instead of iterating through all possible types
        if polars_op == "And":
            # With 'And', filtering by just one of the two children will still
            # give a superset of the filtered rows. The rest of the filters can
            # be applied by polars instead of BigQuery.
            if left is None:
                return right
            if right is None:
                return left

            return f"({left} AND {right})"

        # The rest of these operators need both left and right to be converted
        # correctly for correctness.
        if left is None or right is None:
            return None

        sql_op = _BINARY_OPS.get(polars_op)
        if sql_op is None:
            return None
        return f"({left} {sql_op} {right})"

    if "Function" in expr_json:
        function_json = expr_json["Function"]
        return _json_function_to_sql(function_json)

    if "Column" in expr_json:
        return f"`{expr_json['Column']}`"  # TODO: do we need to escape any characters?

    if "Literal" in expr_json:
        literal = expr_json["Literal"]
        return _json_literal_to_sql(literal)

    # Got some op that we don't know how to handle.
    return None


def _source_to_table_path_and_billing_project(
    source: str, *, default_project_id: str | None
) -> tuple[str, str]:
    """Converts source from project.dataset.table format to a BigQuery Storage path."""
    parts = source.split(".")
    if len(parts) == 3:
        if default_project_id is not None:
            billing_project_id = default_project_id
        else:
            billing_project_id = parts[0]

        return (
            f"projects/{parts[0]}/datasets/{parts[1]}/tables/{parts[2]}",
            billing_project_id,
        )
    elif len(parts) == 2:
        if default_project_id is None:
            message = f"source {source!r} is missing project and no billing_project_id was set."
            raise ValueError(message)

        billing_project_id = default_project_id
        return (
            f"projects/{default_project_id}/datasets/{parts[0]}/tables/{parts[1]}",
            billing_project_id,
        )

    message = (
        "expected 2 or 3 parts in the form of project.dataset.table "
        "(project optional if billing_project_id is set), but got "
        f"{len(parts)} parts in source: {source!r}."
    )
    raise ValueError(message)


def _predicate_to_row_restriction(predicate: pl.Expr) -> str | None:
    predicate_json_file = io.BytesIO()
    predicate.meta.serialize(predicate_json_file, format="json")
    predicate_json_file.seek(0)
    predicate_json = json.load(predicate_json_file)
    return _json_expr_to_row_restriction(predicate_json)


def _to_read_request(
    original_columns: list[str],
    table_path: str,
    billing_project_id: str,
    with_columns: list[str] | None,
    predicate: str | None,
) -> bigquery_storage_v1.types.CreateReadSessionRequest:
    """Create gRPC request to read a BigQuery table."""
    import google.cloud.bigquery_storage_v1.types as types  # type: ignore[import-untyped]

    read_request = types.CreateReadSessionRequest()
    read_session = types.ReadSession()
    read_options = types.ReadSession.TableReadOptions()

    if predicate is not None:
        predicate_expr = pl.Expr.deserialize(predicate)
        read_options.row_restriction = _predicate_to_row_restriction(predicate_expr)

    if with_columns is not None:
        read_options.selected_fields = with_columns
    else:
        read_options.selected_fields = original_columns

    read_session.read_options = read_options
    read_session.table = table_path
    read_session.data_format = types.DataFormat.ARROW

    read_request.parent = f"projects/{billing_project_id}"
    read_request.read_session = read_session

    # single-threaded for simplicity, consider increasing this to the number of
    # parallel workers.
    read_request.max_stream_count = 1
    return read_request
