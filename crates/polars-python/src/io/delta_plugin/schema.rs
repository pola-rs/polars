//! The schema-only answer of the Delta plugin: the table schema and the resolved version.
use polars::prelude::*;
use polars_arrow::ffi;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_io_ext_ffi::delta_v1::{Output, OutputHeader};

pub(super) fn schema_only(output: Output) -> PolarsResult<(Schema, i64, String)> {
    let Output {
        header,
        schema,
        files: _,
    } = output;
    let header = OutputHeader::from_ffi(&header);
    polars_ensure!(
        !schema.is_released(),
        ComputeError: "delta plugin returned a released schema"
    );
    let schema: ffi::ArrowSchema = unsafe { schema.transmute_into() };
    let field = unsafe { ffi::import_field_from_c(&schema)? };
    let ArrowDataType::Struct(fields) = field.dtype else {
        polars_bail!(
            ComputeError:
            "delta plugin returned a non-struct schema: {:?}", field.dtype
        )
    };
    let schema = fields.iter().map(Field::from).collect();
    Ok((schema, header.resolved_version, header.table_id))
}
