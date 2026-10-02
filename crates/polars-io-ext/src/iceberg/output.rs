//! The `iceberg.v1` output (see `polars_io_ext_ffi::iceberg_v1` for the layout).
//!
//! Arrow data is built with `polars-arrow` and crosses only through the Arrow C Data Interface,
//! so it does not tie the plugin to the host's Polars version.
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray, Utf8ViewArray};
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::ffi;
use polars_arrow::offset::OffsetsBuffer;
use polars_error::{PolarsError, PolarsResult};
use polars_io_ext_ffi::common::{ArrowArrayStream, ArrowSchema, FfiError, FfiErrorKind};
use polars_io_ext_ffi::iceberg_v1::{Output, OutputHeader};

use crate::iceberg::host::PluginResult;

pub struct Resolved {
    /// Table schema of the selected snapshot, as struct fields.
    pub schema: Vec<Field>,
    pub files: FilesTable,
    /// `(physical, deleted)` row counts, if they can be used.
    pub row_count: Option<(u64, u64)>,
    /// Field IDs whose identity-partition constants could not be loaded, with the reason.
    pub constant_errors: Vec<(i32, String)>,
    /// `initial-default` values, as length-1 arrays.
    pub initial_defaults: Vec<(i32, Box<dyn Array>)>,
}

#[derive(Default)]
pub struct FilesTable {
    pub paths: Vec<String>,
    pub sizes: Vec<u64>,
    pub record_counts: Vec<u64>,
    pub deletes: Vec<Vec<DeleteRef>>,
    pub constants: Vec<(i32, Box<dyn Array>)>,
    /// `None` if statistics were not requested.
    pub stats: Option<Vec<(String, Box<dyn Array>)>>,
}

pub struct DeleteRef {
    pub kind: DeleteKind,
    pub path: String,
}

#[derive(Clone, Copy)]
pub enum DeleteKind {
    Position,
    DeletionVector,
}

impl DeleteKind {
    fn as_str(self) -> &'static str {
        match self {
            DeleteKind::Position => "position",
            DeleteKind::DeletionVector => "deletion_vector",
        }
    }
}

impl Resolved {
    pub fn into_ffi(self) -> PluginResult<Output> {
        let header = OutputHeader {
            row_count: self.row_count,
            constant_errors: self
                .constant_errors
                .into_iter()
                .map(|(id, msg)| (id as u32, msg))
                .collect(),
            statistics: self.files.stats.is_some(),
        };

        let files = self.files.into_struct().map_err(polars_err_to_ffi)?;
        let files_dtype = files.dtype().clone();
        let files_batches: Vec<PolarsResult<Box<dyn Array>>> = if files.is_empty() {
            vec![]
        } else {
            vec![Ok(files.boxed())]
        };
        let files_stream = ffi::export_iterator(
            Box::new(files_batches.into_iter()),
            Field::new("files".into(), files_dtype, false),
        );

        let mut table_fields = vec![];
        let mut table_arrays = vec![];
        if !self.initial_defaults.is_empty() {
            let defaults = keyed_struct(self.initial_defaults, 1).map_err(polars_err_to_ffi)?;
            table_fields.push(Field::new(
                "initial_defaults".into(),
                defaults.dtype().clone(),
                false,
            ));
            table_arrays.push(defaults.boxed());
        }
        let table_values_stream = if table_fields.is_empty() {
            ffi::export_iterator(
                Box::new(std::iter::empty()),
                Field::new("table_values".into(), ArrowDataType::Struct(vec![]), false),
            )
        } else {
            let dtype = ArrowDataType::Struct(table_fields);
            let values = StructArray::try_new(dtype.clone(), 1, table_arrays, None)
                .map_err(polars_err_to_ffi)?;
            ffi::export_iterator(
                Box::new(std::iter::once(Ok(values.boxed()))),
                Field::new("table_values".into(), dtype, false),
            )
        };

        Ok(Output {
            header: header.into_ffi(),
            schema: export_schema(self.schema),
            files: unsafe { ArrowArrayStream::transmute_from(files_stream) },
            table_values: unsafe { ArrowArrayStream::transmute_from(table_values_stream) },
        })
    }
}

impl FilesTable {
    fn into_struct(self) -> PolarsResult<StructArray> {
        let n = self.paths.len();

        let delete_dtype = ArrowDataType::Struct(vec![
            Field::new("kind".into(), ArrowDataType::Utf8View, false),
            Field::new("path".into(), ArrowDataType::Utf8View, false),
        ]);

        let mut fields = vec![
            Field::new("path".into(), ArrowDataType::Utf8View, false),
            Field::new("size".into(), ArrowDataType::UInt64, false),
            Field::new("record_count".into(), ArrowDataType::UInt64, false),
        ];
        let mut arrays: Vec<Box<dyn Array>> = vec![
            Utf8ViewArray::from_slice_values(&self.paths).boxed(),
            PrimitiveArray::<u64>::from_vec(self.sizes).boxed(),
            PrimitiveArray::<u64>::from_vec(self.record_counts).boxed(),
        ];

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0i64);
        let mut kinds = vec![];
        let mut paths = vec![];
        for deletes in &self.deletes {
            for d in deletes {
                kinds.push(d.kind.as_str());
                paths.push(d.path.as_str());
            }
            offsets.push(kinds.len() as i64);
        }
        let delete_values = StructArray::try_new(
            delete_dtype.clone(),
            kinds.len(),
            vec![
                Utf8ViewArray::from_slice_values(&kinds).boxed(),
                Utf8ViewArray::from_slice_values(&paths).boxed(),
            ],
            None,
        )?;
        let deletes_dtype = ListArray::<i64>::default_datatype(delete_dtype);
        fields.push(Field::new("deletes".into(), deletes_dtype.clone(), false));
        arrays.push(
            ListArray::<i64>::try_new(
                deletes_dtype,
                OffsetsBuffer::try_from(offsets)?,
                delete_values.boxed(),
                None,
            )?
            .boxed(),
        );

        if !self.constants.is_empty() {
            let constants = keyed_struct(self.constants, n)?;
            fields.push(Field::new(
                "constants".into(),
                constants.dtype().clone(),
                false,
            ));
            arrays.push(constants.boxed());
        }

        if let Some(stats) = self.stats
            && !stats.is_empty()
        {
            let stats_fields = stats
                .iter()
                .map(|(name, a)| Field::new(name.as_str().into(), a.dtype().clone(), true))
                .collect();
            let dtype = ArrowDataType::Struct(stats_fields);
            let stats = StructArray::try_new(
                dtype.clone(),
                n,
                stats.into_iter().map(|(_, a)| a).collect(),
                None,
            )?;
            fields.push(Field::new("stats".into(), dtype, false));
            arrays.push(stats.boxed());
        }

        StructArray::try_new(ArrowDataType::Struct(fields), n, arrays, None)
    }
}

/// A struct array with one child per field ID, named by the ID.
fn keyed_struct(values: Vec<(i32, Box<dyn Array>)>, len: usize) -> PolarsResult<StructArray> {
    let fields = values
        .iter()
        .map(|(id, a)| Field::new(id.to_string().into(), a.dtype().clone(), true))
        .collect();
    StructArray::try_new(
        ArrowDataType::Struct(fields),
        len,
        values.into_iter().map(|(_, a)| a).collect(),
        None,
    )
}

/// Export table schema fields as an Arrow struct schema.
fn export_schema(fields: Vec<Field>) -> ArrowSchema {
    let schema =
        ffi::export_field_to_c(&Field::new("".into(), ArrowDataType::Struct(fields), false));
    unsafe { ArrowSchema::transmute_from(schema) }
}

fn polars_err_to_ffi(e: PolarsError) -> FfiError {
    FfiError::new(FfiErrorKind::OTHER, e.to_string())
}
