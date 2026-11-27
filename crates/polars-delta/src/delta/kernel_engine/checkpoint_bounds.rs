//! Physical encodings that need adaptation before Kernel can replay a checkpoint.
use std::sync::{Arc, Mutex};

use delta_kernel::arrow::array::{Array, ArrayRef, RecordBatch, StructArray, new_null_array};
use delta_kernel::arrow::datatypes::DataType as ArrowDataType;
use delta_kernel::engine::arrow_conversion::TryFromArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::object_store::path::Path;
use delta_kernel::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
#[allow(deprecated)]
use delta_kernel::parquet::arrow::async_reader::ParquetObjectReader;
use delta_kernel::parquet::basic::Type;
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::{EngineData, FileMeta, KernelResult, ParquetFooter};
use polars_utils::aliases::PlHashMap;
use url::Url;

use super::runtime;
use super::store::HostStore;

pub(super) struct CheckpointLayout {
    // Includes typed partition instants as well as data-skipping bounds.
    pub(super) unsafe_bounds: Vec<Vec<String>>,
    pub(super) binary_partition_key: bool,
    pub(super) binary_partition_value: bool,
}

pub(super) struct CheckpointBounds {
    store: Arc<HostStore>,
    // The host keeps the same pinned generation for the footer and body.
    by_file: Mutex<PlHashMap<Url, Arc<CheckpointLayout>>>,
}

impl CheckpointBounds {
    pub(super) fn new(store: Arc<HostStore>) -> Arc<Self> {
        Arc::new(Self {
            store,
            by_file: Mutex::new(PlHashMap::default()),
        })
    }

    pub(super) fn wanted(schema: &StructType) -> bool {
        let Some(DataType::Struct(add)) = schema.field("add").map(|f| f.data_type()) else {
            return false;
        };
        if add
            .field("partitionValues_parsed")
            .is_some_and(|f| contains_timestamp(f.data_type()))
        {
            return true;
        }
        let Some(DataType::Struct(stats)) = add.field("stats_parsed").map(|f| f.data_type()) else {
            return false;
        };
        ["minValues", "maxValues"].into_iter().any(|name| {
            stats
                .field(name)
                .is_some_and(|f| contains_timestamp(f.data_type()))
        })
    }

    pub(super) fn footer(&self, location: &Url, alias: &FileMeta) -> KernelResult<ParquetFooter> {
        let metadata = self.read(location, alias)?;
        let schema = Arc::new(StructType::try_from_arrow(metadata.schema().as_ref())?);
        Ok(ParquetFooter { schema })
    }

    pub(super) fn layout(
        &self,
        location: &Url,
        alias: &FileMeta,
    ) -> KernelResult<Arc<CheckpointLayout>> {
        if let Some(layout) = self.by_file.lock().unwrap().get(location).cloned() {
            return Ok(layout);
        }
        self.read(location, alias)?;
        Ok(self.by_file.lock().unwrap()[location].clone())
    }

    fn read(&self, location: &Url, alias: &FileMeta) -> KernelResult<ArrowReaderMetadata> {
        let path = Path::from_url_path(alias.location.path())?;
        #[allow(deprecated)]
        let mut reader = ParquetObjectReader::new(self.store.clone(), path);
        if alias.size != 0 {
            reader = reader.with_file_size(alias.size);
        }
        let metadata = runtime::block_on(
            None,
            ArrowReaderMetadata::load_async(
                &mut reader,
                ArrowReaderOptions::new().with_skip_arrow_metadata(true),
            ),
        )?;
        let unsafe_bounds = metadata
            .parquet_schema()
            .columns()
            .iter()
            .filter_map(|column| {
                let path = column.path().parts();
                // The default reader decodes INT96 as nanoseconds before casting to
                // microseconds. It can wrap old dates into plausible newer instants.
                (column.physical_type() == Type::INT96
                    && (matches!(path, [add, stats, bound, _, ..]
                        if add == "add" && stats == "stats_parsed"
                            && matches!(bound.as_str(), "minValues" | "maxValues"))
                        || matches!(path, [add, partitions, _]
                            if add == "add" && partitions == "partitionValues_parsed")))
                .then(|| path.to_vec())
            })
            .collect();
        let (binary_partition_key, binary_partition_value) =
            binary_partition_fields(metadata.schema());
        self.by_file.lock().unwrap().insert(
            location.clone(),
            Arc::new(CheckpointLayout {
                unsafe_bounds,
                binary_partition_key,
                binary_partition_value,
            }),
        );
        Ok(metadata)
    }

    pub(super) fn suppress(
        batch: Box<dyn EngineData>,
        paths: &[Vec<String>],
    ) -> KernelResult<Box<dyn EngineData>> {
        let batch = ArrowEngineData::try_from_engine_data(batch)?;
        let batch: RecordBatch = (*batch).into();
        let array = StructArray::from(batch);
        let paths = paths.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let array = suppress_struct(&array, &paths)?;
        Ok(Box::new(ArrowEngineData::new(array.into())))
    }
}

fn binary_partition_fields(schema: &delta_kernel::arrow::datatypes::Schema) -> (bool, bool) {
    let Ok(add) = schema.field_with_name("add") else {
        return (false, false);
    };
    let ArrowDataType::Struct(fields) = add.data_type() else {
        return (false, false);
    };
    let Some(partitions) = fields
        .iter()
        .find(|field| field.name() == "partitionValues")
    else {
        return (false, false);
    };
    let ArrowDataType::Map(entries, _) = partitions.data_type() else {
        return (false, false);
    };
    let ArrowDataType::Struct(fields) = entries.data_type() else {
        return (false, false);
    };
    let binary = |field: &delta_kernel::arrow::datatypes::Field| {
        matches!(
            field.data_type(),
            ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView
        )
    };
    match fields.as_ref() {
        [key, value] => (binary(key), binary(value)),
        _ => (false, false),
    }
}

fn contains_timestamp(dtype: &DataType) -> bool {
    match dtype {
        DataType::Struct(fields) => fields.fields().any(|f| contains_timestamp(f.data_type())),
        _ => *dtype == DataType::TIMESTAMP || *dtype == DataType::TIMESTAMP_NTZ,
    }
}

fn suppress_struct(array: &StructArray, paths: &[&[String]]) -> KernelResult<StructArray> {
    let mut columns = array.columns().to_vec();
    for (field, column) in array.fields().iter().zip(&mut columns) {
        let paths = paths
            .iter()
            .filter_map(|path| match path {
                [name, rest @ ..] if name == field.name() => Some(rest),
                _ => None,
            })
            .collect::<Vec<_>>();
        if paths.iter().any(|path| path.is_empty()) {
            *column = new_null_array(column.data_type(), column.len());
        } else if !paths.is_empty()
            && let Some(child) = column.as_any().downcast_ref::<StructArray>()
        {
            *column = Arc::new(suppress_struct(child, &paths)?) as ArrayRef;
        }
    }
    Ok(StructArray::try_new_with_length(
        array.fields().clone(),
        columns,
        array.nulls().cloned(),
        array.len(),
    )?)
}
