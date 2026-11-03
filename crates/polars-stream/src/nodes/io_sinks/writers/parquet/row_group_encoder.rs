use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::array::{Array, DictionaryArray, PrimitiveArray};
use polars_arrow::datatypes::{ArrowSchemaRef, Field as ArrowField, IntegerType};
use polars_async::executor::{self, TaskMetricAggregator, TaskPriority};
use polars_async::primitives::connector;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_buffer::Buffer;
use polars_core::prelude::Column;
use polars_error::PolarsResult;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    CompressedPage, Compressor, Encoding, SchemaDescriptor, WriteOptions, array_to_columns,
};
use polars_utils::UnitVec;

use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::writers::parquet::EncodedRowGroup;

/// A dictionary-coded column as an arrow dictionary array, so that the writer writes its
/// dictionary as is rather than building one by hashing the values. Only the dictionary values
/// that rows use are kept. `None` if the column is not dictionary-coded or not written as a flat,
/// dictionary-encoded string column.
fn dictionary_array(
    c: &Column,
    field: &ArrowField,
    encodings: &[Encoding],
) -> Option<Box<dyn Array>> {
    use polars_core::prelude::*;

    let d = c.as_dictionary_column()?;
    // The values are written from views whatever the field's string type (by default
    // `LargeUtf8`); the file's schema still comes from the field.
    if !matches!(
        field.dtype,
        ArrowDataType::Utf8View | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8
    ) || encodings != [Encoding::RleDictionary]
    {
        return None;
    }

    let (used, mut keys) = d.used_codes(usize::MAX)?;
    if used.is_empty() && !keys.is_empty() {
        // Every row is null; there is no dictionary to write.
        return None;
    }

    // A dictionary may hold a value more than once (e.g. merged from two row groups' ones);
    // write each used value once.
    let dict_values = d.dict().values();
    let mut first_of: PlHashMap<&str, u32> = PlHashMap::with_capacity(used.len());
    let mut unique: Vec<IdxSize> = Vec::with_capacity(used.len());
    let remap: Vec<u32> = used
        .iter()
        .map(|c| {
            // SAFETY: the used codes are in bounds for the dictionary.
            let v = unsafe { dict_values.value_unchecked(*c as usize) };
            *first_of.entry(v).or_insert_with(|| {
                unique.push(*c);
                unique.len() as u32 - 1
            })
        })
        .collect();
    // Mostly distinct values are written PLAIN, as `encode_as_dictionary_optional` decides for an
    // uncoded column.
    if d.len() > 128 && (unique.len() as f64) / (d.len() as f64) > 0.75 {
        return None;
    }
    if unique.len() < used.len() {
        // Null rows' keys are 0, which stays in bounds.
        for k in keys.iter_mut() {
            *k = remap[*k as usize];
        }
    }

    let values = StringChunked::with_chunk(PlSmallStr::EMPTY, dict_values.clone());
    // SAFETY: the used codes are in bounds for the dictionary.
    let values = unsafe { values.take_unchecked(&IdxCa::from_vec(PlSmallStr::EMPTY, unique)) };
    let values = values.rechunk().downcast_as_array().clone();
    let validity = d.codes().rechunk_validity();
    let keys = PrimitiveArray::new(ArrowDataType::UInt32, keys.into(), validity);
    let dtype = ArrowDataType::Dictionary(
        IntegerType::UInt32,
        Box::new(ArrowDataType::Utf8View),
        false,
    );
    // SAFETY: every key indexes the used values (null keys are 0, and there is a value if any
    // key is valid).
    let array =
        unsafe { DictionaryArray::<u32>::try_new_unchecked(dtype, keys, values.boxed()) }.unwrap();
    Some(array.boxed())
}

pub struct RowGroupEncoder {
    pub morsel_rx: connector::Receiver<SinkMorsel>,
    pub encoded_row_group_tx:
        tokio::sync::mpsc::Sender<executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>>,
    /// Note: We assume it is checked in IR that this will match the schema of incoming morsels.
    pub arrow_schema: ArrowSchemaRef,
    pub schema_descriptor: Arc<SchemaDescriptor>,
    pub write_options: WriteOptions,
    pub encodings: Buffer<Vec<Encoding>>,
    pub num_leaf_columns: usize,
    pub task_metrics: Option<Arc<TaskMetricAggregator>>,
}

impl RowGroupEncoder {
    pub async fn run(self) -> PolarsResult<()> {
        let RowGroupEncoder {
            mut morsel_rx,
            encoded_row_group_tx,
            arrow_schema,
            schema_descriptor,
            write_options,
            encodings,
            num_leaf_columns,
            task_metrics,
        } = self;

        while let Ok(morsel) = morsel_rx.recv().await {
            let arrow_schema = Arc::clone(&arrow_schema);
            let schema_descriptor = Arc::clone(&schema_descriptor);
            let encodings = Buffer::clone(&encodings);
            let column_task_metrics = task_metrics.clone();

            let row_group_encode_handle = executor::AbortOnDropHandle::new(executor::spawn(
                TaskPriority::High,
                task_metrics.as_deref(),
                async move {
                    let (df, morsel_permit) = morsel.into_inner();
                    let num_rows = df.height();

                    let mut data: Vec<Vec<CompressedPage>> = Vec::with_capacity(num_leaf_columns);

                    for fut in parallelize_first_to_local(
                        TaskPriority::High,
                        column_task_metrics.as_deref(),
                        df.into_columns().into_iter().enumerate().map(|(i, c)| {
                            let arrow_schema = Arc::clone(&arrow_schema);
                            let schema_descriptor = Arc::clone(&schema_descriptor);
                            let encodings = Buffer::clone(&encodings);
                            let leaf_task_metrics = column_task_metrics.clone();

                            async move {
                                let parquet_type = &schema_descriptor.fields()[i];
                                let encodings = encodings[i].as_slice();
                                let field = arrow_schema.get_at_index(i).unwrap().1;
                                let array = match dictionary_array(&c, field, encodings) {
                                    Some(array) => array,
                                    None => c
                                        .as_materialized_series()
                                        .rechunk()
                                        .to_arrow_with_field(0, Cow::Borrowed(field), true)?,
                                };

                                let encode_page_iters = array_to_columns(
                                    array,
                                    parquet_type.clone(),
                                    write_options,
                                    encodings,
                                )?;

                                let mut data: UnitVec<Vec<CompressedPage>> =
                                    UnitVec::with_capacity(encode_page_iters.len());

                                // Nested columns can have many leaves, encode those in parallel.
                                for fut in parallelize_first_to_local(
                                    TaskPriority::High,
                                    leaf_task_metrics.as_deref(),
                                    encode_page_iters.into_iter().map(
                                        |encode_page_iter| async move {
                                            Compressor::new_from_vec(
                                                encode_page_iter.map(|result| {
                                                    result.map_err(|e| {
                                                        ParquetError::FeatureNotSupported(format!(
                                                            "reraised in polars: {e}",
                                                        ))
                                                    })
                                                }),
                                                write_options.compression,
                                                vec![],
                                            )
                                            .collect::<ParquetResult<Vec<CompressedPage>>>()
                                        },
                                    ),
                                ) {
                                    data.push(fut.await?);
                                }

                                PolarsResult::Ok(data)
                            }
                        }),
                    ) {
                        data.extend(fut.await?);
                    }

                    Ok(EncodedRowGroup {
                        num_rows,
                        data,
                        morsel_permit,
                    })
                },
            ));

            if encoded_row_group_tx
                .send(row_group_encode_handle)
                .await
                .is_err()
            {
                break;
            }
        }

        Ok(())
    }
}
