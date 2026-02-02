use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_buffer::Buffer;
use polars_core::prelude::CompatLevel;
use polars_error::PolarsResult;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::ParquetError;
use polars_parquet::write::{
    CompressedPage, Compressor, Encoding, SchemaDescriptor, WriteOptions, array_to_columns,
};
use polars_utils::UnitVec;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::opt_spawned_future::parallelize_first_to_local;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorsel;
use crate::nodes::io_sinks::writers::parquet::EncodedRowGroup;

pub struct RowGroupEncoder {
    pub morsel_rx: connector::Receiver<SinkMorsel>,
    pub encoded_row_group_tx:
        tokio::sync::mpsc::Sender<async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>>,
    /// Note: We assume it is checked in IR that this will match the schema of incoming morsels.
    pub arrow_schema: ArrowSchemaRef,
    pub schema_descriptor: Arc<SchemaDescriptor>,
    pub write_options: WriteOptions,
    pub encodings: Buffer<Vec<Encoding>>,
    pub num_leaf_columns: usize,
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
        } = self;

        while let Ok(morsel) = morsel_rx.recv().await {
            let arrow_schema = Arc::clone(&arrow_schema);
            let schema_descriptor = Arc::clone(&schema_descriptor);
            let encodings = Buffer::clone(&encodings);

            let row_group_encode_handle = async_executor::AbortOnDropHandle::new(
                async_executor::spawn(TaskPriority::High, async move {
                    let (df, morsel_permit) = morsel.into_inner();
                    let num_rows = df.height();

                    let mut data: Vec<Vec<CompressedPage>> = Vec::with_capacity(num_leaf_columns);

                    for fut in parallelize_first_to_local(
                        TaskPriority::High,
                        df.into_columns().into_iter().enumerate().map(|(i, c)| {
                            let arrow_schema = Arc::clone(&arrow_schema);
                            let schema_descriptor = Arc::clone(&schema_descriptor);
                            let encodings = Buffer::clone(&encodings);

                            async move {
                                let parquet_type = &schema_descriptor.fields()[i];
                                let encodings = encodings[i].as_slice();
                                let array =
                                    c.as_materialized_series().rechunk().to_arrow_with_field(
                                        0,
                                        CompatLevel::newest(),
                                        Some(arrow_schema.get_at_index(i).unwrap().1),
                                    )?;

                                let mut data: UnitVec<Vec<CompressedPage>> =
                                    UnitVec::with_capacity(num_leaf_columns);

                                for encode_page_iter in array_to_columns(
                                    array,
                                    parquet_type.clone(),
                                    write_options,
                                    encodings,
                                )? {
                                    let compressed_pages: Vec<CompressedPage> =
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
                                        .collect::<ParquetResult<_>>()?;

                                    data.push(compressed_pages)
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
                }),
            );

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
