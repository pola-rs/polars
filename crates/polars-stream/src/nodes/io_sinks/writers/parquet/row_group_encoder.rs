use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::array::{Array, DictionaryArray, PrimitiveArray};
use polars_arrow::datatypes::{ArrowSchemaRef, Field as ArrowField, IntegerType};
use polars_async::executor::{self, TaskMetricAggregator, TaskPriority};
use polars_async::primitives::connector;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_buffer::Buffer;
use polars_core::frame::column::DictionaryColumn;
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

/// Estimate distinct valid codes without compacting them or hashing string values.
fn estimate_dictionary_code_cardinality(d: &DictionaryColumn) -> usize {
    let mut cardinality = polars_compute::hyperloglogplus::HyperLogLog::<u32>::new();
    // Repeated codes need only update HLL once. Collisions merely evict this cache entry.
    let mut recent_codes = [0_u64; 256];
    let mut add = |code: &u32| {
        let tag = u64::from(*code) + 1;
        let entry = &mut recent_codes[*code as usize & 255];
        if *entry != tag {
            *entry = tag;
            cardinality.add(code);
        }
    };
    for codes in d.codes().downcast_iter() {
        if codes.has_nulls() {
            for code in codes.iter().flatten() {
                add(code);
            }
        } else {
            for code in codes.values_iter() {
                add(code);
            }
        }
    }
    cardinality.count()
}

/// A bounded sample can veto early rejection when different codes contain equal strings.
fn dictionary_sample_has_unique_mapping(d: &DictionaryColumn) -> bool {
    use polars_core::prelude::{InitHashMaps, PlHashMap};

    const SAMPLE_SIZE: usize = 2_048;
    if d.is_empty() {
        return false;
    }
    // SplitMix64 with a fixed seed spreads global positions across clustered code layouts.
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let mut positions = Vec::with_capacity(SAMPLE_SIZE);
    for _ in 0..SAMPLE_SIZE {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        positions.push(((z ^ (z >> 31)) % d.len() as u64) as usize);
    }
    positions.sort_unstable();
    let values = d.dict().values();
    let mut first_of: PlHashMap<&str, u32> = PlHashMap::with_capacity(SAMPLE_SIZE);
    let mut positions = positions.into_iter().peekable();
    let mut offset = 0;
    for codes in d.codes().downcast_iter() {
        let end = offset + codes.len();
        while let Some(&position) = positions.peek().filter(|&&p| p < end) {
            positions.next();
            let index = position - offset;
            // Check validity before reading a code: null payloads may be out of bounds.
            if codes.is_valid(index) {
                let code = codes.values()[index];
                // SAFETY: valid codes are in bounds for the dictionary.
                let value = unsafe { values.value_unchecked(code as usize) };
                if let Some(first) = first_of.insert(value, code) {
                    if first != code {
                        return false;
                    }
                }
            }
        }
        offset = end;
    }
    !first_of.is_empty()
}

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

    // Output row groups can use almost as many codes as rows while retaining a much larger
    // source dictionary. Avoid compacting their codes and building a string lookup table only
    // to discard the result below. A bounded string sample vetoes early rejection if it finds
    // duplicate mappings; it cannot guarantee finding every duplicate. The ordinary writer
    // still makes the final encoding decision.
    const MIN_CARDINALITY_PREFLIGHT_ROWS: usize = 16_384;
    let max_dictionary_values = d.len() - d.len().div_ceil(4);
    if d.len() >= MIN_CARDINALITY_PREFLIGHT_ROWS
        && d.dict().len() > max_dictionary_values
        && d.len() - d.null_count() > max_dictionary_values
        && estimate_dictionary_code_cardinality(d) > max_dictionary_values
        && dictionary_sample_has_unique_mapping(d)
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

#[cfg(test)]
mod tests {
    use polars_core::frame::column::{DictionaryColumn, SharedDictionary};
    use polars_core::prelude::*;

    use super::*;

    fn encode_dictionary(values: Vec<String>, codes: UInt32Chunked) -> Option<Box<dyn Array>> {
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        let column =
            DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap();
        dictionary_array(
            &Column::Dictionary(column),
            &ArrowField::new("s".into(), ArrowDataType::LargeUtf8, true),
            &[Encoding::RleDictionary],
        )
    }

    #[test]
    fn dictionary_sink_preserves_sparse_multichunk_codes_and_nulls() {
        let values = (0..40_000).map(|i| format!("value-{i}")).collect();
        let mut codes = UInt32Chunked::from_iter_options(
            PlSmallStr::EMPTY,
            (0..10_000).map(|i| if i % 97 == 0 { None } else { Some(i % 2) }),
        );
        codes.append(&codes.clone()).unwrap();
        let null_count = codes.null_count();
        let array = encode_dictionary(values, codes).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        assert_eq!(array.len(), 20_000);
        assert_eq!(array.values().len(), 2);
        assert_eq!(array.null_count(), null_count);
        // Code zero is a real value, even when null rows also hold zero keys.
        assert!(array.keys().values().contains(&0));
    }

    #[test]
    fn dictionary_sink_preserves_mostly_null_rows() {
        let values = (0..40_000).map(|i| format!("value-{i}")).collect();
        let codes = UInt32Chunked::from_iter_options(
            PlSmallStr::EMPTY,
            (0..20_000).map(|i| (i % 2 == 0).then_some(0)),
        );
        let array = encode_dictionary(values, codes).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        assert_eq!(array.values().len(), 1);
        assert_eq!(array.null_count(), 10_000);
    }

    #[test]
    fn dictionary_sink_keeps_small_row_group_boundary() {
        for len in [128, 129] {
            let values = (0..len).map(|i| format!("value-{i}")).collect();
            let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..len).collect());
            assert_eq!(encode_dictionary(values, codes).is_some(), len == 128);
        }
    }

    #[test]
    fn dictionary_sink_preserves_duplicate_values_despite_distinct_codes() {
        for len in [16_383, 16_384, 32_768] {
            // Code NDV is high, but the sample finds equal strings under different codes.
            let values = (0..len).map(|i| format!("value-{}", i % 8)).collect();
            let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..len).collect());
            assert!(encode_dictionary(values, codes).is_some());
        }
    }

    #[test]
    fn dictionary_sink_preserves_moderately_duplicated_values() {
        let values = (0..32_768).map(|i| format!("value-{}", i / 8)).collect();
        let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..32_768).collect());
        let array = encode_dictionary(values, codes).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        assert_eq!(array.values().len(), 4_096);
    }

    #[test]
    fn dictionary_sink_preflight_rejects_distinct_values() {
        for len in [16_384, 32_768] {
            let values = (0..len).map(|i| format!("value-{i}")).collect();
            let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..len).collect());
            assert!(encode_dictionary(values, codes).is_none());
        }
    }

    #[test]
    fn dictionary_sink_unique_cardinality_supports_early_rejection_across_chunks() {
        let values: Vec<_> = (0..32_768).map(|i| format!("value-{i}")).collect();
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        let mut codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..7_001).collect());
        codes
            .append(&UInt32Chunked::from_vec(
                PlSmallStr::EMPTY,
                (7_001..32_768).collect(),
            ))
            .unwrap();
        let column =
            DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap();
        assert!(estimate_dictionary_code_cardinality(&column) > 24_576);
    }

    #[test]
    fn dictionary_sink_cardinality_spans_chunks_and_ignores_null_codes() {
        let values = (0..32_768).map(|i| format!("value-{}", i % 8)).collect();
        let mut codes = UInt32Chunked::from_iter_options(
            PlSmallStr::EMPTY,
            (0..16_384).map(|i| (i % 11 != 0).then_some(i)),
        );
        codes
            .append(&UInt32Chunked::from_iter_options(
                PlSmallStr::EMPTY,
                (16_384..32_768).map(|i| (i % 11 != 0).then_some(i)),
            ))
            .unwrap();
        let null_count = codes.null_count();
        let array = encode_dictionary(values, codes).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        assert_eq!(array.null_count(), null_count);
        assert_eq!(array.values().len(), 8);
    }

    #[test]
    fn dictionary_sink_cached_cardinality_matches_uncached() {
        for pattern in [vec![0, 1, 0, 1], vec![0, 256, 512], (0..32_768).collect()] {
            for with_nulls in [false, true] {
                let mut codes = UInt32Chunked::from_iter_options(
                    PlSmallStr::EMPTY,
                    (0..65_536).map(|i| {
                        if with_nulls && i % 11 == 0 {
                            None
                        } else {
                            Some(pattern[i % pattern.len()])
                        }
                    }),
                );
                codes.append(&codes.clone()).unwrap();
                let mut uncached = polars_compute::hyperloglogplus::HyperLogLog::<u32>::new();
                for chunk in codes.downcast_iter() {
                    for code in chunk.iter().flatten() {
                        uncached.add(code);
                    }
                }
                let values: Vec<_> = (0..32_768).map(|i| format!("value-{i}")).collect();
                let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
                let column =
                    DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes)
                        .unwrap();
                assert_eq!(
                    estimate_dictionary_code_cardinality(&column),
                    uncached.count()
                );
            }
        }
    }

    #[test]
    fn dictionary_sink_preserves_medium_used_domain_of_large_dictionary() {
        let values = (0..262_144).map(|i| format!("value-{i}")).collect();
        let codes = UInt32Chunked::from_vec(
            PlSmallStr::EMPTY,
            (0..122_880).map(|i| i % 20_011).collect(),
        );
        let array = encode_dictionary(values, codes).unwrap();
        let array = array
            .as_any()
            .downcast_ref::<DictionaryArray<u32>>()
            .unwrap();
        assert_eq!(array.values().len(), 20_011);
    }

    #[test]
    fn dictionary_sink_cardinality_does_not_read_out_of_bounds_null_codes() {
        let len = 32_768;
        let values = (0..len).map(|i| format!("value-{}", i % 8)).collect();
        let codes = PrimitiveArray::new(
            ArrowDataType::UInt32,
            (0..len)
                .map(|i| if i % 64 == 0 { u32::MAX } else { i })
                .collect(),
            Some((0..len).map(|i| i % 64 != 0).collect()),
        );
        let codes = UInt32Chunked::with_chunk(PlSmallStr::EMPTY, codes);
        let array = encode_dictionary(values, codes).unwrap();
        assert_eq!(array.null_count(), 512);
    }

    #[test]
    fn dictionary_sink_sample_distinguishes_codes_from_values() {
        for layout in 0..4 {
            let values: Vec<_> = (0..32_768)
                .map(|i| {
                    format!(
                        "value-{}",
                        match layout {
                            0 => i,
                            1 => i % 8,
                            2 => i / 8,
                            _ => 0,
                        }
                    )
                })
                .collect();
            let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
            let mut codes = UInt32Chunked::from_vec(
                PlSmallStr::EMPTY,
                (0..7_001)
                    .map(|i| if layout == 3 { 0 } else { i })
                    .collect(),
            );
            codes
                .append(&UInt32Chunked::from_vec(
                    PlSmallStr::EMPTY,
                    (7_001..32_768)
                        .map(|i| if layout == 3 { 0 } else { i })
                        .collect(),
                ))
                .unwrap();
            let column =
                DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes)
                    .unwrap();
            assert_eq!(
                dictionary_sample_has_unique_mapping(&column),
                layout == 0 || layout == 3
            );
        }
    }

    #[test]
    fn dictionary_sink_sample_vetoes_empty_valid_sample() {
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(["value"]);
        let codes = UInt32Chunked::full_null(PlSmallStr::EMPTY, 32_768);
        let column =
            DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap();
        assert!(!dictionary_sample_has_unique_mapping(&column));
    }

    #[test]
    fn dictionary_sink_all_null_rows_have_no_dictionary() {
        let values = (0..40_000).map(|i| format!("value-{i}")).collect();
        let codes = UInt32Chunked::full_null(PlSmallStr::EMPTY, 20_000);
        assert!(encode_dictionary(values, codes).is_none());
    }
}
