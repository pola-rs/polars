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

/// The valid codes at the sorted, distinct `positions`, in position order.
fn sampled_codes(d: &DictionaryColumn, positions: &[usize]) -> Vec<u32> {
    let mut codes_out = Vec::with_capacity(positions.len());
    let mut positions = positions.iter().copied().peekable();
    let mut offset = 0;
    for codes in d.codes().downcast_iter() {
        let end = offset + codes.len();
        while let Some(position) = positions.next_if(|&p| p < end) {
            let index = position - offset;
            // Check validity before reading a code: null payloads may be out of bounds.
            if codes.is_valid(index) {
                codes_out.push(codes.values()[index]);
            }
        }
        offset = end;
    }
    codes_out
}

/// A bounded sample of rows can veto early rejection when different codes contain equal strings
/// (or when it holds no valid row). A sample cannot find every duplicate, so a miss only costs the
/// exact path's work, which the ordinary writer's final check still decides on. Samples rows at a
/// fixed seed, so decisions are deterministic per build, though not across `rand` versions or
/// pointer widths.
fn dictionary_sample_has_unique_mapping(d: &DictionaryColumn) -> bool {
    use polars_core::prelude::{InitHashMaps, PlHashMap};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    let n = d.len();
    if n == 0 {
        return false;
    }
    // A string under two codes is found only if both rows are drawn: with p such pairs per row,
    // about p * s^2 / n pairs are. 4 * sqrt(n) rows give about 4 at p = 1/4, which is the density
    // when the codes are nearly all distinct. Near the boundary (codes C minus unique strings U
    // below about n / 16) p is lower and the veto may be missed: that costs speed and dictionary
    // reuse, never values.
    let sample_size = (4.0 * (n as f64).sqrt()).ceil().max(2_048.0) as usize;
    let sample_size = sample_size.min(n);
    // Draws with replacement cost O(s), unlike `index::sample`, which may allocate n entries.
    // Repeated positions are dropped, so the sample can be up to about s^2 / (2 * n) rows short.
    let mut rng = SmallRng::seed_from_u64(0x243f_6a88_85a3_08d3);
    let mut positions: Vec<usize> = (0..sample_size).map(|_| rng.random_range(0..n)).collect();
    positions.sort_unstable();
    positions.dedup();
    let codes = sampled_codes(d, &positions);
    if codes.is_empty() {
        return false;
    }

    let values = d.dict().values();
    let mut first_of: PlHashMap<&str, u32> = PlHashMap::with_capacity(codes.len());
    for code in codes {
        // SAFETY: valid codes are in bounds for the dictionary.
        let value = unsafe { values.value_unchecked(code as usize) };
        if let Some(first) = first_of.insert(value, code) {
            if first != code {
                return false;
            }
        }
    }
    true
}

/// Whether the group is certainly or likely mostly distinct values: the rows hold almost as many
/// distinct codes as `max_dictionary_values` and the sample found no equal strings under
/// different codes. The estimate comes first, so the sample (random reads of the dictionary and
/// string hashing) only runs for groups whose codes look nearly distinct.
fn rejects_dictionary(d: &DictionaryColumn, max_dictionary_values: usize) -> bool {
    estimate_dictionary_code_cardinality(d) > max_dictionary_values
        && dictionary_sample_has_unique_mapping(d)
}

/// Whether `dictionary_array` skips preparing the group: it is large, its dictionary and its
/// valid rows both exceed the most dictionary values that are worth writing, and
/// `rejects_dictionary` finds its codes nearly distinct with no duplicate mapping in the sample.
/// A bounded sample of rows vetoes the rejection if it finds duplicate mappings; it cannot
/// guarantee finding every duplicate.
fn preflight_rejects(d: &DictionaryColumn) -> bool {
    const MIN_CARDINALITY_PREFLIGHT_ROWS: usize = 16_384;
    let max_dictionary_values = d.len() - d.len().div_ceil(4);
    d.len() >= MIN_CARDINALITY_PREFLIGHT_ROWS
        && d.dict().len() > max_dictionary_values
        && d.len() - d.null_count() > max_dictionary_values
        && rejects_dictionary(d, max_dictionary_values)
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
    // to discard the result below. The ordinary writer still makes the final encoding decision.
    if preflight_rejects(d) {
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
            // Layout 3 has one hot code, so the sample finds no duplicate string.
            assert_eq!(
                dictionary_sample_has_unique_mapping(&column),
                layout == 0 || layout == 3
            );
            assert_eq!(rejects_dictionary(&column, 24_576), layout == 0);
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
    fn dictionary_sink_sample_finds_sparse_duplicate_pairs_in_large_groups() {
        // Every code is used once and every second code holds the string of the one before it,
        // so half a pair per row. A 2,048-row sample draws both rows of a pair about twice in
        // 1M rows; scaled to the row count it does about eight times.
        let n: u32 = 1 << 20;
        let values: Vec<_> = (0..n)
            .map(|i| format!("value-{}", if i % 2 == 1 { i - 1 } else { i }))
            .collect();
        let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..n).collect());
        let values_array = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        let column = DictionaryColumn::try_new(
            "s".into(),
            SharedDictionary::new(values_array),
            codes.clone(),
        )
        .unwrap();
        assert!(!dictionary_sample_has_unique_mapping(&column));
        assert!(!rejects_dictionary(
            &column,
            n as usize - (n as usize).div_ceil(4)
        ));
        assert!(encode_dictionary(values, codes).is_some());
    }

    #[test]
    fn dictionary_sink_rejects_skewed_near_distinct_codes() {
        // More than 0.75 * n distinct codes, with the surplus rows in codes of a few copies or
        // of about n / s copies: the estimate and the sample still reject early.
        let n: u32 = 1 << 19;
        let max = n as usize - (n as usize).div_ceil(4);
        let values: Vec<_> = (0..n).map(|i| format!("value-{i}")).collect();
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        for (copies, heavy) in [(20_u32, 5_243_u32), (181, 580)] {
            // `heavy` codes of `copies` rows, the rest distinct: C is about 0.8 * n.
            let heavy_rows = copies * heavy;
            let codes: Vec<u32> = (0..n)
                .map(|i| {
                    if i < heavy_rows {
                        i / copies
                    } else {
                        i - heavy_rows + heavy
                    }
                })
                .collect();
            let distinct = codes
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert!(distinct > max);
            let codes = UInt32Chunked::from_vec(PlSmallStr::EMPTY, codes);
            let column =
                DictionaryColumn::try_new("s".into(), SharedDictionary::new(values.clone()), codes)
                    .unwrap();
            assert!(
                dictionary_sample_has_unique_mapping(&column),
                "{copies} copies"
            );
            assert!(rejects_dictionary(&column, max), "{copies} copies");
        }
    }

    #[test]
    fn dictionary_sink_keeps_hot_codes_of_a_large_dictionary() {
        // Two hot codes of a large dictionary, as in a filtered sink: the estimate keeps them
        // without a sample.
        let values: Vec<_> = (0..262_144).map(|i| format!("value-{i}")).collect();
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        let codes =
            UInt32Chunked::from_vec(PlSmallStr::EMPTY, (0..16_384).map(|i| i % 2).collect());
        let column =
            DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap();
        assert!(!rejects_dictionary(&column, 12_288));
    }

    fn near_distinct_column(
        n: u32,
        dict_len: u32,
        code: impl Fn(u32) -> Option<u32>,
        string: impl Fn(u32) -> u32,
    ) -> DictionaryColumn {
        let values: Vec<_> = (0..dict_len)
            .map(|i| format!("value-{}", string(i)))
            .collect();
        let values = polars_arrow::array::Utf8ViewArray::from_slice_values(&values);
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, (0..n).map(code));
        DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap()
    }

    #[test]
    fn dictionary_sink_preflight_predicate_row_boundary() {
        for (n, expected) in [(16_384, true), (16_383, false)] {
            let d = near_distinct_column(n, 32_768, Some, |i| i);
            assert_eq!(preflight_rejects(&d), expected, "{n} rows");
        }
    }

    #[test]
    fn dictionary_sink_preflight_predicate_keeps_few_codes_or_small_dictionary() {
        let n = 16_384;
        // Large dictionary, two hot codes.
        let d = near_distinct_column(n, 262_144, |i| Some(i % 2), |i| i);
        assert!(!preflight_rejects(&d));
        // Dictionary no larger than the most worth writing: 12,288 for 16,384 rows.
        let d = near_distinct_column(n, 12_288, |i| Some(i % 12_288), |i| i);
        assert!(!preflight_rejects(&d));
    }

    #[test]
    fn dictionary_sink_preflight_predicate_keeps_mostly_null_groups() {
        // 12,288 valid rows, all distinct codes, in a large dictionary: not above the maximum.
        let d = near_distinct_column(16_384, 32_768, |i| (i % 4 != 0).then_some(i), |i| i);
        assert!(!preflight_rejects(&d));
        // Many more nulls.
        let d = near_distinct_column(16_384, 32_768, |i| (i % 2 == 0).then_some(i), |i| i);
        assert!(!preflight_rejects(&d));
    }

    #[test]
    fn dictionary_sink_preflight_predicate_is_vetoed_by_duplicate_mapping() {
        // Distinct codes everywhere, but eight distinct strings.
        let d = near_distinct_column(32_768, 32_768, Some, |i| i % 8);
        assert!(!preflight_rejects(&d));
        // The same layout without duplicates qualifies.
        let d = near_distinct_column(32_768, 32_768, Some, |i| i);
        assert!(preflight_rejects(&d));
    }

    #[test]
    fn dictionary_sink_all_null_rows_have_no_dictionary() {
        let values = (0..40_000).map(|i| format!("value-{i}")).collect();
        let codes = UInt32Chunked::full_null(PlSmallStr::EMPTY, 20_000);
        assert!(encode_dictionary(values, codes).is_none());
    }
}
