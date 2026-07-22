use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
use arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_error::{PolarsError, PolarsResult};

use super::{Filter, PredicateFilter, column_iter_to_arrays, column_iter_to_arrays_selected};
use crate::arrow::read::expr::{ParquetColumnExpr, SpecializedParquetColumnExpr};
use crate::parquet::read::{BasicDecompressor, PageReader, read_metadata};
use crate::write::{
    CompressionOptions, Encoding, FileWriter, RowGroupIterator, StatisticsOptions, Version,
    WriteOptions,
};

struct GreaterThan {
    threshold: i32,
    null_matches: bool,
    evaluated_rows: Option<Arc<AtomicUsize>>,
}

impl ParquetColumnExpr for GreaterThan {
    fn evaluate_mut(&self, values: &dyn Array, mask: &mut BitmapBuilder) {
        let values = values
            .as_any()
            .downcast_ref::<PrimitiveArray<i32>>()
            .unwrap();
        if let Some(evaluated_rows) = self.evaluated_rows.as_ref() {
            evaluated_rows.fetch_add(values.len(), Ordering::Relaxed);
        }
        mask.reserve(values.len());
        for value in values.values().iter() {
            mask.push(*value > self.threshold);
        }
    }

    fn evaluate_null(&self) -> bool {
        self.null_matches
    }

    fn as_specialized(&self) -> Option<&SpecializedParquetColumnExpr> {
        None
    }
}

fn write_primitive(values: PrimitiveArray<i32>) -> PolarsResult<Vec<u8>> {
    let field = Field::new("value".into(), ArrowDataType::Int32, true);
    let schema = ArrowSchema::from_iter([field]);
    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_page_size: Some(256),
    };
    let batch = RecordBatchT::try_new(
        values.len(),
        Arc::new(schema.clone()),
        vec![Box::new(values) as Box<dyn Array>],
    )?;
    let row_groups = RowGroupIterator::try_new(
        std::iter::once(Ok::<_, PolarsError>(batch)),
        &schema,
        options,
        Buffer::from_iter([vec![Encoding::Plain]]),
    )?;

    let mut writer = FileWriter::try_new(Cursor::new(Vec::new()), schema, options)?;
    for row_group in row_groups {
        writer.write(u64::MAX, row_group?)?;
    }
    writer.end(None)?;
    Ok(writer.into_inner().into_inner())
}

fn decode_primitive(
    data: Vec<u8>,
    predicate: PredicateFilter,
    input_selection: Option<Bitmap>,
) -> PolarsResult<(Vec<Option<i32>>, Bitmap)> {
    let mut reader = Cursor::new(&data);
    let metadata = read_metadata(&mut reader)?;
    let column = &metadata.row_groups[0].parquet_columns()[0];
    let byte_range = column.byte_range();
    let column_data =
        Buffer::from_vec(data[byte_range.start as usize..byte_range.end as usize].to_vec());
    let max_page_size = column_data.len() * 2 + 1024;
    let pages = PageReader::new(Cursor::new(column_data), column, vec![], max_page_size);
    let decompressor = BasicDecompressor::new(pages, vec![]);
    let field = Field::new("value".into(), ArrowDataType::Int32, true);
    let columns = vec![decompressor];
    let types = vec![&column.descriptor().descriptor.primitive_type];
    let (arrays, mask) = match input_selection {
        Some(input_selection) => {
            column_iter_to_arrays_selected(columns, types, field, predicate, input_selection)?
        },
        None => column_iter_to_arrays(columns, types, field, Some(Filter::Predicate(predicate)))?,
    };
    let values = arrays
        .iter()
        .flat_map(|array| {
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap()
                .iter()
                .map(|value| value.copied())
        })
        .collect();
    Ok((values, mask))
}

fn input_values(len: usize) -> PrimitiveArray<i32> {
    PrimitiveArray::from_iter((0..len).map(|index| (index % 11 != 0).then_some(index as i32)))
}

#[test]
fn selected_predicate_refines_mask_in_column_coordinates() -> PolarsResult<()> {
    let len = 1_024;
    let input_selection = Bitmap::from_iter((0..len).map(|index| index % 3 == 0));
    let evaluated_rows = Arc::new(AtomicUsize::new(0));
    let predicate = PredicateFilter {
        predicate: Arc::new(GreaterThan {
            threshold: 500,
            null_matches: false,
            evaluated_rows: Some(evaluated_rows.clone()),
        }),
        include_values: true,
    };

    let (values, mask) = decode_primitive(
        write_primitive(input_values(len))?,
        predicate,
        Some(input_selection.clone()),
    )?;
    let expected_mask = Bitmap::from_iter(
        (0..len).map(|index| input_selection.get_bit(index) && index % 11 != 0 && index > 500),
    );
    let expected_values = (0..len)
        .filter(|&index| expected_mask.get_bit(index))
        .map(|index| Some(index as i32))
        .collect::<Vec<_>>();

    assert_eq!(mask, expected_mask);
    assert_eq!(values, expected_values);
    assert_eq!(
        evaluated_rows.load(Ordering::Relaxed),
        input_selection.set_bits(),
    );
    Ok(())
}

#[test]
fn selected_predicate_respects_matching_nulls() -> PolarsResult<()> {
    let len = 1_024;
    let input_selection = Bitmap::from_iter((0..len).map(|index| index % 5 != 0));
    let predicate = PredicateFilter {
        predicate: Arc::new(GreaterThan {
            threshold: 900,
            null_matches: true,
            evaluated_rows: None,
        }),
        include_values: true,
    };

    let (values, mask) = decode_primitive(
        write_primitive(input_values(len))?,
        predicate,
        Some(input_selection.clone()),
    )?;
    let expected_mask = Bitmap::from_iter(
        (0..len).map(|index| input_selection.get_bit(index) && (index % 11 == 0 || index > 900)),
    );
    let expected_values = (0..len)
        .filter(|&index| expected_mask.get_bit(index))
        .map(|index| (index % 11 != 0).then_some(index as i32))
        .collect::<Vec<_>>();

    assert_eq!(mask, expected_mask);
    assert_eq!(values, expected_values);
    Ok(())
}

#[test]
fn selected_predicate_can_return_only_the_mask() -> PolarsResult<()> {
    let len = 1_024;
    let input_selection = Bitmap::from_iter((0..len).map(|index| index % 7 == 0));
    let predicate = PredicateFilter {
        predicate: Arc::new(GreaterThan {
            threshold: 700,
            null_matches: false,
            evaluated_rows: None,
        }),
        include_values: false,
    };

    let (values, mask) = decode_primitive(
        write_primitive(input_values(len))?,
        predicate,
        Some(input_selection.clone()),
    )?;
    let expected_mask = Bitmap::from_iter(
        (0..len).map(|index| input_selection.get_bit(index) && index % 11 != 0 && index > 700),
    );

    assert_eq!(mask, expected_mask);
    assert!(values.is_empty());
    Ok(())
}

#[test]
fn selected_predicate_handles_an_empty_selection() -> PolarsResult<()> {
    let len = 1_024;
    let predicate = PredicateFilter {
        predicate: Arc::new(GreaterThan {
            threshold: 0,
            null_matches: true,
            evaluated_rows: None,
        }),
        include_values: true,
    };

    let (values, mask) = decode_primitive(
        write_primitive(input_values(len))?,
        predicate,
        Some(Bitmap::new_zeroed(len)),
    )?;

    assert_eq!(mask, Bitmap::new_zeroed(len));
    assert!(values.is_empty());
    Ok(())
}

#[test]
fn selected_predicate_dense_selection_matches_unselected_decode() -> PolarsResult<()> {
    let len = 1_024;
    let data = write_primitive(input_values(len))?;
    let predicate = || GreaterThan {
        threshold: 500,
        null_matches: true,
        evaluated_rows: None,
    };
    let (expected_values, expected_mask) = decode_primitive(
        data.clone(),
        PredicateFilter {
            predicate: Arc::new(predicate()),
            include_values: true,
        },
        None,
    )?;
    let (values, mask) = decode_primitive(
        data,
        PredicateFilter {
            predicate: Arc::new(predicate()),
            include_values: true,
        },
        Some(Bitmap::new_with_value(true, len)),
    )?;

    assert_eq!(mask, expected_mask);
    assert_eq!(values, expected_values);
    Ok(())
}

#[test]
fn selected_predicate_rejects_wrong_selection_length() -> PolarsResult<()> {
    let len = 1_024;
    let predicate = PredicateFilter {
        predicate: Arc::new(GreaterThan {
            threshold: 500,
            null_matches: false,
            evaluated_rows: None,
        }),
        include_values: true,
    };

    let error = decode_primitive(
        write_primitive(input_values(len))?,
        predicate,
        Some(Bitmap::new_with_value(true, len - 1)),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("input selection has length 1023")
    );
    Ok(())
}
