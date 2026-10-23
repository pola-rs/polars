use polars_parquet_format::{BoundaryOrder, ColumnIndex, OffsetIndex, PageLocation};

use crate::parquet::error::ParquetResult;
use crate::parquet::write::page::{PageWriteSpec, is_data_page};

/// Serializes the column index of a column chunk.
///
/// Returns `None` if a page has no statistics, or if it has non-null values but no min/max. The
/// latter happens if all non-null values of the page are NaN or if min/max are turned off. The
/// column index could only mark such a page as a null page, which would be wrong.
pub fn serialize_column_index(pages: &[PageWriteSpec]) -> ParquetResult<Option<ColumnIndex>> {
    let mut null_pages = Vec::with_capacity(pages.len());
    let mut min_values = Vec::with_capacity(pages.len());
    let mut max_values = Vec::with_capacity(pages.len());
    let mut null_counts = Vec::with_capacity(pages.len());

    for spec in pages.iter().filter(|x| is_data_page(x)) {
        let Some(stats) = &spec.statistics else {
            return Ok(None);
        };
        let stats = stats.serialize();
        null_counts.push(stats.null_count);

        if spec.num_nulls == spec.num_values {
            // The bounds of a null page must be empty.
            min_values.push(vec![]);
            max_values.push(vec![]);
            null_pages.push(true);
        } else if let (Some(min_value), Some(max_value)) = (stats.min_value, stats.max_value) {
            min_values.push(min_value);
            max_values.push(max_value);
            null_pages.push(false);
        } else {
            return Ok(None);
        }
    }
    Ok(Some(ColumnIndex {
        null_pages,
        min_values,
        max_values,
        boundary_order: BoundaryOrder::UNORDERED,
        // The null counts are only known if the statistics have them.
        null_counts: null_counts.into_iter().collect(),
        repetition_level_histograms: None,
        definition_level_histograms: None,
    }))
}

pub fn serialize_offset_index(pages: &[PageWriteSpec]) -> ParquetResult<OffsetIndex> {
    let mut first_row_index = 0;
    let page_locations = pages
        .iter()
        .filter(|x| is_data_page(x))
        .map(|spec| {
            let location = PageLocation {
                offset: spec.offset.try_into()?,
                compressed_page_size: spec.bytes_written.try_into()?,
                first_row_index,
            };
            let num_rows = spec.num_rows;
            first_row_index += num_rows as i64;
            Ok(location)
        })
        .collect::<ParquetResult<Vec<_>>>()?;

    Ok(OffsetIndex {
        page_locations,
        unencoded_byte_array_data_bytes: None,
    })
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::{Array, Float64Array, Int128Array, ListArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::{ArrowDataType, Field};

    use super::*;
    use crate::arrow::write::{
        CompressionOptions, Encoding, StatisticsOptions, Version, WriteOptions, array_to_columns,
        to_parquet_type,
    };
    use crate::parquet::write::compress;
    use crate::parquet::write::page::write_page;

    fn page_specs(
        array: Box<dyn Array>,
        data_page_size: Option<usize>,
        statistics: StatisticsOptions,
    ) -> Vec<PageWriteSpec> {
        let field = Field::new("a".into(), array.dtype().clone(), true);
        let options = WriteOptions {
            statistics,
            version: Version::V1,
            compression: CompressionOptions::Uncompressed,
            data_page_size,
        };
        let mut columns = array_to_columns(
            array,
            to_parquet_type(&field).unwrap(),
            options,
            &[Encoding::Plain],
        )
        .unwrap();
        let mut buffer = vec![];
        columns
            .pop()
            .unwrap()
            .map(|page| {
                let page = compress(page.unwrap(), vec![], CompressionOptions::Uncompressed);
                write_page(&mut buffer, 0, &page.unwrap(), None).unwrap()
            })
            .collect()
    }

    #[test]
    fn null_pages_have_empty_bounds() {
        // One row per page: [1.0, null], null, [] and [null]. Only the first page has a value.
        let values = Float64Array::from([Some(1.0), None, None]).boxed();
        let array = ListArray::<i32>::new(
            ArrowDataType::List(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Float64,
                true,
            ))),
            vec![0, 2, 2, 2, 3].try_into().unwrap(),
            values,
            Some(Bitmap::from([true, false, true, true])),
        );

        let index = serialize_column_index(&page_specs(
            array.boxed(),
            Some(1),
            StatisticsOptions::full(),
        ))
        .unwrap()
        .unwrap();
        let one = 1.0f64.to_le_bytes().to_vec();
        assert_eq!(index.null_pages, [false, true, true, true]);
        assert_eq!(index.min_values, [one.clone(), vec![], vec![], vec![]]);
        assert_eq!(index.max_values, [one, vec![], vec![], vec![]]);
    }

    #[test]
    fn nan_is_left_out_of_bounds() {
        let array = Float64Array::from([Some(f64::NAN), Some(2.0), None, Some(1.0)]).boxed();
        let index = serialize_column_index(&page_specs(array, None, StatisticsOptions::full()))
            .unwrap()
            .unwrap();
        assert_eq!(index.null_pages, [false]);
        assert_eq!(index.min_values, [1.0f64.to_le_bytes().to_vec()]);
        assert_eq!(index.max_values, [2.0f64.to_le_bytes().to_vec()]);
    }

    #[test]
    fn null_counts_need_the_null_count_statistic() {
        let statistics = StatisticsOptions {
            null_count: false,
            ..StatisticsOptions::full()
        };
        let array = Float64Array::from([Some(1.0), None]).boxed();
        let index = serialize_column_index(&page_specs(array, Some(1), statistics))
            .unwrap()
            .unwrap();
        assert_eq!(index.null_counts, None);
    }

    #[test]
    fn pages_without_statistics_have_no_column_index() {
        // Int128 never writes min/max, so without null_count its pages have no statistics.
        let statistics = StatisticsOptions {
            null_count: false,
            ..StatisticsOptions::full()
        };
        let array = Int128Array::from([Some(1), None]).boxed();
        let index = serialize_column_index(&page_specs(array, None, statistics)).unwrap();
        assert!(index.is_none());
    }
}
