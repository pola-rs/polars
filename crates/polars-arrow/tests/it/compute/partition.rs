use polars_arrow::array::*;
use polars_arrow::compute::partition::*;
use polars_arrow::compute::sort::{SortColumn, SortOptions};
use polars_arrow::datatypes::DataType;
use polars_arrow::error::Result;

#[test]
fn lexicographical_partition_ranges_empty() {
    let input = vec![];
    assert!(
        lexicographical_partition_ranges(&input).is_err(),
        "lexicographical_partition_ranges should reject columns with empty rows"
    );
}

#[test]
fn lexicographical_partition_ranges_unaligned_rows() {
    let values1 = Int64Array::from([None, Some(-1)]);
    let values2 = Utf8Array::<i32>::from([Some("foo")]);
    let input = vec![
        SortColumn {
            values: &values1,
            options: None,
        },
        SortColumn {
            values: &values2,
            options: None,
        },
    ];
    assert!(
        lexicographical_partition_ranges(&input).is_err(),
        "lexicographical_partition_ranges should reject columns with different row counts"
    );
}

#[test]
fn lexicographical_partition_single_column() -> Result<()> {
    let values = Int64Array::from_slice([1, 2, 2, 2, 2, 2, 2, 2, 9]);
    let input = vec![SortColumn {
        values: &values,
        options: Some(SortOptions {
            descending: false,
            nulls_first: true,
        }),
    }];
    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(
            vec![(0_usize..1_usize), (1_usize..8_usize), (8_usize..9_usize)],
            results.collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn lexicographical_partition_all_equal_values() -> Result<()> {
    let values = Int64Array::from_trusted_len_values_iter(std::iter::repeat(1).take(1000));
    let input = vec![SortColumn {
        values: &values,
        options: Some(SortOptions {
            descending: false,
            nulls_first: true,
        }),
    }];

    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(vec![(0_usize..1000_usize)], results.collect::<Vec<_>>());
    }
    Ok(())
}

#[test]
fn lexicographical_partition_all_null_values() -> Result<()> {
    let values1 = new_null_array(DataType::Int8, 1000);
    let values2 = new_null_array(DataType::UInt16, 1000);
    let input = vec![
        SortColumn {
            values: values1.as_ref(),
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: values2.as_ref(),
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        },
    ];
    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(vec![(0_usize..1000_usize)], results.collect::<Vec<_>>());
    }
    Ok(())
}

#[test]
fn lexicographical_partition_unique_column_1() -> Result<()> {
    let values1 = Int64Array::from(vec![None, Some(-1)]);
    let values2 = Utf8Array::<i32>::from(vec![Some("foo"), Some("bar")]);
    let input = vec![
        SortColumn {
            values: &values1,
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: &values2,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(
            vec![(0_usize..1_usize), (1_usize..2_usize)],
            results.collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn lexicographical_partition_unique_column_2() -> Result<()> {
    let values1 = Int64Array::from(vec![None, Some(-1), Some(-1)]);
    let values2 = Utf8Array::<i32>::from(vec![Some("foo"), Some("bar"), Some("apple")]);

    let input = vec![
        SortColumn {
            values: &values1,
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: &values2,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(
            vec![(0_usize..1_usize), (1_usize..2_usize), (2_usize..3_usize),],
            results.collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn lexicographical_partition_non_unique_column_1() -> Result<()> {
    let values1 = Int64Array::from(vec![None, Some(-1), Some(-1), Some(1)]);
    let values2 = Utf8Array::<i32>::from(vec![Some("foo"), Some("bar"), Some("bar"), Some("bar")]);

    let input = vec![
        SortColumn {
            values: &values1,
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: &values2,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    {
        let results = lexicographical_partition_ranges(&input)?;
        assert_eq!(
            vec![(0_usize..1_usize), (1_usize..3_usize), (3_usize..4_usize),],
            results.collect::<Vec<_>>()
        );
    }
    Ok(())
}
