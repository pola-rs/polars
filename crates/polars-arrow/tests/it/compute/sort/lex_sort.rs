use polars_arrow::array::*;
use polars_arrow::compute::sort::{lexsort, SortColumn, SortOptions};

fn test_lex_sort_arrays(input: Vec<SortColumn>, expected: Vec<Box<dyn Array>>) {
    let sorted = lexsort::<i32>(&input, None).unwrap();
    assert_eq!(sorted, expected);

    let sorted = lexsort::<i32>(&input, Some(4)).unwrap();
    let expected = expected
        .into_iter()
        .map(|x| x.sliced(0, 4))
        .collect::<Vec<_>>();
    assert_eq!(sorted, expected);

    let sorted = lexsort::<i32>(&input, Some(2)).unwrap();
    let expected = expected
        .into_iter()
        .map(|x| x.sliced(0, 2))
        .collect::<Vec<_>>();
    assert_eq!(sorted, expected);
}

#[test]
fn test_lex_sort_mixed_types() {
    let c1 = Int64Array::from(&[Some(0), Some(2), Some(-1), Some(0)]);
    let c2 = UInt32Array::from(&[Some(101), Some(8), Some(7), Some(102)]);
    let c3 = Int64Array::from(&[Some(-1), Some(-2), Some(-3), Some(-4)]);

    let input = vec![
        SortColumn {
            values: &c1,
            options: None,
        },
        SortColumn {
            values: &c2,
            options: None,
        },
        SortColumn {
            values: &c3,
            options: None,
        },
    ];
    let c1 = Int64Array::from([Some(-1), Some(0), Some(0), Some(2)]);
    let c2 = UInt32Array::from([Some(7), Some(101), Some(102), Some(8)]);
    let c3 = Int64Array::from([Some(-3), Some(-1), Some(-4), Some(-2)]);
    let expected = vec![c1.boxed(), c2.boxed(), c3.boxed()];
    test_lex_sort_arrays(input, expected);
}

#[test]
fn test_lex_sort_mixed_types2() {
    // test mix of string and in64 with option
    let c1 = Int64Array::from([Some(0), Some(2), Some(-1), Some(0)]);
    let c2 = Utf8Array::<i32>::from([Some("foo"), Some("9"), Some("7"), Some("bar")]);
    let input = vec![
        SortColumn {
            values: &c1,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: &c2,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    let expected = vec![
        Int64Array::from([Some(2), Some(0), Some(0), Some(-1)]).boxed(),
        Utf8Array::<i32>::from([Some("9"), Some("foo"), Some("bar"), Some("7")]).boxed(),
    ];
    test_lex_sort_arrays(input, expected);
}

/*
    // test sort with nulls first
    let input = vec![
        SortColumn {
            values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                None,
                Some(-1),
                Some(2),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
        SortColumn {
            values: Arc::new(StringArray::from(vec![
                Some("foo"),
                Some("world"),
                Some("hello"),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    let expected = vec![
        Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            None,
            None,
            Some(2),
            Some(-1),
        ])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            None,
            Some("foo"),
            Some("hello"),
            Some("world"),
        ])) as ArrayRef,
    ];
    test_lex_sort_arrays(input, expected);

    // test sort with nulls last
    let input = vec![
        SortColumn {
            values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                None,
                Some(-1),
                Some(2),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
        },
        SortColumn {
            values: Arc::new(StringArray::from(vec![
                Some("foo"),
                Some("world"),
                Some("hello"),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
        },
    ];
    let expected = vec![
        Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            Some(2),
            Some(-1),
            None,
            None,
        ])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            Some("hello"),
            Some("world"),
            Some("foo"),
            None,
        ])) as ArrayRef,
    ];
    test_lex_sort_arrays(input, expected);

    // test sort with opposite options
    let input = vec![
        SortColumn {
            values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                None,
                Some(-1),
                Some(2),
                Some(-1),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        },
        SortColumn {
            values: Arc::new(StringArray::from(vec![
                Some("foo"),
                Some("bar"),
                Some("world"),
                Some("hello"),
                None,
            ])) as ArrayRef,
            options: Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
        },
    ];
    let expected = vec![
        Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            Some(-1),
            Some(-1),
            Some(2),
            None,
            None,
        ])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            Some("hello"),
            Some("bar"),
            Some("world"),
            None,
            Some("foo"),
        ])) as ArrayRef,
    ];
    test_lex_sort_arrays(input, expected);
}
*/
