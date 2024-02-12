mod lex_sort;
mod row;

use polars_arrow::array::*;
use polars_arrow::compute::sort::*;
use polars_arrow::datatypes::*;
use polars_arrow::types::NativeType;

fn to_indices_boolean_arrays(data: &[Option<bool>], options: SortOptions, expected_data: &[i32]) {
    let output = BooleanArray::from(data);
    let expected = Int32Array::from_slice(expected_data);
    let output = sort_to_indices(&output, &options, None).unwrap();
    assert_eq!(output, expected)
}

fn primitive_arrays<T>(
    data: &[Option<T>],
    data_type: DataType,
    options: SortOptions,
    expected_data: &[Option<T>],
) where
    T: NativeType,
{
    let input = PrimitiveArray::<T>::from(data).to(data_type.clone());
    let expected = PrimitiveArray::<T>::from(expected_data).to(data_type);
    let output = sort(&input, &options, None).unwrap();
    assert_eq!(expected, output.as_ref())
}

fn to_indices_string_arrays(data: &[Option<&str>], options: SortOptions, expected_data: &[i32]) {
    let input = Utf8Array::<i32>::from(data);
    let expected = Int32Array::from_slice(expected_data);
    let output = sort_to_indices(&input, &options, None).unwrap();
    assert_eq!(output, expected)
}

fn string_arrays(data: &[Option<&str>], options: SortOptions, expected_data: &[Option<&str>]) {
    let input = Utf8Array::<i32>::from(data);
    let expected = Utf8Array::<i32>::from(expected_data);
    let output = sort(&input, &options, None).unwrap();
    assert_eq!(expected, output.as_ref())
}

fn string_dict_arrays(data: &[Option<&str>], options: SortOptions, expected_data: &[Option<&str>]) {
    let mut input = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();
    input.try_extend(data.iter().copied()).unwrap();
    let input = input.into_arc();

    let mut expected = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();
    expected.try_extend(expected_data.iter().copied()).unwrap();
    let expected = expected.into_arc();

    let output = sort(input.as_ref(), &options, None).unwrap();
    assert_eq!(expected.as_ref(), output.as_ref())
}

/*
fn list_arrays<T>(
    data: Vec<Option<Vec<Option<T::Native>>>>,
    options: Option<SortOptions>,
    expected_data: Vec<Option<Vec<Option<T::Native>>>>,
    fixed_length: Option<i32>,
) where
    T: ArrowPrimitiveType,
    PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
{
    // for FixedSizedList
    if let Some(length) = fixed_length {
        let input = Arc::new(build_fixed_size_list_nullable(data.clone(), length));
        let sorted = sort(&(input as ArrayRef), options).unwrap();
        let expected = Arc::new(build_fixed_size_list_nullable(
            expected_data.clone(),
            length,
        )) as ArrayRef;

        assert_eq!(&sorted, &expected);
    }

    // for List
    let input = Arc::new(build_generic_list_nullable::<i32, T>(data.clone()));
    let sorted = sort(&(input as ArrayRef), options).unwrap();
    let expected =
        Arc::new(build_generic_list_nullable::<i32, T>(expected_data.clone()))
            as ArrayRef;

    assert_eq!(&sorted, &expected);

    // for LargeList
    let input = Arc::new(build_generic_list_nullable::<i64, T>(data));
    let sorted = sort(&(input as ArrayRef), options).unwrap();
    let expected =
        Arc::new(build_generic_list_nullable::<i64, T>(expected_data)) as ArrayRef;

    assert_eq!(&sorted, &expected);
}

fn test_lex_sort_arrays(input: Vec<SortColumn>, expected_output: Vec<ArrayRef>) {
    let sorted = lexsort(&input).unwrap();

    for (result, expected) in sorted.iter().zip(expected_output.iter()) {
        assert_eq!(result, expected);
    }
}
*/

#[test]
fn boolean() {
    // boolean
    to_indices_boolean_arrays(
        &[None, Some(false), Some(true), Some(true), Some(false), None],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[0, 5, 1, 4, 2, 3],
    );

    // boolean, descending
    to_indices_boolean_arrays(
        &[None, Some(false), Some(true), Some(true), Some(false), None],
        SortOptions {
            descending: true,
            nulls_first: false,
        },
        &[2, 3, 1, 4, 5, 0],
    );

    // boolean, descending, nulls first
    to_indices_boolean_arrays(
        &[None, Some(false), Some(true), Some(true), Some(false), None],
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        &[5, 0, 2, 3, 1, 4],
    );
}

#[test]
#[ignore] // improve equality for NaN values. These are right but the equality fails
fn test_nans() {
    primitive_arrays::<f64>(
        &[None, Some(0.0), Some(2.0), Some(-1.0), Some(f64::NAN), None],
        DataType::Float64,
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        &[None, None, Some(f64::NAN), Some(2.0), Some(0.0), Some(-1.0)],
    );
    primitive_arrays::<f64>(
        &[Some(f64::NAN), Some(f64::NAN), Some(f64::NAN), Some(1.0)],
        DataType::Float64,
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        &[Some(f64::NAN), Some(f64::NAN), Some(f64::NAN), Some(1.0)],
    );

    primitive_arrays::<f64>(
        &[None, Some(0.0), Some(2.0), Some(-1.0), Some(f64::NAN), None],
        DataType::Float64,
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[None, None, Some(-1.0), Some(0.0), Some(2.0), Some(f64::NAN)],
    );
    // nans
    primitive_arrays::<f64>(
        &[Some(f64::NAN), Some(f64::NAN), Some(f64::NAN), Some(1.0)],
        DataType::Float64,
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[Some(1.0), Some(f64::NAN), Some(f64::NAN), Some(f64::NAN)],
    );
}

#[test]
fn to_indices_strings() {
    to_indices_string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        // &[3, 0, 5, 1, 4, 2] is also valid
        &[0, 3, 5, 1, 4, 2],
    );

    to_indices_string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: false,
        },
        // &[2, 4, 1, 5, 3, 0] is also valid
        &[2, 4, 1, 5, 0, 3],
    );

    to_indices_string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        // &[3, 0, 5, 1, 4, 2] is also valid
        &[0, 3, 5, 1, 4, 2],
    );

    to_indices_string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        // &[3, 0, 2, 4, 1, 5] is also valid
        &[0, 3, 2, 4, 1, 5],
    );
}

#[test]
fn strings() {
    string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("-ad"),
            Some("bad"),
            Some("glad"),
            Some("sad"),
        ],
    );

    string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: false,
        },
        &[
            Some("sad"),
            Some("glad"),
            Some("bad"),
            Some("-ad"),
            None,
            None,
        ],
    );

    string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("-ad"),
            Some("bad"),
            Some("glad"),
            Some("sad"),
        ],
    );

    string_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("sad"),
            Some("glad"),
            Some("bad"),
            Some("-ad"),
        ],
    );
}

#[test]
fn string_dicts() {
    string_dict_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("-ad"),
            Some("bad"),
            Some("glad"),
            Some("sad"),
        ],
    );

    string_dict_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: false,
        },
        &[
            Some("sad"),
            Some("glad"),
            Some("bad"),
            Some("-ad"),
            None,
            None,
        ],
    );

    string_dict_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("-ad"),
            Some("bad"),
            Some("glad"),
            Some("sad"),
        ],
    );

    string_dict_arrays(
        &[
            None,
            Some("bad"),
            Some("sad"),
            None,
            Some("glad"),
            Some("-ad"),
        ],
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        &[
            None,
            None,
            Some("sad"),
            Some("glad"),
            Some("bad"),
            Some("-ad"),
        ],
    );
}

/*
#[test]
fn list() {
    list_arrays::<i8>(
        vec![
            Some(vec![Some(1)]),
            Some(vec![Some(4)]),
            Some(vec![Some(2)]),
            Some(vec![Some(3)]),
        ],
        Some(SortOptions {
            descending: false,
            nulls_first: false,
        }),
        vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2)]),
            Some(vec![Some(3)]),
            Some(vec![Some(4)]),
        ],
        Some(1),
    );

    list_arrays::<i32>(
        vec![
            Some(vec![Some(1), Some(0)]),
            Some(vec![Some(4), Some(3), Some(2), Some(1)]),
            Some(vec![Some(2), Some(3), Some(4)]),
            Some(vec![Some(3), Some(3), Some(3), Some(3)]),
            Some(vec![Some(1), Some(1)]),
        ],
        Some(SortOptions {
            descending: false,
            nulls_first: false,
        }),
        vec![
            Some(vec![Some(1), Some(0)]),
            Some(vec![Some(1), Some(1)]),
            Some(vec![Some(2), Some(3), Some(4)]),
            Some(vec![Some(3), Some(3), Some(3), Some(3)]),
            Some(vec![Some(4), Some(3), Some(2), Some(1)]),
        ],
        None,
    );

    list_arrays::<i32>(
        vec![
            None,
            Some(vec![Some(4), None, Some(2)]),
            Some(vec![Some(2), Some(3), Some(4)]),
            None,
            Some(vec![Some(3), Some(3), None]),
        ],
        Some(SortOptions {
            descending: false,
            nulls_first: false,
        }),
        vec![
            Some(vec![Some(2), Some(3), Some(4)]),
            Some(vec![Some(3), Some(3), None]),
            Some(vec![Some(4), None, Some(2)]),
            None,
            None,
        ],
        Some(3),
    );
}

#[test]
fn test_lex_sort_single_column() {
    let input = vec![SortColumn {
        values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            Some(17),
            Some(2),
            Some(-1),
            Some(0),
        ])) as ArrayRef,
        options: None,
    }];
    let expected = vec![Arc::new(PrimitiveArray::<Int64Type>::from(vec![
        Some(-1),
        Some(0),
        Some(2),
        Some(17),
    ])) as ArrayRef];
    test_lex_sort_arrays(input, expected);
}

#[test]
fn test_lex_sort_unaligned_rows() {
    let input = vec![
        SortColumn {
            values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![None, Some(-1)]))
                as ArrayRef,
            options: None,
        },
        SortColumn {
            values: Arc::new(StringArray::from(vec![Some("foo")])) as ArrayRef,
            options: None,
        },
    ];
    assert!(
        lexsort(&input).is_err(),
        "lexsort should reject columns with different row counts"
    );
}
*/

#[test]
fn consistency() {
    use polars_arrow::array::new_null_array;
    use polars_arrow::datatypes::DataType::*;
    use polars_arrow::datatypes::TimeUnit;

    let datatypes = vec![
        Null,
        Boolean,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
    ];

    datatypes.into_iter().for_each(|d1| {
        let array = new_null_array(d1.clone(), 10);
        let options = SortOptions {
            descending: true,
            nulls_first: true,
        };
        if can_sort(&d1) {
            assert!(sort(array.as_ref(), &options, None).is_ok());
        } else {
            assert!(sort(array.as_ref(), &options, None).is_err());
        }
    });
}
