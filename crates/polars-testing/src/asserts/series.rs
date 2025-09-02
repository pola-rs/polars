/// Asserts that two series are equal according to the specified options.
///
/// This macro compares two Polars Series objects and panics with a detailed error message if they are not equal.
/// It provides two forms:
/// - With custom comparison options
/// - With default comparison options
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_testing::assert_series_equal;
/// use polars_testing::asserts::SeriesEqualOptions;
///
/// // Create two series to compare
/// let s1 = Series::new("a".into(), &[1, 2, 3]);
/// let s2 = Series::new("a".into(), &[1, 2, 3]);
///
/// // Assert with default options
/// assert_series_equal!(&s1, &s2);
///
/// // Assert with custom options
/// let options = SeriesEqualOptions::default()
///     .with_check_exact(true)
///     .with_check_dtypes(false);
/// assert_series_equal!(&s1, &s2, options);
/// ```
///
/// # Panics
///
/// Panics when the series are not equal according to the specified comparison criteria.
///
#[macro_export]
macro_rules! assert_series_equal {
    ($left:expr, $right:expr $(, $options:expr)?) => {
        {
            #[allow(unused_assignments)]
            #[allow(unused_mut)]
            let mut options = $crate::asserts::SeriesEqualOptions::default();
            $(options = $options;)?

            match $crate::asserts::assert_series_equal($left, $right, options) {
                Ok(_) => {},
                Err(e) => panic!("{}", e),
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    // Testing default struct implementation
    #[test]
    fn test_series_equal_options() {
        let options = crate::asserts::SeriesEqualOptions::default();

        assert!(options.check_dtypes);
        assert!(options.check_names);
        assert!(options.check_order);
        assert!(options.check_exact);
        assert_eq!(options.rel_tol, 1e-5);
        assert_eq!(options.abs_tol, 1e-8);
        assert!(!options.categorical_as_str);
    }

    // Testing with basic parameters
    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_series_length_mismatch() {
        let s1 = Series::new("".into(), &[1, 2]);
        let s2 = Series::new("".into(), &[1, 2, 3]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "name mismatch")]
    fn test_series_names_mismatch() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s2".into(), &[1, 2, 3]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_check_names_false() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s2".into(), &[1, 2, 3]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_names(false);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn test_series_dtype_mismatch() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &["1", "2", "3"]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_check_dtypes_false() {
        let s1 = Series::new("s1".into(), &[1, 2, 3]);
        let s2 = Series::new("s1".into(), &[1.0, 2.0, 3.0]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_dtypes(false);

        assert_series_equal!(&s1, &s2, options);
    }

    // Testing basic equality
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_value_mismatch_int() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[2, 3, 4]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_values_match_int() {
        let s1 = Series::new("".into(), &[1, 2, 3]);
        let s2 = Series::new("".into(), &[1, 2, 3]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_value_mismatch_str() {
        let s1 = Series::new("".into(), &["foo", "bar"]);
        let s2 = Series::new("".into(), &["moo", "car"]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_values_match_str() {
        let s1 = Series::new("".into(), &["foo", "bar"]);
        let s2 = Series::new("".into(), &["foo", "bar"]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_values_mismatch_float() {
        let s1 = Series::new("".into(), &[1.1, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[2.2, 3.3, 4.4]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_values_match_float() {
        let s1 = Series::new("".into(), &[1.1, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[1.1, 2.2, 3.3]);

        assert_series_equal!(&s1, &s2);
    }

    // Testing float value precision equality
    #[test]
    #[should_panic(expected = "values not within tolerance")]
    fn test_series_float_exceeded_tol() {
        let s1 = Series::new("".into(), &[1.0, 2.2, 3.3]);
        let s2 = Series::new("".into(), &[1.00012, 2.200025, 3.300035]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    fn test_series_float_within_tol() {
        let s1 = Series::new("".into(), &[1.0, 2.0, 3.0]);
        let s2 = Series::new("".into(), &[1.000005, 2.000015, 3.000025]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    fn test_series_float_exact_tolerance_boundary() {
        let s1 = Series::new("".into(), &[1.0, 2.0, 3.0]);
        let s2 = Series::new("".into(), &[1.0, 2.0 + 1e-5, 3.0]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    fn test_series_float_custom_rel_tol() {
        let s1 = Series::new("".into(), &[10.0, 100.0, 1000.0]);
        let s2 = Series::new("".into(), &[10.05, 100.1, 1000.2]);

        let options = crate::asserts::SeriesEqualOptions::default()
            .with_check_exact(false)
            .with_rel_tol(0.01);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "values not within tolerance")]
    fn test_series_float_custom_abs_tol() {
        let s1 = Series::new("".into(), &[0.001, 0.01, 0.1]);
        let s2 = Series::new("".into(), &[0.001, 0.02, 0.1]);

        let options = crate::asserts::SeriesEqualOptions::default()
            .with_check_exact(false)
            .with_abs_tol(0.005);

        assert_series_equal!(&s1, &s2, options);
    }

    // Testing equality with special values
    #[test]
    fn test_series_empty_equal() {
        let s1 = Series::default();
        let s2 = Series::default();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_nan_equal() {
        let s1 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);
        let s2 = Series::new("".into(), &[f64::NAN, f64::NAN, f64::NAN]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_null_equal() {
        let s1 = Series::new("".into(), &[None::<i32>, None::<i32>, None::<i32>]);
        let s2 = Series::new("".into(), &[None::<i32>, None::<i32>, None::<i32>]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_infinity_values_mismatch() {
        let s1 = Series::new("".into(), &[1.0, f64::INFINITY, 3.0]);
        let s2 = Series::new("".into(), &[1.0, f64::NEG_INFINITY, 3.0]);

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_infinity_values_match() {
        let s1 = Series::new("".into(), &[1.0, f64::INFINITY, f64::NEG_INFINITY]);
        let s2 = Series::new("".into(), &[1.0, f64::INFINITY, f64::NEG_INFINITY]);

        assert_series_equal!(&s1, &s2);
    }

    // Testing null and nan counts for floats
    #[test]
    #[should_panic(expected = "null value mismatch")]
    fn test_series_check_exact_false_null() {
        let s1 = Series::new("".into(), &[Some(1.0), None::<f64>, Some(3.0)]);
        let s2 = Series::new("".into(), &[Some(1.0), Some(2.0), Some(3.0)]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    #[should_panic(expected = "nan value mismatch")]
    fn test_series_check_exact_false_nan() {
        let s1 = Series::new("".into(), &[1.0, f64::NAN, 3.0]);
        let s2 = Series::new("".into(), &[1.0, 2.0, 3.0]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_exact(false);

        assert_series_equal!(&s1, &s2, options);
    }

    // Testing sorting operations
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_sorting_unequal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default();

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    fn test_series_sorting_equal() {
        let s1 = Series::new("".into(), &[Some(1), Some(2), Some(3), None::<i32>]);
        let s2 = Series::new("".into(), &[Some(2), None::<i32>, Some(3), Some(1)]);

        let options = crate::asserts::SeriesEqualOptions::default().with_check_order(false);

        assert_series_equal!(&s1, &s2, options);
    }

    // Testing categorical operations
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_categorical_mismatch() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let s2 = Series::new("".into(), &["apple", "orange", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_categorical_match() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let s2 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_categorical_str_mismatch() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let s2 = Series::new("".into(), &["apple", "orange", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();

        let options = crate::asserts::SeriesEqualOptions::default().with_categorical_as_str(true);

        assert_series_equal!(&s1, &s2, options);
    }

    #[test]
    fn test_series_categorical_str_match() {
        let s1 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let s2 = Series::new("".into(), &["apple", "banana", "cherry"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();

        let options = crate::asserts::SeriesEqualOptions::default().with_categorical_as_str(true);

        assert_series_equal!(&s1, &s2, options);
    }

    // Testing equality of nested values
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_list_values_int_mismatch() {
        let s1 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>(),
            ],
        );

        let s2 = Series::new(
            "".into(),
            &[
                [0, 2, 3].iter().collect::<Series>(),
                [4, 7, 6].iter().collect::<Series>(),
                [7, 8, 10].iter().collect::<Series>(),
            ],
        );

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_list_values_int_match() {
        let s1 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>(),
            ],
        );

        let s2 = Series::new(
            "".into(),
            &[
                [1, 2, 3].iter().collect::<Series>(),
                [4, 5, 6].iter().collect::<Series>(),
                [7, 8, 9].iter().collect::<Series>(),
            ],
        );

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "nested value mismatch")]
    fn test_series_list_values_float_mismatch() {
        let s1 = Series::new(
            "".into(),
            &[
                [1.1, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 5.0, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 9.0].iter().collect::<Series>(),
            ],
        );

        let s2 = Series::new(
            "".into(),
            &[
                [0.5, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 7.5, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 10.2].iter().collect::<Series>(),
            ],
        );

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_list_values_float_match() {
        let s1 = Series::new(
            "".into(),
            &[
                [1.1, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 5.0, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 9.0].iter().collect::<Series>(),
            ],
        );

        let s2 = Series::new(
            "".into(),
            &[
                [1.1, 2.0, 3.0].iter().collect::<Series>(),
                [4.0, 5.0, 6.0].iter().collect::<Series>(),
                [7.0, 8.0, 9.0].iter().collect::<Series>(),
            ],
        );

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_struct_values_str_mismatch() {
        let field1 = Series::new("field1".into(), &["a", "d", "g"]);
        let field2 = Series::new("field2".into(), &["b", "e", "h"]);

        let s1_fields = [field1.clone(), field2];
        let s1_struct =
            StructChunked::from_series("".into(), field1.len(), s1_fields.iter()).unwrap();
        let s1 = s1_struct.into_series();

        let field1_alt = Series::new("field1".into(), &["a", "DIFFERENT", "g"]);
        let field2_alt = Series::new("field2".into(), &["b", "e", "h"]);

        let s2_fields = [field1_alt.clone(), field2_alt];
        let s2_struct =
            StructChunked::from_series("".into(), field1_alt.len(), s2_fields.iter()).unwrap();
        let s2 = s2_struct.into_series();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_struct_values_str_match() {
        let field1 = Series::new("field1".into(), &["a", "d", "g"]);
        let field2 = Series::new("field2".into(), &["b", "e", "h"]);

        let s1_fields = [field1.clone(), field2.clone()];
        let s1_struct =
            StructChunked::from_series("".into(), field1.len(), s1_fields.iter()).unwrap();
        let s1 = s1_struct.into_series();

        let s2_fields = [field1.clone(), field2];
        let s2_struct =
            StructChunked::from_series("".into(), field1.len(), s2_fields.iter()).unwrap();
        let s2 = s2_struct.into_series();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_struct_values_mixed_mismatch() {
        let id = Series::new("id".into(), &[1, 2, 3]);
        let value = Series::new("value".into(), &["a", "b", "c"]);
        let active = Series::new("active".into(), &[true, false, true]);

        let s1_fields = [id.clone(), value.clone(), active.clone()];
        let s1_struct = StructChunked::from_series("".into(), id.len(), s1_fields.iter()).unwrap();
        let s1 = s1_struct.into_series();

        let id_alt = Series::new("id".into(), &[1, 99, 3]);
        let s2_fields = [id_alt, value, active];
        let s2_struct = StructChunked::from_series("".into(), id.len(), s2_fields.iter()).unwrap();
        let s2 = s2_struct.into_series();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_struct_values_mixed_match() {
        let id = Series::new("id".into(), &[1, 2, 3]);
        let value = Series::new("value".into(), &["a", "b", "c"]);
        let active = Series::new("active".into(), &[true, false, true]);

        let s1_fields = [id.clone(), value.clone(), active.clone()];
        let s1_struct = StructChunked::from_series("".into(), id.len(), s1_fields.iter()).unwrap();
        let s1 = s1_struct.into_series();

        let s2_fields = [id.clone(), value, active];
        let s2_struct = StructChunked::from_series("".into(), id.len(), s2_fields.iter()).unwrap();
        let s2 = s2_struct.into_series();

        assert_series_equal!(&s1, &s2);
    }

    // Testing equality of deeply nested values
    #[test]
    #[should_panic(expected = "nested value mismatch")]
    fn test_deeply_nested_list_float_mismatch() {
        let inner_list_1 = Series::new("inner".into(), &[1.0, 2.0]);
        let outer_list_1 = Series::new("outer".into(), &[inner_list_1]);
        let s1 = Series::new("nested".into(), &[outer_list_1]);

        let inner_list_2 = Series::new("inner".into(), &[1.0, 3.0]);
        let outer_list_2 = Series::new("outer".into(), &[inner_list_2]);
        let s2 = Series::new("nested".into(), &[outer_list_2]);

        assert_series_equal!(&s1, &s2);
    }
    #[test]
    fn test_deeply_nested_list_float_match() {
        let inner_list_1 = Series::new("".into(), &[1.0, 2.0]);
        let outer_list_1 = Series::new("".into(), &[inner_list_1]);

        let s1 = Series::new("".into(), &[outer_list_1]);

        let inner_list_2 = Series::new("".into(), &[1.0, 2.0]);
        let outer_list_2 = Series::new("".into(), &[inner_list_2]);
        let s2 = Series::new("".into(), &[outer_list_2]);

        assert_series_equal!(&s1, &s2);
    }

    // Testing equality of temporal types
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_datetime_values_mismatch() {
        let dt1: i64 = 1672567200000000000;
        let dt2: i64 = 1672653600000000000;
        let dt3: i64 = 1672657200000000000;

        let s1 = Series::new("".into(), &[dt1, dt2])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap();
        let s2 = Series::new("".into(), &[dt1, dt3])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_datetime_values_match() {
        let dt1: i64 = 1672567200000000000;
        let dt2: i64 = 1672653600000000000;

        let s1 = Series::new("".into(), &[dt1, dt2])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap();
        let s2 = Series::new("".into(), &[dt1, dt2])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    // Testing equality of decimal types
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_decimal_values_mismatch() {
        let s1 = Series::new("".into(), &[1, 2])
            .cast(&DataType::Decimal(Some(10), Some(2)))
            .unwrap();
        let s2 = Series::new("".into(), &[1, 3])
            .cast(&DataType::Decimal(Some(10), Some(2)))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_decimal_values_match() {
        let s1 = Series::new("".into(), &[1, 2])
            .cast(&DataType::Decimal(Some(10), Some(2)))
            .unwrap();
        let s2 = Series::new("".into(), &[1, 2])
            .cast(&DataType::Decimal(Some(10), Some(2)))
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    // Testing equality of binary types
    #[test]
    #[should_panic(expected = "exact value mismatch")]
    fn test_series_binary_values_mismatch() {
        let s1 = Series::new("".into(), &[vec![1u8, 2, 3], vec![4, 5, 6]])
            .cast(&DataType::Binary)
            .unwrap();
        let s2 = Series::new("".into(), &[vec![1u8, 2, 3], vec![4, 5, 7]])
            .cast(&DataType::Binary)
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }

    #[test]
    fn test_series_binary_values_match() {
        let s1 = Series::new("".into(), &[vec![1u8, 2, 3], vec![4, 5, 6]])
            .cast(&DataType::Binary)
            .unwrap();
        let s2 = Series::new("".into(), &[vec![1u8, 2, 3], vec![4, 5, 6]])
            .cast(&DataType::Binary)
            .unwrap();

        assert_series_equal!(&s1, &s2);
    }
}
