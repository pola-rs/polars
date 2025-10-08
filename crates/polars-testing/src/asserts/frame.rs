/// Asserts that two DataFrames are equal according to the specified options.
///
/// This macro compares two Polars DataFrame objects and panics with a detailed error message if they are not equal.
/// It provides two forms:
/// - With custom comparison options
/// - With default comparison options
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_testing::assert_dataframe_equal;
/// use polars_testing::asserts::DataFrameEqualOptions;
///
/// // Create two DataFrames to compare
/// let df1 = df! {
///     "a" => [1, 2, 3],
///     "b" => [4.0, 5.0, 6.0],
/// }.unwrap();
/// let df2 = df! {
///     "a" => [1, 2, 3],
///     "b" => [4.0, 5.0, 6.0],
/// }.unwrap();
///
/// // Assert with default options
/// assert_dataframe_equal!(&df1, &df2);
///
/// // Assert with custom options
/// let options = DataFrameEqualOptions::default()
///     .with_check_exact(true)
///     .with_check_row_order(false);
/// assert_dataframe_equal!(&df1, &df2, options);
/// ```
///
/// # Panics
///
/// Panics when the DataFrames are not equal according to the specified comparison criteria.
///
#[macro_export]
macro_rules! assert_dataframe_equal {
    ($left:expr, $right:expr $(, $options:expr)?) => {
        #[allow(unused_assignments)]
        #[allow(unused_mut)]
        let mut options = $crate::asserts::DataFrameEqualOptions::default();
        $(options = $options;)?

        match $crate::asserts::assert_dataframe_equal($left, $right, options) {
            Ok(_) => {},
            Err(e) => panic!("{}", e),
        }
    };
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use polars_core::prelude::*;

    // Testing default struct implementation
    #[test]
    fn test_dataframe_equal_options() {
        let options = crate::asserts::DataFrameEqualOptions::default();

        assert!(options.check_row_order);
        assert!(options.check_column_order);
        assert!(options.check_dtypes);
        assert!(!options.check_exact);
        assert_eq!(options.rel_tol, 1e-5);
        assert_eq!(options.abs_tol, 1e-8);
        assert!(!options.categorical_as_str);
    }

    // Testing dataframe schema equality parameters
    #[test]
    #[should_panic(expected = "height (row count) mismatch")]
    fn test_dataframe_height_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2]).into(),
            Series::new("col2".into(), &["a", "b"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "columns mismatch")]
    fn test_dataframe_column_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("different_col".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "dtypes do not match")]
    fn test_dataframe_dtype_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, 2.0, 3.0]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_dtype_mismatch_ignored() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, 2.0, 3.0]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let options = crate::asserts::DataFrameEqualOptions::default().with_check_dtypes(false);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    #[test]
    #[should_panic(expected = "columns are not in the same order")]
    fn test_dataframe_column_order_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col1".into(), &[1, 2, 3]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_column_order_mismatch_ignored() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col1".into(), &[1, 2, 3]).into(),
        ])
        .unwrap();

        let options =
            crate::asserts::DataFrameEqualOptions::default().with_check_column_order(false);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    #[test]
    #[should_panic(expected = "columns mismatch: [\"col3\"] in left, but not in right")]
    fn test_dataframe_left_has_extra_column() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "columns mismatch: [\"col3\"] in right, but not in left")]
    fn test_dataframe_right_has_extra_column() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    // Testing basic equality
    #[test]
    #[should_panic(expected = "value mismatch for column")]
    fn test_dataframe_value_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "changed"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_equal() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
            Series::new("col3".into(), &[true, false, true]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_row_order_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[3, 1, 2]).into(),
            Series::new("col2".into(), &["c", "a", "b"]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_row_order_ignored() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1, 2, 3]).into(),
            Series::new("col2".into(), &["a", "b", "c"]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[3, 1, 2]).into(),
            Series::new("col2".into(), &["c", "a", "b"]).into(),
        ])
        .unwrap();

        let options = crate::asserts::DataFrameEqualOptions::default().with_check_row_order(false);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    // Testing more comprehensive equality
    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_complex_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("integers".into(), &[1, 2, 3, 4, 5]).into(),
            Series::new("floats".into(), &[1.1, 2.2, 3.3, 4.4, 5.5]).into(),
            Series::new("strings".into(), &["a", "b", "c", "d", "e"]).into(),
            Series::new("booleans".into(), &[true, false, true, false, true]).into(),
            Series::new("opt_ints".into(), &[Some(1), None, Some(3), Some(4), None]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("integers".into(), &[1, 2, 99, 4, 5]).into(),
            Series::new("floats".into(), &[1.1, 2.2, 3.3, 9.9, 5.5]).into(),
            Series::new("strings".into(), &["a", "b", "c", "CHANGED", "e"]).into(),
            Series::new("booleans".into(), &[true, false, false, false, true]).into(),
            Series::new("opt_ints".into(), &[Some(1), None, Some(3), None, None]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_complex_match() {
        let df1 = DataFrame::new(vec![
            Series::new("integers".into(), &[1, 2, 3, 4, 5]).into(),
            Series::new("floats".into(), &[1.1, 2.2, 3.3, 4.4, 5.5]).into(),
            Series::new("strings".into(), &["a", "b", "c", "d", "e"]).into(),
            Series::new("booleans".into(), &[true, false, true, false, true]).into(),
            Series::new("opt_ints".into(), &[Some(1), None, Some(3), Some(4), None]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("integers".into(), &[1, 2, 3, 4, 5]).into(),
            Series::new("floats".into(), &[1.1, 2.2, 3.3, 4.4, 5.5]).into(),
            Series::new("strings".into(), &["a", "b", "c", "d", "e"]).into(),
            Series::new("booleans".into(), &[true, false, true, false, true]).into(),
            Series::new("opt_ints".into(), &[Some(1), None, Some(3), Some(4), None]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    // Testing float value precision equality
    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_numeric_exact_fail() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0000001, 2.0000002, 3.0000003]).into(),
        ])
        .unwrap();

        let df2 =
            DataFrame::new(vec![Series::new("col1".into(), &[1.0, 2.0, 3.0]).into()]).unwrap();

        let options = crate::asserts::DataFrameEqualOptions::default().with_check_exact(true);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    #[test]
    fn test_dataframe_numeric_tolerance_pass() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0000001, 2.0000002, 3.0000003]).into(),
        ])
        .unwrap();

        let df2 =
            DataFrame::new(vec![Series::new("col1".into(), &[1.0, 2.0, 3.0]).into()]).unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    // Testing equality with special values
    #[test]
    fn test_empty_dataframe_equal() {
        let df1 = DataFrame::default();
        let df2 = DataFrame::default();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_empty_dataframe_schema_equal() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &Vec::<i32>::new()).into(),
            Series::new("col2".into(), &Vec::<String>::new()).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &Vec::<i32>::new()).into(),
            Series::new("col2".into(), &Vec::<String>::new()).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_single_row_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[42]).into(),
            Series::new("col2".into(), &["value"]).into(),
            Series::new("col3".into(), &[true]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[42]).into(),
            Series::new("col2".into(), &["different"]).into(),
            Series::new("col3".into(), &[true]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_single_row_match() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[42]).into(),
            Series::new("col2".into(), &["value"]).into(),
            Series::new("col3".into(), &[true]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[42]).into(),
            Series::new("col2".into(), &["value"]).into(),
            Series::new("col3".into(), &[true]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_null_values_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[Some(1), None, Some(3)]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[Some(1), Some(2), None]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_null_values_match() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[Some(1), None, Some(3)]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[Some(1), None, Some(3)]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_nan_values_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::NAN, 3.0]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, 2.0, f64::NAN]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_nan_values_match() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::NAN, 3.0]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::NAN, 3.0]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_infinity_values_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::INFINITY, 3.0]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::NEG_INFINITY, 3.0]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_infinity_values_match() {
        let df1 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::INFINITY, 3.0]).into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new("col1".into(), &[1.0, f64::INFINITY, 3.0]).into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    // Testing categorical operations
    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_categorical_as_string_mismatch() {
        let mut categorical1 = Series::new("categories".into(), &["a", "b", "c", "d"]);
        categorical1 = categorical1
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let df1 = DataFrame::new(vec![categorical1.into()]).unwrap();

        let mut categorical2 = Series::new("categories".into(), &["a", "b", "c", "e"]);
        categorical2 = categorical2
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let df2 = DataFrame::new(vec![categorical2.into()]).unwrap();

        let options =
            crate::asserts::DataFrameEqualOptions::default().with_categorical_as_str(true);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    #[test]
    fn test_dataframe_categorical_as_string_match() {
        let mut categorical1 = Series::new("categories".into(), &["a", "b", "c", "d"]);
        categorical1 = categorical1
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let df1 = DataFrame::new(vec![categorical1.into()]).unwrap();

        let mut categorical2 = Series::new("categories".into(), &["a", "b", "c", "d"]);
        categorical2 = categorical2
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let df2 = DataFrame::new(vec![categorical2.into()]).unwrap();

        let options =
            crate::asserts::DataFrameEqualOptions::default().with_categorical_as_str(true);
        assert_dataframe_equal!(&df1, &df2, options);
    }

    // Testing nested types
    #[test]
    #[should_panic(expected = "value mismatch")]
    fn test_dataframe_nested_values_mismatch() {
        let df1 = DataFrame::new(vec![
            Series::new(
                "list_col".into(),
                &[
                    Some(vec![1, 2, 3]),
                    Some(vec![4, 5, 6]),
                    None,
                    Some(vec![7, 8, 9]),
                ],
            )
            .into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new(
                "list_col".into(),
                &[
                    Some(vec![1, 2, 3]),
                    Some(vec![4, 5, 99]),
                    None,
                    Some(vec![7, 8, 9]),
                ],
            )
            .into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }

    #[test]
    fn test_dataframe_nested_values_match() {
        let df1 = DataFrame::new(vec![
            Series::new(
                "list_col".into(),
                &[Some(vec![1, 2, 3]), Some(vec![]), None, Some(vec![7, 8, 9])],
            )
            .into(),
        ])
        .unwrap();

        let df2 = DataFrame::new(vec![
            Series::new(
                "list_col".into(),
                &[Some(vec![1, 2, 3]), Some(vec![]), None, Some(vec![7, 8, 9])],
            )
            .into(),
        ])
        .unwrap();

        assert_dataframe_equal!(&df1, &df2);
    }
}
