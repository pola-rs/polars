use std::ops::Not;

use polars_core::datatypes::unpack_dtypes;
use polars_core::prelude::*;
use polars_ops::series::is_close;

/// Configuration options for comparing Series equality.
///
/// Controls the behavior of Series equality comparisons by specifying
/// which aspects to check and the tolerance for floating point comparisons.
pub struct SeriesEqualOptions {
    /// Whether to check that the data types match.
    pub check_dtypes: bool,
    /// Whether to check that the Series names match.
    pub check_names: bool,
    /// Whether to check that elements appear in the same order.
    pub check_order: bool,
    /// Whether to check for exact equality (true) or approximate equality (false) for floating point values.
    pub check_exact: bool,
    /// Relative tolerance for approximate equality of floating point values.
    pub rel_tol: f64,
    /// Absolute tolerance for approximate equality of floating point values.
    pub abs_tol: f64,
    /// Whether to compare categorical values as strings.
    pub categorical_as_str: bool,
}

impl Default for SeriesEqualOptions {
    /// Creates a new `SeriesEqualOptions` with default settings.
    ///
    /// Default configuration:
    /// - Checks data types, names, and order
    /// - Uses exact equality comparisons
    /// - Sets relative tolerance to 1e-5 and absolute tolerance to 1e-8 for floating point comparisons
    /// - Does not convert categorical values to strings for comparison
    fn default() -> Self {
        Self {
            check_dtypes: true,
            check_names: true,
            check_order: true,
            check_exact: true,
            rel_tol: 1e-5,
            abs_tol: 1e-8,
            categorical_as_str: false,
        }
    }
}

impl SeriesEqualOptions {
    /// Creates a new `SeriesEqualOptions` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to check that data types match.
    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    /// Sets whether to check that Series names match.
    pub fn with_check_names(mut self, value: bool) -> Self {
        self.check_names = value;
        self
    }

    /// Sets whether to check that elements appear in the same order.
    pub fn with_check_order(mut self, value: bool) -> Self {
        self.check_order = value;
        self
    }

    /// Sets whether to check for exact equality (true) or approximate equality (false) for floating point values.
    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    /// Sets the relative tolerance for approximate equality of floating point values.
    pub fn with_rel_tol(mut self, value: f64) -> Self {
        self.rel_tol = value;
        self
    }

    /// Sets the absolute tolerance for approximate equality of floating point values.
    pub fn with_abs_tol(mut self, value: f64) -> Self {
        self.abs_tol = value;
        self
    }

    /// Sets whether to compare categorical values as strings.
    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

/// Change a (possibly nested) Categorical data type to a String data type.
fn categorical_dtype_to_string_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Categorical(..) => DataType::String,
        DataType::List(inner) => {
            let inner_cast = categorical_dtype_to_string_dtype(inner);
            DataType::List(Box::new(inner_cast))
        },
        DataType::Array(inner, size) => {
            let inner_cast = categorical_dtype_to_string_dtype(inner);
            DataType::Array(Box::new(inner_cast), *size)
        },
        DataType::Struct(fields) => {
            let transformed_fields = fields
                .iter()
                .map(|field| {
                    Field::new(
                        field.name().clone(),
                        categorical_dtype_to_string_dtype(field.dtype()),
                    )
                })
                .collect::<Vec<Field>>();

            DataType::Struct(transformed_fields)
        },
        _ => dtype.clone(),
    }
}

/// Cast a (possibly nested) Categorical Series to a String Series.
fn categorical_series_to_string(s: &Series) -> PolarsResult<Series> {
    let dtype = s.dtype();
    let noncat_dtype = categorical_dtype_to_string_dtype(dtype);

    if *dtype != noncat_dtype {
        Ok(s.cast(&noncat_dtype)?)
    } else {
        Ok(s.clone())
    }
}

/// Returns true if both DataTypes are floating point types.
fn are_both_floats(left: &DataType, right: &DataType) -> bool {
    left.is_float() && right.is_float()
}

/// Returns true if both DataTypes are list-like (either List or Array types).
fn are_both_lists(left: &DataType, right: &DataType) -> bool {
    matches!(left, DataType::List(_) | DataType::Array(_, _))
        && matches!(right, DataType::List(_) | DataType::Array(_, _))
}

/// Returns true if both DataTypes are struct types.
fn are_both_structs(left: &DataType, right: &DataType) -> bool {
    left.is_struct() && right.is_struct()
}

/// Returns true if both DataTypes are nested types (lists or structs) that contain floating point types within them.
/// First checks if both types are either lists or structs, then unpacks their nested DataTypes to determine if
/// at least one floating point type exists in each of the nested structures.
fn comparing_nested_floats(left: &DataType, right: &DataType) -> bool {
    if !are_both_lists(left, right) && !are_both_structs(left, right) {
        return false;
    }

    let left_dtypes = unpack_dtypes(left, false);
    let right_dtypes = unpack_dtypes(right, false);

    let left_has_floats = left_dtypes.iter().any(|dt| dt.is_float());
    let right_has_floats = right_dtypes.iter().any(|dt| dt.is_float());

    left_has_floats && right_has_floats
}

/// Ensures that null values in two Series match exactly and returns an error if any mismatches are found.
fn assert_series_null_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    let null_value_mismatch = left.is_null().not_equal(&right.is_null());

    if null_value_mismatch.any() {
        return Err(polars_err!(
            assertion_error = "Series",
            "null value mismatch",
            left.null_count(),
            right.null_count()
        ));
    }

    Ok(())
}

/// Validates that NaN patterns are identical between two float Series, returning error if any mismatches are found.
fn assert_series_nan_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    if !are_both_floats(left.dtype(), right.dtype()) {
        return Ok(());
    }
    let left_nan = left.is_nan()?;
    let right_nan = right.is_nan()?;

    let nan_value_mismatch = left_nan.not_equal(&right_nan);

    let left_nan_count = left_nan.sum().unwrap_or(0);
    let right_nan_count = right_nan.sum().unwrap_or(0);

    if nan_value_mismatch.any() {
        return Err(polars_err!(
            assertion_error = "Series",
            "nan value mismatch",
            left_nan_count,
            right_nan_count
        ));
    }

    Ok(())
}

/// Verifies that two Series have values within a specified tolerance.
///
/// This function checks if the values in `left` and `right` Series that are marked as unequal
/// in the `unequal` boolean array are within the specified relative and absolute tolerances.
///
/// # Arguments
///
/// * `left` - The first Series to compare
/// * `right` - The second Series to compare
/// * `unequal` - Boolean ChunkedArray indicating which elements to check (true = check this element)
/// * `rel_tol` - Relative tolerance (relative to the maximum absolute value of the two Series)
/// * `abs_tol` - Absolute tolerance added to the relative tolerance
///
/// # Returns
///
/// * `Ok(())` if all values are within tolerance
/// * `Err` with details about problematic values if any values exceed the tolerance
///
/// # Formula
///
/// Values are considered within tolerance if:
/// `|left - right| <= max(rel_tol * max(abs(left), abs(right)), abs_tol)` OR values are exactly equal
///
fn assert_series_values_within_tolerance(
    left: &Series,
    right: &Series,
    unequal: &ChunkedArray<BooleanType>,
    rel_tol: f64,
    abs_tol: f64,
) -> PolarsResult<()> {
    let left_unequal = left.filter(unequal)?;
    let right_unequal = right.filter(unequal)?;

    let within_tolerance = is_close(&left_unequal, &right_unequal, abs_tol, rel_tol, false)?;
    if within_tolerance.all() {
        Ok(())
    } else {
        let exceeded_indices = within_tolerance.not();
        let problematic_left = left_unequal.filter(&exceeded_indices)?;
        let problematic_right = right_unequal.filter(&exceeded_indices)?;

        Err(polars_err!(
            assertion_error = "Series",
            "values not within tolerance",
            problematic_left,
            problematic_right
        ))
    }
}

/// Compares two Series for equality with configurable options for ordering, exact matching, and tolerance.
///
/// This function verifies that the values in `left` and `right` Series are equal according to
/// the specified comparison criteria. It handles different types including floats and nested types
/// with appropriate equality checks.
///
/// # Arguments
///
/// * `left` - The first Series to compare
/// * `right` - The second Series to compare
/// * `check_order` - If true, elements must be in the same order; if false, Series will be sorted before comparison
/// * `check_exact` - If true, requires exact equality; if false, allows approximate equality for floats within tolerance
/// * `rel_tol` - Relative tolerance for float comparison (used when `check_exact` is false)
/// * `abs_tol` - Absolute tolerance for float comparison (used when `check_exact` is false)
/// * `categorical_as_str` - If true, converts categorical Series to strings before comparison
///
/// # Returns
///
/// * `Ok(())` if Series match according to specified criteria
/// * `Err` with details about mismatches if Series differ
///
/// # Behavior
///
/// 1. Handles categorical Series based on `categorical_as_str` flag
/// 2. Sorts Series if `check_order` is false
/// 3. For nested float types, delegates to `assert_series_nested_values_equal`
/// 4. For non-float types or when `check_exact` is true, requires exact match
/// 5. For float types with approximate matching:
///    - Verifies null values match using `assert_series_null_values_match`
///    - Verifies NaN values match using `assert_series_nan_values_match`
///    - Verifies float values are within tolerance using `assert_series_values_within_tolerance`
///
#[allow(clippy::too_many_arguments)]
fn assert_series_values_equal(
    left: &Series,
    right: &Series,
    check_order: bool,
    check_exact: bool,
    check_dtypes: bool,
    rel_tol: f64,
    abs_tol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    // When `check_dtypes` is `false` and both series are entirely null,
    // consider them equal regardless of their underlying data types
    if !check_dtypes && left.dtype() != right.dtype() {
        if left.null_count() == left.len() && right.null_count() == right.len() {
            return Ok(());
        }
    }

    let (left, right) = if categorical_as_str {
        (
            categorical_series_to_string(left)?,
            categorical_series_to_string(right)?,
        )
    } else {
        (left.clone(), right.clone())
    };

    let (left, right) = if !check_order {
        (
            left.sort(SortOptions::default())?,
            right.sort(SortOptions::default())?,
        )
    } else {
        (left, right)
    };

    let unequal = match left.not_equal_missing(&right) {
        Ok(result) => result,
        Err(_) => {
            return Err(polars_err!(
                assertion_error = "Series",
                "incompatible data types",
                left.dtype(),
                right.dtype()
            ));
        },
    };

    if comparing_nested_floats(left.dtype(), right.dtype()) {
        let filtered_left = left.filter(&unequal)?;
        let filtered_right = right.filter(&unequal)?;

        match assert_series_nested_values_equal(
            &filtered_left,
            &filtered_right,
            check_exact,
            check_dtypes,
            rel_tol,
            abs_tol,
            categorical_as_str,
        ) {
            Ok(_) => return Ok(()),
            Err(_) => {
                return Err(polars_err!(
                    assertion_error = "Series",
                    "nested value mismatch",
                    left,
                    right
                ));
            },
        }
    }

    if !unequal.any() {
        return Ok(());
    }

    if check_exact || !left.dtype().is_float() || !right.dtype().is_float() {
        return Err(polars_err!(
            assertion_error = "Series",
            "exact value mismatch",
            left,
            right
        ));
    }

    assert_series_null_values_match(&left, &right)?;
    assert_series_nan_values_match(&left, &right)?;
    assert_series_values_within_tolerance(&left, &right, &unequal, rel_tol, abs_tol)?;

    Ok(())
}

/// Recursively compares nested Series structures (lists or structs) for equality.
///
/// This function handles the comparison of complex nested data structures by recursively
/// applying appropriate equality checks based on the nested data type.
///
/// # Arguments
///
/// * `left` - The first nested Series to compare
/// * `right` - The second nested Series to compare
/// * `check_exact` - If true, requires exact equality; if false, allows approximate equality for floats
/// * `rel_tol` - Relative tolerance for float comparison (used when `check_exact` is false)
/// * `abs_tol` - Absolute tolerance for float comparison (used when `check_exact` is false)
/// * `categorical_as_str` - If true, converts categorical Series to strings before comparison
///
/// # Returns
///
/// * `Ok(())` if nested Series match according to specified criteria
/// * `Err` with details about mismatches if Series differ
///
/// # Behavior
///
/// For List types:
/// 1. Iterates through corresponding elements in both Series
/// 2. Returns error if null values are encountered
/// 3. Creates single-element Series for each value and explodes them
/// 4. Recursively calls `assert_series_values_equal` on the exploded Series
///
/// For Struct types:
/// 1. Unnests both struct Series to access their columns
/// 2. Iterates through corresponding columns
/// 3. Recursively calls `assert_series_values_equal` on each column pair
///
fn assert_series_nested_values_equal(
    left: &Series,
    right: &Series,
    check_exact: bool,
    check_dtypes: bool,
    rel_tol: f64,
    abs_tol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    if are_both_lists(left.dtype(), right.dtype()) {
        let zipped = left.iter().zip(right.iter());

        for (s1, s2) in zipped {
            if s1.is_null() || s2.is_null() {
                return Err(polars_err!(
                    assertion_error = "Series",
                    "nested value mismatch",
                    s1,
                    s2
                ));
            } else {
                let s1_series = Series::new("".into(), std::slice::from_ref(&s1));
                let s2_series = Series::new("".into(), std::slice::from_ref(&s2));

                assert_series_values_equal(
                    &s1_series.explode(ExplodeOptions {
                        empty_as_null: true,
                        keep_nulls: true,
                    })?,
                    &s2_series.explode(ExplodeOptions {
                        empty_as_null: true,
                        keep_nulls: true,
                    })?,
                    true,
                    check_exact,
                    check_dtypes,
                    rel_tol,
                    abs_tol,
                    categorical_as_str,
                )?
            }
        }
    } else {
        let ls = left.struct_()?.clone().unnest();
        let rs = right.struct_()?.clone().unnest();

        for col_name in ls.get_column_names() {
            let s1_column = ls.column(col_name)?;
            let s2_column = rs.column(col_name)?;

            let s1_series = s1_column.as_materialized_series();
            let s2_series = s2_column.as_materialized_series();

            assert_series_values_equal(
                s1_series,
                s2_series,
                true,
                check_exact,
                check_dtypes,
                rel_tol,
                abs_tol,
                categorical_as_str,
            )?
        }
    }

    Ok(())
}

/// Verifies that two Series are equal according to a set of configurable criteria.
///
/// This function serves as the main entry point for comparing Series, checking various
/// metadata properties before comparing the actual values.
///
/// # Arguments
///
/// * `left` - The first Series to compare
/// * `right` - The second Series to compare
/// * `options` - A `SeriesEqualOptions` struct containing configuration parameters:
///   * `check_names` - If true, verifies Series names match
///   * `check_dtypes` - If true, verifies data types match
///   * `check_order` - If true, elements must be in the same order
///   * `check_exact` - If true, requires exact equality for float values
///   * `rel_tol` - Relative tolerance for float comparison
///   * `abs_tol` - Absolute tolerance for float comparison
///   * `categorical_as_str` - If true, converts categorical Series to strings before comparison
///
/// # Returns
///
/// * `Ok(())` if Series match according to all specified criteria
/// * `Err` with details about the first mismatch encountered:
///   * Length mismatch
///   * Name mismatch (if checking names)
///   * Data type mismatch (if checking dtypes)
///   * Value mismatches (via `assert_series_values_equal`)
///
/// # Order of Checks
///
/// 1. Series length
/// 2. Series names (if `check_names` is true)
/// 3. Data types (if `check_dtypes` is true)
/// 4. Series values (delegated to `assert_series_values_equal`)
///
pub fn assert_series_equal(
    left: &Series,
    right: &Series,
    options: SeriesEqualOptions,
) -> PolarsResult<()> {
    // Short-circuit if they're the same series object
    if std::ptr::eq(left, right) {
        return Ok(());
    }

    if left.len() != right.len() {
        return Err(polars_err!(
            assertion_error = "Series",
            "length mismatch",
            left.len(),
            right.len()
        ));
    }

    if options.check_names && left.name() != right.name() {
        return Err(polars_err!(
            assertion_error = "Series",
            "name mismatch",
            left.name(),
            right.name()
        ));
    }

    if options.check_dtypes && left.dtype() != right.dtype() {
        return Err(polars_err!(
            assertion_error = "Series",
            "dtype mismatch",
            left.dtype(),
            right.dtype()
        ));
    }

    assert_series_values_equal(
        left,
        right,
        options.check_order,
        options.check_exact,
        options.check_dtypes,
        options.rel_tol,
        options.abs_tol,
        options.categorical_as_str,
    )
}

/// Configuration options for comparing DataFrame equality.
///
/// Controls the behavior of DataFrame equality comparisons by specifying
/// which aspects to check and the tolerance for floating point comparisons.
pub struct DataFrameEqualOptions {
    /// Whether to check that rows appear in the same order.
    pub check_row_order: bool,
    /// Whether to check that columns appear in the same order.
    pub check_column_order: bool,
    /// Whether to check that the data types match for corresponding columns.
    pub check_dtypes: bool,
    /// Whether to check for exact equality (true) or approximate equality (false) for floating point values.
    pub check_exact: bool,
    /// Relative tolerance for approximate equality of floating point values.
    pub rel_tol: f64,
    /// Absolute tolerance for approximate equality of floating point values.
    pub abs_tol: f64,
    /// Whether to compare categorical values as strings.
    pub categorical_as_str: bool,
}

impl Default for DataFrameEqualOptions {
    /// Creates a new `DataFrameEqualOptions` with default settings.
    ///
    /// Default configuration:
    /// - Checks row order, column order, and data types
    /// - Uses approximate equality comparisons for floating point values
    /// - Sets relative tolerance to 1e-5 and absolute tolerance to 1e-8 for floating point comparisons
    /// - Does not convert categorical values to strings for comparison
    fn default() -> Self {
        Self {
            check_row_order: true,
            check_column_order: true,
            check_dtypes: true,
            check_exact: false,
            rel_tol: 1e-5,
            abs_tol: 1e-8,
            categorical_as_str: false,
        }
    }
}

impl DataFrameEqualOptions {
    /// Creates a new `DataFrameEqualOptions` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to check that rows appear in the same order.
    pub fn with_check_row_order(mut self, value: bool) -> Self {
        self.check_row_order = value;
        self
    }

    /// Sets whether to check that columns appear in the same order.
    pub fn with_check_column_order(mut self, value: bool) -> Self {
        self.check_column_order = value;
        self
    }

    /// Sets whether to check that data types match for corresponding columns.
    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    /// Sets whether to check for exact equality (true) or approximate equality (false) for floating point values.
    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    /// Sets the relative tolerance for approximate equality of floating point values.
    pub fn with_rel_tol(mut self, value: f64) -> Self {
        self.rel_tol = value;
        self
    }

    /// Sets the absolute tolerance for approximate equality of floating point values.
    pub fn with_abs_tol(mut self, value: f64) -> Self {
        self.abs_tol = value;
        self
    }

    /// Sets whether to compare categorical values as strings.
    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

/// Compares schemas for equality based on specified criteria.
///
/// This function validates that two schemas have compatible schemas by checking
/// column names, their order, and optionally their data types according to the
/// provided configuration parameters.
///
/// # Arguments
///
/// * `left` - The first schema to compare
/// * `right` - The second schema to compare
/// * `check_dtypes` - If true, requires data types to match for corresponding columns
/// * `check_column_order` - If true, requires columns to appear in the same order
///
/// # Returns
///
/// * `Ok(())` if schemas match according to specified criteria
/// * `Err` with details about schema mismatches if schemas differ
///
/// # Behavior
///
/// The function performs schema validation in the following order:
///
/// 1. **Fast path**: Returns immediately if schemas are identical
/// 2. **Column name validation**: Ensures both schemas have the same set of column names
///    - Reports columns present in left but missing in right
///    - Reports columns present in right but missing in left
/// 3. **Column order validation**: If `check_column_order` is true, verifies columns appear in the same sequence
/// 4. **Data type validation**: If `check_dtypes` is true, ensures corresponding columns have matching data types
///    - When `check_column_order` is false, compares data type sets for equality
///    - When `check_column_order` is true, performs more precise type checking
///
pub fn assert_schema_equal(
    left_schema: &Schema,
    right_schema: &Schema,
    check_dtypes: bool,
    check_column_order: bool,
) -> PolarsResult<()> {
    assert_schema_equal_impl(
        left_schema,
        right_schema,
        check_dtypes,
        check_column_order,
        "Schemas",
    )
}

fn assert_schema_equal_impl(
    left_schema: &Schema,
    right_schema: &Schema,
    check_dtypes: bool,
    check_column_order: bool,
    context: &'static str,
) -> PolarsResult<()> {
    let mut one_sided_names: Vec<&PlSmallStr> = vec![];
    let mut column_name_order_mismatch = false;
    let mut dtype_mismatch = false;

    for (l_idx, (l_name, l_dtype)) in left_schema.iter().enumerate() {
        let Some((r_idx, _, r_dtype)) = right_schema.get_full(l_name) else {
            one_sided_names.reserve_exact(left_schema.len() - l_idx);
            one_sided_names.push(l_name);
            continue;
        };

        if check_column_order && l_idx != r_idx {
            column_name_order_mismatch = true;
        }

        if check_dtypes && l_dtype != r_dtype {
            dtype_mismatch = true;
        }
    }

    if !one_sided_names.is_empty() {
        polars_bail!(
            assertion_error = context,
            format!(
                "columns mismatch: {:?} in left, but not in right",
                one_sided_names
            ),
            left_schema.names_display(),
            right_schema.names_display()
        )
    }

    debug_assert!(right_schema.len() >= left_schema.len());

    if right_schema.len() > left_schema.len() {
        one_sided_names.reserve_exact(right_schema.len() - left_schema.len());
        one_sided_names.extend(
            right_schema
                .iter_names()
                .filter(|name| !left_schema.contains(name)),
        );

        polars_bail!(
            assertion_error = context,
            format!(
                "columns mismatch: {:?} in right, but not in left",
                one_sided_names
            ),
            left_schema.names_display(),
            right_schema.names_display()
        )
    }

    debug_assert_eq!(left_schema.len(), right_schema.len());

    if check_column_order && column_name_order_mismatch {
        polars_bail!(
            assertion_error = context,
            "columns are not in the same order",
            left_schema.names_display(),
            right_schema.names_display()
        )
    }

    if check_dtypes && dtype_mismatch {
        polars_bail!(
            assertion_error = context,
            "dtypes do not match",
            left_schema.values_display(),
            right_schema.values_display()
        )
    }

    Ok(())
}

/// Verifies that two DataFrames are equal according to a set of configurable criteria.
///
/// This function serves as the main entry point for comparing DataFrames, first validating
/// schema compatibility and then comparing the actual data values column by column.
///
/// # Arguments
///
/// * `left` - The first DataFrame to compare
/// * `right` - The second DataFrame to compare
/// * `options` - A `DataFrameEqualOptions` struct containing configuration parameters:
///   * `check_row_order` - If true, rows must be in the same order
///   * `check_column_order` - If true, columns must be in the same order
///   * `check_dtypes` - If true, verifies data types match for corresponding columns
///   * `check_exact` - If true, requires exact equality for float values
///   * `rel_tol` - Relative tolerance for float comparison
///   * `abs_tol` - Absolute tolerance for float comparison
///   * `categorical_as_str` - If true, converts categorical values to strings before comparison
///
/// # Returns
///
/// * `Ok(())` if DataFrames match according to all specified criteria
/// * `Err` with details about the first mismatch encountered:
///   * Schema mismatches (column names, order, or data types)
///   * Height (row count) mismatch
///   * Value mismatches in specific columns
///
/// # Order of Checks
///
/// 1. Schema validation (column names, order, and data types via `assert_schema_equal`)
/// 2. DataFrame height (row count)
/// 3. Row ordering (sorts both DataFrames if `check_row_order` is false)
/// 4. Column-by-column value comparison (delegated to `assert_series_values_equal`)
///
/// # Behavior
///
/// When `check_row_order` is false, both DataFrames are sorted using all columns to ensure
/// consistent ordering before value comparison. This allows for row-order-independent equality
/// checking while maintaining deterministic results.
///
pub fn assert_dataframe_equal(
    left: &DataFrame,
    right: &DataFrame,
    options: DataFrameEqualOptions,
) -> PolarsResult<()> {
    // Short-circuit if they're the same DataFrame object
    if std::ptr::eq(left, right) {
        return Ok(());
    }

    let left_schema = left.schema();
    let right_schema = right.schema();

    assert_schema_equal_impl(
        left_schema,
        right_schema,
        options.check_dtypes,
        options.check_column_order,
        "DataFrames",
    )?;

    if left.height() != right.height() {
        return Err(polars_err!(
            assertion_error = "DataFrames",
            "height (row count) mismatch",
            left.height(),
            right.height()
        ));
    }

    let left_cols = left.get_column_names_owned();

    let (left, right) = if !options.check_row_order {
        (
            left.sort(left_cols.clone(), SortMultipleOptions::default())?,
            right.sort(left_cols.clone(), SortMultipleOptions::default())?,
        )
    } else {
        (left.clone(), right.clone())
    };

    for col in left_cols.iter() {
        let s_left = left.column(col)?;
        let s_right = right.column(col)?;

        let s_left_series = s_left.as_materialized_series();
        let s_right_series = s_right.as_materialized_series();

        match assert_series_values_equal(
            s_left_series,
            s_right_series,
            true,
            options.check_exact,
            options.check_dtypes,
            options.rel_tol,
            options.abs_tol,
            options.categorical_as_str,
        ) {
            Ok(_) => {},
            Err(_) => {
                return Err(polars_err!(
                    assertion_error = "DataFrames",
                    format!("value mismatch for column {:?}", col),
                    format!("{:?}", s_left_series),
                    format!("{:?}", s_right_series)
                ));
            },
        }
    }

    Ok(())
}
