use std::ops::Not;

use polars_core::datatypes::unpack_dtypes;
use polars_core::prelude::*;
use polars_ops::series::abs;

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
    pub rtol: f64,
    /// Absolute tolerance for approximate equality of floating point values.
    pub atol: f64,
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
            rtol: 1e-5,
            atol: 1e-8,
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
    pub fn with_rtol(mut self, value: f64) -> Self {
        self.rtol = value;
        self
    }

    /// Sets the absolute tolerance for approximate equality of floating point values.
    pub fn with_atol(mut self, value: f64) -> Self {
        self.atol = value;
        self
    }

    /// Sets whether to compare categorical values as strings.
    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

/// Change a (possibly nested) Categorical data type to a String data type.
pub fn categorical_dtype_to_string_dtype(dtype: &DataType) -> DataType {
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
pub fn categorical_series_to_string(s: &Series) -> Series {
    let dtype = s.dtype();
    let noncat_dtype = categorical_dtype_to_string_dtype(dtype);

    if *dtype != noncat_dtype {
        s.cast(&noncat_dtype).unwrap()
    } else {
        s.clone()
    }
}

/// Returns true if both DataTypes are floating point types.
pub fn comparing_floats(left: &DataType, right: &DataType) -> bool {
    left.is_float() && right.is_float()
}

/// Returns true if both DataTypes are list-like (either List or Array types).
pub fn comparing_lists(left: &DataType, right: &DataType) -> bool {
    matches!(left, DataType::List(_) | DataType::Array(_, _))
        && matches!(right, DataType::List(_) | DataType::Array(_, _))
}

/// Returns true if both DataTypes are struct types.
pub fn comparing_structs(left: &DataType, right: &DataType) -> bool {
    left.is_struct() && right.is_struct()
}

/// Returns true if both DataTypes are nested types (lists or structs) that contain floating point types within them.
/// First checks if both types are either lists or structs, then unpacks their nested DataTypes to determine if
/// at least one floating point type exists in each of the nested structures.
pub fn comparing_nested_floats(left: &DataType, right: &DataType) -> bool {
    if !comparing_lists(left, right) && !comparing_structs(left, right) {
        return false;
    }

    let left_dtypes = unpack_dtypes(left, false);
    let right_dtypes = unpack_dtypes(right, false);

    let left_has_floats = left_dtypes.iter().any(|dt| dt.is_float());
    let right_has_floats = right_dtypes.iter().any(|dt| dt.is_float());

    left_has_floats && right_has_floats
}

/// Ensures that null values in two Series match exactly and returns an error if any mismatches are found.
pub fn assert_series_null_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
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
pub fn assert_series_nan_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    if !comparing_floats(left.dtype(), right.dtype()) {
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
/// * `rtol` - Relative tolerance (multiplied by the absolute value of the right Series)
/// * `atol` - Absolute tolerance added to the relative tolerance
///
/// # Returns
///
/// * `Ok(())` if all values are within tolerance
/// * `Err` with details about problematic values if any values exceed the tolerance
///
/// # Formula
///
/// Values are considered within tolerance if:
/// `|left - right| <= (rtol * |right| + atol)` OR values are exactly equal
///
pub fn assert_series_values_within_tolerance(
    left: &Series,
    right: &Series,
    unequal: &ChunkedArray<BooleanType>,
    rtol: f64,
    atol: f64,
) -> PolarsResult<()> {
    let left_unequal = left.filter(unequal)?;
    let right_unequal = right.filter(unequal)?;

    let difference = (&left_unequal - &right_unequal)?;
    let abs_difference = abs(&difference)?;

    let right_abs = abs(&right_unequal)?;

    let rtol_series = Series::new("rtol".into(), &[rtol]);
    let atol_series = Series::new("atol".into(), &[atol]);

    let rtol_part = (&right_abs * &rtol_series)?;
    let tolerance = (&rtol_part + &atol_series)?;

    let finite_mask = right_unequal.is_finite()?;
    let diff_within_tol = abs_difference.lt_eq(&tolerance)?;
    let equal_values = left_unequal.equal(&right_unequal)?;

    let within_tolerance = (diff_within_tol & finite_mask) | equal_values;

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
/// * `rtol` - Relative tolerance for float comparison (used when `check_exact` is false)
/// * `atol` - Absolute tolerance for float comparison (used when `check_exact` is false)
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
pub fn assert_series_values_equal(
    left: &Series,
    right: &Series,
    check_order: bool,
    check_exact: bool,
    rtol: f64,
    atol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    let (left, right) = if categorical_as_str {
        (
            categorical_series_to_string(left),
            categorical_series_to_string(right),
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
        (left.clone(), right.clone())
    };

    let unequal = left.not_equal_missing(&right)?;

    if comparing_nested_floats(left.dtype(), right.dtype()) {
        let filtered_left = left.filter(&unequal)?;
        let filtered_right = right.filter(&unequal)?;

        match assert_series_nested_values_equal(
            &filtered_left,
            &filtered_right,
            check_exact,
            rtol,
            atol,
            categorical_as_str,
        ) {
            Ok(_) => {
                return Ok(());
            },
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
    assert_series_values_within_tolerance(&left, &right, &unequal, rtol, atol)?;

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
/// * `rtol` - Relative tolerance for float comparison (used when `check_exact` is false)
/// * `atol` - Absolute tolerance for float comparison (used when `check_exact` is false)
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
pub fn assert_series_nested_values_equal(
    left: &Series,
    right: &Series,
    check_exact: bool,
    rtol: f64,
    atol: f64,
    categorical_as_str: bool,
) -> PolarsResult<()> {
    if comparing_lists(left.dtype(), right.dtype()) {
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
                let s1_series = Series::new("".into(), &[s1.clone()]);
                let s2_series = Series::new("".into(), &[s2.clone()]);

                match assert_series_values_equal(
                    &s1_series.explode()?,
                    &s2_series.explode()?,
                    true,
                    check_exact,
                    rtol,
                    atol,
                    categorical_as_str,
                ) {
                    Ok(_) => continue,
                    Err(e) => return Err(e),
                }
            }
        }
    } else {
        let ls = left.struct_()?.clone().unnest();
        let rs = right.struct_()?.clone().unnest();

        let ls_cols = ls.get_columns();
        let rs_cols = rs.get_columns();

        for (s1, s2) in ls_cols.iter().zip(rs_cols.iter()) {
            match assert_series_values_equal(
                s1.as_series().unwrap(),
                s2.as_series().unwrap(),
                true,
                check_exact,
                rtol,
                atol,
                categorical_as_str,
            ) {
                Ok(_) => continue,
                Err(e) => return Err(e),
            }
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
///   * `rtol` - Relative tolerance for float comparison
///   * `atol` - Absolute tolerance for float comparison
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
            "data type mismatch",
            left.dtype(),
            right.dtype()
        ));
    }

    assert_series_values_equal(
        left,
        right,
        options.check_order,
        options.check_exact,
        options.rtol,
        options.atol,
        options.categorical_as_str,
    )
}
