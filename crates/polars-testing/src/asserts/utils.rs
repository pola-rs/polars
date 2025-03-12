use std::ops::{BitAnd, BitOr, BitXor, Not};

use polars_core::datatypes::unpack_dtypes;
use polars_core::prelude::*;
use polars_ops::series::abs;

// **Note:** Created a struct with a default implementation
// for the user to create default arguments. This seems to be the
// idiomatic approach and resolves Clippy's "too many parameters (9/7)"
// suggestion.
pub struct SeriesEqualOptions {
    pub check_dtypes: bool,
    pub check_names: bool,
    pub check_order: bool,
    pub check_exact: bool,
    pub rtol: f64,
    pub atol: f64,
    pub categorical_as_str: bool,
}

impl Default for SeriesEqualOptions {
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_check_dtypes(mut self, value: bool) -> Self {
        self.check_dtypes = value;
        self
    }

    pub fn with_check_names(mut self, value: bool) -> Self {
        self.check_names = value;
        self
    }

    pub fn with_check_order(mut self, value: bool) -> Self {
        self.check_order = value;
        self
    }

    pub fn with_check_exact(mut self, value: bool) -> Self {
        self.check_exact = value;
        self
    }

    pub fn with_rtol(mut self, value: f64) -> Self {
        self.rtol = value;
        self
    }

    pub fn with_atol(mut self, value: f64) -> Self {
        self.atol = value;
        self
    }

    pub fn with_categorical_as_str(mut self, value: bool) -> Self {
        self.categorical_as_str = value;
        self
    }
}

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

pub fn categorical_series_to_string(s: &Series) -> Series {
    let dtype = s.dtype();
    let noncat_dtype = categorical_dtype_to_string_dtype(dtype);

    if *dtype != noncat_dtype {
        s.cast(&noncat_dtype).unwrap()
    } else {
        s.clone()
    }
}

pub fn comparing_floats(left: &DataType, right: &DataType) -> bool {
    left.is_float() && right.is_float()
}

pub fn comparing_lists(left: &DataType, right: &DataType) -> bool {
    matches!(left, DataType::List(_) | DataType::Array(_, _))
        && matches!(right, DataType::List(_) | DataType::Array(_, _))
}

pub fn comparing_structs(left: &DataType, right: &DataType) -> bool {
    left.is_struct() && right.is_struct()
}

// **Change**: When translating this code to Rust, there originally was
// no `unpack_dtypes()` functionality in the code base, so I created it in
// `polars-core/src/datatypes/dtype.rs` and Gijs merged the PR (#21574).
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

// **Change:** The Python function originally took both `left` and `right`
// as input parameters and returned both, but I just made it take a
// single Series and used tuple destructuring.
pub fn sort_series(s: &Series) -> PolarsResult<Series> {
    s.sort(SortOptions::default())
        .map_err(|_| polars_err!(op = "sort", s.dtype()))
}

pub fn assert_series_null_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    let null_value_mismatch = left.is_null().not_equal(&right.is_null());

    // **Change**: Rather than simply returning the Series as Lists
    // I thought it made more sense to return null value counts from the
    // left and right Series
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

pub fn assert_series_nan_values_match(left: &Series, right: &Series) -> PolarsResult<()> {
    if !comparing_floats(left.dtype(), right.dtype()) {
        return Ok(());
    }
    let left_nan = left.is_nan()?;
    let right_nan = right.is_nan()?;

    let nan_value_mismatch = left_nan.not_equal(&right_nan);

    // **Change**: Rather than simply returning the Series as Lists
    // I thought it made more sense to return NaN value counts from the
    // left and right Series
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

pub fn assert_series_values_within_tolerance(
    left: &Series,
    right: &Series,
    // **Change**: In Python, this input type is a Series,
    // but in Rust, to use the `.filter()` method on a Series
    // the object inside the filter must be a `ChunkedArray`.
    unequal: &ChunkedArray<BooleanType>,
    rtol: f64,
    atol: f64,
) -> PolarsResult<()> {
    let left_unequal = left.filter(unequal)?;
    let right_unequal = right.filter(unequal)?;

    let difference = (&left_unequal - &right_unequal)?;
    let abs_difference = abs(&difference)?;

    let right_abs = abs(&right_unequal)?;

    // **Change**: The scalar values needed to be converted into Series
    // so that arithmetic operations between the Series could happen.
    // I looked through the Rust API documentation and didn't see
    // any method for multiplying a Series by a scalar.
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
        // **Change**: Rather than simply returning the Series as Lists
        // I thought it made more sense to return the problematic values in both
        // left and right Series
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
        (sort_series(&left)?, sort_series(&right)?)
    } else {
        (left.clone(), right.clone())
    };

    // **Change**: After looking through the Rust API documentation, it seems
    // there is no direct `ne_missing()` function in Rust like there is in
    // Python for Series, so I had to break it down into smaller parts
    // which can be refactored later, if that functionality is eventually made.
    let left_null = left.is_null();
    let right_null = right.is_null();
    let null_mismatch = left_null.clone().bitxor(right_null.clone());

    let values_equal = left.equal(&right)?;
    let both_null = left_null.bitand(right_null);

    let combined_equal = values_equal.bitor(both_null);
    let unequal = combined_equal.not().bitor(null_mismatch);

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
                // **Change**: This had to convert `AnyValue` to Series because
                // the input type for `assert_series_values_equal()` are Series
                // objects and not `AnyValue`, which would cause an error.
                let s1_series = Series::new("".into(), &[s1.clone()]);
                let s2_series = Series::new("".into(), &[s2.clone()]);

                // **Change**: Using `explode()` to prevent infinite recursion.
                // When an `AnyValue` representing a nested structure (like a list) is wrapped in a new Series,
                // it creates another layer of nesting. Without `explode()`, this would cause infinite recursion
                // as each recursive call would add more nesting instead of resolving it.
                // The `explode()` call flattens one level of the nested structure before comparison,
                // breaking the recursive cycle.
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
        // **Change**: This section had to be modified because
        // using `unnest()` on a Struct type in Rust creates a DataFrame
        // not a Series. Thus, I had to find a workaround so that the
        // `assert_series_values_equal()` function would have the proper
        // input of Series.
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

pub fn assert_series_equal(
    left: &Series,
    right: &Series,
    options: SeriesEqualOptions,
) -> PolarsResult<()> {
    // **Change**: The Python code has an `_assert_correct_input_type()`
    // function to make sure that both inputs are Series. However,
    // this was not implemented in Rust due to Rust's strict
    // static type-checking system.

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
