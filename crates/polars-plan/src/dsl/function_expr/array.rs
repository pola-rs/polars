use polars_ops::chunked_array::array::*;

use super::*;
use crate::{map, map_as_slice};
use polars_core::with_match_physical_numeric_polars_type;
use std::collections::HashMap;
use arrow::bitmap::MutableBitmap;
use arrow::array::{PrimitiveArray, FixedSizeListArray};
use arrayvec::ArrayString;


#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, Default)]
#[derive(Serialize, Deserialize)]
pub struct ArrayKwargs {
    // Not sure how to get a serializable DataType here
    // For prototype, use fixed size string
    pub dtype_expr: ArrayString::<256>,
}



#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrayFunction {
    Min,
    Max,
    Sum,
    ToList,
    Unique(bool),
    NUnique,
    Std(u8),
    Var(u8),
    Median,
    #[cfg(feature = "array_any_all")]
    Any,
    #[cfg(feature = "array_any_all")]
    All,
    Array(ArrayKwargs),
    Sort(SortOptions),
    Reverse,
    ArgMin,
    ArgMax,
    Get(bool),
    Join(bool),
    #[cfg(feature = "is_in")]
    Contains,
    #[cfg(feature = "array_count")]
    CountMatches,
    Shift,
}

impl ArrayFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use ArrayFunction::*;
        match self {
            Min | Max => mapper.map_to_list_and_array_inner_dtype(),
            Sum => mapper.nested_sum_type(),
            ToList => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
            Unique(_) => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
            NUnique => mapper.with_dtype(IDX_DTYPE),
            Std(_) => mapper.map_to_float_dtype(),
            Var(_) => mapper.map_to_float_dtype(),
            Median => mapper.map_to_float_dtype(),
            #[cfg(feature = "array_any_all")]
            Any | All => mapper.with_dtype(DataType::Boolean),
            // TODO: Figure out how to bind keyword argument
            Array(kwargs) => array_output_type(mapper.args(), kwargs),
            Sort(_) => mapper.with_same_dtype(),
            Reverse => mapper.with_same_dtype(),
            ArgMin | ArgMax => mapper.with_dtype(IDX_DTYPE),
            Get(_) => mapper.map_to_list_and_array_inner_dtype(),
            Join(_) => mapper.with_dtype(DataType::String),
            #[cfg(feature = "is_in")]
            Contains => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "array_count")]
            CountMatches => mapper.with_dtype(IDX_DTYPE),
            Shift => mapper.with_same_dtype(),
        }
    }
}

fn deserialize_dtype(dtype_expr: &str) -> PolarsResult<Option<DataType>> {
    match dtype_expr.len() {
        0 => Ok(None),
        _ => match serde_json::from_str::<Expr>(dtype_expr) {
            Ok(Expr::DtypeColumn(dtypes)) if dtypes.len() == 1 => Ok(Some(dtypes[0].clone())),
            Ok(_) => Err(
                polars_err!(ComputeError: "Expected a DtypeColumn expression with a single dtype"),
            ),
            Err(_) => Err(polars_err!(ComputeError: "Could not deserialize dtype expression")),
        },
    }
}

fn get_expected_dtype(inputs: &[DataType], kwargs: &ArrayKwargs) -> PolarsResult<DataType> {
    // Decide what dtype to use for the constructed array
    // For now, the logic is to use the dtype in kwargs, if specified
    // Otherwise, use the type of the first column.
    //
    // An alternate idea could be to call try_get_supertype for the types.
    // Or logic like DataFrame::get_supertype_all
    // The problem is, I think this cast may be too general and we may only want to support primitive types
    // Also, we don't support String yet.
    let expected_dtype = deserialize_dtype(&kwargs.dtype_expr)?
        .unwrap_or(inputs[0].clone());
    Ok(expected_dtype)
}

fn array_output_type(input_fields: &[Field], kwargs: &ArrayKwargs) -> PolarsResult<Field> {
    // Expected target type is either the provided dtype or the type of the first column
    let dtypes: Vec<DataType> = input_fields.into_iter().map(|f| f.dtype().clone()).collect();
    let expected_dtype  = get_expected_dtype(&dtypes, kwargs)?;

    for field in input_fields.iter() {
        if !field.dtype().is_numeric() {
            polars_bail!(ComputeError: "all input fields must be numeric")
        }
    }

    Ok(Field::new(
        PlSmallStr::from_static("array"),
        DataType::Array(Box::new(expected_dtype), input_fields.len()),
    ))
}

fn map_array_dtype_to_list_dtype(datatype: &DataType) -> PolarsResult<DataType> {
    if let DataType::Array(inner, _) = datatype {
        Ok(DataType::List(inner.clone()))
    } else {
        polars_bail!(ComputeError: "expected array dtype")
    }
}

impl Display for ArrayFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ArrayFunction::*;
        let name = match self {
            Min => "min",
            Max => "max",
            Sum => "sum",
            ToList => "to_list",
            Unique(_) => "unique",
            NUnique => "n_unique",
            Std(_) => "std",
            Var(_) => "var",
            Median => "median",
            #[cfg(feature = "array_any_all")]
            Any => "any",
            #[cfg(feature = "array_any_all")]
            All => "all",
            Array(_) => "array",
            Sort(_) => "sort",
            Reverse => "reverse",
            ArgMin => "arg_min",
            ArgMax => "arg_max",
            Get(_) => "get",
            Join(_) => "join",
            #[cfg(feature = "is_in")]
            Contains => "contains",
            #[cfg(feature = "array_count")]
            CountMatches => "count_matches",
            Shift => "shift",
        };
        write!(f, "arr.{name}")
    }
}

impl From<ArrayFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: ArrayFunction) -> Self {
        use ArrayFunction::*;
        match func {
            Min => map!(min),
            Max => map!(max),
            Sum => map!(sum),
            ToList => map!(to_list),
            Unique(stable) => map!(unique, stable),
            NUnique => map!(n_unique),
            Std(ddof) => map!(std, ddof),
            Var(ddof) => map!(var, ddof),
            Median => map!(median),
            #[cfg(feature = "array_any_all")]
            Any => map!(any),
            #[cfg(feature = "array_any_all")]
            All => map!(all),
            Array(kwargs) => map_as_slice!(array_new, kwargs),
            Sort(options) => map!(sort, options),
            Reverse => map!(reverse),
            ArgMin => map!(arg_min),
            ArgMax => map!(arg_max),
            Get(null_on_oob) => map_as_slice!(get, null_on_oob),
            Join(ignore_nulls) => map_as_slice!(join, ignore_nulls),
            #[cfg(feature = "is_in")]
            Contains => map_as_slice!(contains),
            #[cfg(feature = "array_count")]
            CountMatches => map_as_slice!(count_matches),
            Shift => map_as_slice!(shift),
        }
    }
}

// Create a new array from a slice of series
fn array_new(inputs: &[Column], kwargs: ArrayKwargs) -> PolarsResult<Column> {
    array_internal(inputs, kwargs)
}
fn array_internal(inputs: &[Column], kwargs: ArrayKwargs) -> PolarsResult<Column> {
    let dtypes: Vec<DataType> = inputs.into_iter().map(|f| f.dtype().clone()).collect();
    let expected_dtype  = get_expected_dtype(&dtypes, &kwargs)?;

    // This conversion is yuck, there is probably a standard way to go from &[Column] to &[Series]
    let series: Vec<Series> = inputs.iter().map(|col| col.clone().take_materialized_series()).collect();

    // Convert dtype to native numeric type and invoke array_numeric
    let res_series = with_match_physical_numeric_polars_type!(expected_dtype, |$T| {
        array_numeric::<$T>(&series[..], &expected_dtype)
    })?;

    Ok(res_series.into_column())
}

// Combine numeric series into an array
fn array_numeric<'a, T: PolarsNumericType>(inputs: &[Series], dtype: &DataType)
                                           -> PolarsResult<Series> {
    let rows = inputs[0].len();
    let cols = inputs.len();
    let capacity = cols * rows;

    let mut values: Vec<T::Native> = vec![T::Native::default(); capacity];

    // Support for casting
    // Cast fields to the target dtype as needed
    let mut casts = HashMap::new();
    for j in 0..cols {
        if inputs[j].dtype() != dtype {
            let cast_input = inputs[j].cast(dtype)?;
            casts.insert(j, cast_input);
        }
    }

    let mut cols_ca = Vec::new();
    for j in 0..cols {
        if inputs[j].dtype() != dtype {
            cols_ca.push(casts.get(&j).expect("expect conversion").unpack::<T>()?);
        } else {
            cols_ca.push(inputs[j].unpack::<T>()?);
        }
    }

    for i in 0..rows {
        for j in 0..cols {
            values[i * cols + j] = unsafe { cols_ca[j].value_unchecked(i) };
        }
    }

    let validity = if cols_ca.iter().any(|col| col.has_nulls()) {
        let mut validity = MutableBitmap::from_len_zeroed(capacity);
        for (j, col) in cols_ca.iter().enumerate() {
            let mut row_offset = 0;
            for chunk in col.chunks() {
                if let Some(chunk_validity) = chunk.validity() {
                    for set_bit in chunk_validity.true_idx_iter() {
                        validity.set(cols * (row_offset + set_bit) + j, true);
                    }
                } else {
                    for chunk_row in 0..chunk.len() {
                        validity.set(cols * (row_offset + chunk_row) + j, true);
                    }
                }
                row_offset += chunk.len();
            }
        }
        Some(validity.into())
    } else {
        None
    };

    let values_array = PrimitiveArray::from_vec(values).with_validity(validity);
    let dtype = DataType::Array(Box::new(dtype.clone()), cols);
    let arrow_dtype = dtype.to_arrow(CompatLevel::newest());
    let array = FixedSizeListArray::try_new(arrow_dtype.clone(), Box::new(values_array), None)?;
    Ok(unsafe {Series::_try_from_arrow_unchecked("Array".into(), vec![Box::new(array)], &arrow_dtype)?})
}

pub(super) fn max(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_max().into())
}

pub(super) fn min(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_min().into())
}

pub(super) fn sum(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_sum().map(Column::from)
}

pub(super) fn std(s: &Column, ddof: u8) -> PolarsResult<Column> {
    s.array()?.array_std(ddof).map(Column::from)
}

pub(super) fn var(s: &Column, ddof: u8) -> PolarsResult<Column> {
    s.array()?.array_var(ddof).map(Column::from)
}
pub(super) fn median(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_median().map(Column::from)
}

pub(super) fn unique(s: &Column, stable: bool) -> PolarsResult<Column> {
    let ca = s.array()?;
    let out = if stable {
        ca.array_unique_stable()
    } else {
        ca.array_unique()
    };
    out.map(|ca| ca.into_column())
}

pub(super) fn n_unique(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_n_unique()?.into_column())
}

pub(super) fn to_list(s: &Column) -> PolarsResult<Column> {
    let list_dtype = map_array_dtype_to_list_dtype(s.dtype())?;
    s.cast(&list_dtype)
}

#[cfg(feature = "array_any_all")]
pub(super) fn any(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_any().map(Column::from)
}

#[cfg(feature = "array_any_all")]
pub(super) fn all(s: &Column) -> PolarsResult<Column> {
    s.array()?.array_all().map(Column::from)
}

pub(super) fn sort(s: &Column, options: SortOptions) -> PolarsResult<Column> {
    Ok(s.array()?.array_sort(options)?.into_column())
}

pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_reverse().into_column())
}

pub(super) fn arg_min(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_arg_min().into_column())
}

pub(super) fn arg_max(s: &Column) -> PolarsResult<Column> {
    Ok(s.array()?.array_arg_max().into_column())
}

pub(super) fn get(s: &[Column], null_on_oob: bool) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let index = s[1].cast(&DataType::Int64)?;
    let index = index.i64().unwrap();
    ca.array_get(index, null_on_oob).map(Column::from)
}

pub(super) fn join(s: &[Column], ignore_nulls: bool) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let separator = s[1].str()?;
    ca.array_join(separator, ignore_nulls).map(Column::from)
}

#[cfg(feature = "is_in")]
pub(super) fn contains(s: &[Column]) -> PolarsResult<Column> {
    let array = &s[0];
    let item = &s[1];
    polars_ensure!(matches!(array.dtype(), DataType::Array(_, _)),
        SchemaMismatch: "invalid series dtype: expected `Array`, got `{}`", array.dtype(),
    );
    Ok(is_in(
        item.as_materialized_series(),
        array.as_materialized_series(),
    )?
    .with_name(array.name().clone())
    .into_column())
}

#[cfg(feature = "array_count")]
pub(super) fn count_matches(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let element = &args[1];
    polars_ensure!(
        element.len() == 1,
        ComputeError: "argument expression in `arr.count_matches` must produce exactly one element, got {}",
        element.len()
    );
    let ca = s.array()?;
    ca.array_count_matches(element.get(0).unwrap())
        .map(Column::from)
}

pub(super) fn shift(s: &[Column]) -> PolarsResult<Column> {
    let ca = s[0].array()?;
    let n = &s[1];

    ca.array_shift(n.as_materialized_series()).map(Column::from)
}


#[cfg(test)]
mod test {
    use polars_core::datatypes::Field;
    use polars_core::frame::DataFrame;
    use polars_core::prelude::{Column, Series};
    use super::*;

    #[test]
    fn test_array_f64() {
        println!("\ntest_array_f64");
        let f1 = Series::new("f1".into(), &[1.0, 2.0]);
        let f2 = Series::new("f2".into(), &[3.0, 4.0]);

        let mut cols : Vec<Column> = Vec::new();
        cols.push(Column::Series(f1));
        cols.push(Column::Series(f2));

        let array_df = DataFrame::new(cols.clone()).unwrap();
        println!("input df\n{}\n", &array_df);

        let mut fields: Vec<Field> = Vec::new();
        for col in &cols{
            let f: Field = (col.field().to_mut()).clone();
            fields.push(f);
        }
        let kwargs = crate::dsl::function_expr::array::ArrayKwargs {dtype_expr: "{\"DtypeColumn\":[\"Float64\"]}".to_string()};
        let expected_result = crate::dsl::function_expr::array::array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = crate::dsl::function_expr::array::array_internal(array_df.get_columns(), kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }

    fn i32_series() -> (Vec<Column>, Vec<Field>, DataFrame){
        let f1 = Series::new("f1".into(), &[1, 2]);
        let f2 = Series::new("f2".into(), &[3, 4]);

        let mut cols : Vec<Column> = Vec::new();
        cols.push(Column::Series(f1));
        cols.push(Column::Series(f2));

        let array_df = DataFrame::new(cols.clone()).unwrap();
        println!("input df\n{}\n", &array_df);

        let mut fields: Vec<Field> = Vec::new();
        for col in &cols{
            let f: Field = (col.field().to_mut()).clone();
            fields.push(f);
        }
        (cols, fields, array_df)
    }

    #[test]
    fn test_array_i32() {
        println!("\ntest_array_i32");
        let (_cols, fields, array_df) = i32_series();
        let kwargs = crate::dsl::function_expr::array::ArrayKwargs {dtype_expr: "{\"DtypeColumn\":[\"Int32\"]}".to_string()};
        let expected_result = crate::dsl::function_expr::array::array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = crate::dsl::function_expr::array::array_internal(array_df.get_columns(), kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }

    #[test]
    fn test_array_i32_converted() {
        println!("\ntest_array_i32_converted");
        let (_cols, fields, array_df) = i32_series();
        let kwargs = crate::dsl::function_expr::array::ArrayKwargs {dtype_expr: "{\"DtypeColumn\":[\"Float64\"]}".to_string()};
        let expected_result = crate::dsl::function_expr::array::array_output_type(&fields, kwargs.clone()).unwrap();
        println!("expected result\n{:?}\n", &expected_result);

        let new_arr = crate::dsl::function_expr::array::array_internal(array_df.get_columns(), kwargs);
        println!("actual result\n{:?}", &new_arr);

        assert!(new_arr.is_ok());
        assert_eq!(new_arr.unwrap().dtype(), expected_result.dtype());
    }
}
