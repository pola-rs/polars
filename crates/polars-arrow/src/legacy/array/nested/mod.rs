mod list;
mod struct_;

use std::any::Any;

use polars_error::{polars_bail, PolarsResult};

use crate::array::MutableArray;
use crate::datatypes::{ArrowDataType, PhysicalType};

pub fn dyn_array_push<T>(
    arr_dyn: &mut dyn MutableArray,
    value_dyn: Option<&T>,
    dtype: &ArrowDataType,
) -> PolarsResult<()>
where
    T: Any,
{
    match value_dyn {
        Some(value) => dyn_array_push_value(arr_dyn, value, dtype)?,
        None => arr_dyn.push_null(),
    };
    Ok(())
}

fn dyn_array_push_value<T>(
    arr_dyn: &mut dyn MutableArray,
    value_dyn: &T,
    dtype: &ArrowDataType,
) -> PolarsResult<()>
where
    T: Any,
{
    match dtype.to_physical_type() {
        PhysicalType::List => {
            list::dyn_list_array_push::<i32, T>(arr_dyn, value_dyn)?;
        },
        PhysicalType::LargeList => {
            list::dyn_list_array_push::<i64, T>(arr_dyn, value_dyn)?;
        },
        PhysicalType::Struct => {
            struct_::dyn_struct_array_push(arr_dyn, value_dyn)?;
        },
        _ => {
            polars_bail!(nyi = "cannot push dynamic value of type {dtype:?}")
        },
    };
    Ok(())
}
