use polars_core::prelude::*;

use crate::chunked_array::list::SetOperation;

pub fn array_set_operation(
    a: &ArrayChunked,
    b: &ArrayChunked,
    set_op: SetOperation,
) -> PolarsResult<ListChunked> {
    // Convert both ArrayChunked to ListChunked and delegate to list implementation
    let a_list = a.cast(&DataType::List(Box::new(a.inner_dtype().clone())))?;
    let b_list = b.cast(&DataType::List(Box::new(b.inner_dtype().clone())))?;

    let a_list = a_list.list()?.clone();
    let b_list = b_list.list()?.clone();

    crate::chunked_array::list::list_set_operation(&a_list, &b_list, set_op)
}
