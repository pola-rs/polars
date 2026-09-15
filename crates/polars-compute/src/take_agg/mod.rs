#![allow(unsafe_op_in_unsafe_fn)]
//! The kernels that gather a group's elements out of a chunk and reduce them in one pass.

mod binview;
mod boolean;
mod primitive;
mod var;

pub use binview::{
    take_agg_bin_iter_unchecked, take_agg_bin_iter_unchecked_arg,
    take_agg_bin_iter_unchecked_no_null, take_agg_bin_iter_unchecked_no_null_arg,
};
pub use boolean::{
    take_arg_max_bool_iter_unchecked_no_nulls, take_arg_max_bool_iter_unchecked_nulls,
    take_arg_min_bool_iter_unchecked_no_nulls, take_arg_min_bool_iter_unchecked_nulls,
};
pub use primitive::{
    take_agg_no_null_primitive_iter_unchecked, take_agg_primitive_iter_unchecked,
    take_agg_primitive_iter_unchecked_count_nulls,
};
pub use var::{
    online_variance, take_var_no_null_primitive_iter_unchecked,
    take_var_nulls_primitive_iter_unchecked,
};
