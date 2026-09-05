#![allow(unsafe_op_in_unsafe_fn)]
//! The kernels that gather a group's elements out of a chunk and reduce them in one pass.
//!
//! These read the arrays of `polars-array`, whose buffers may stand for a value repeated over
//! every element rather than holding one slot per element. A gather is where that representation
//! pays off most and is easiest to get wrong: the indices a group holds say nothing about the
//! buffer they read, so a buffer of a single slot is read as the one value *every* index gathers,
//! and nothing is written out to make the indices line up.
//!
//! Each kernel branches on the representation once, before it starts walking the indices, so the
//! loop it runs is the same one it would run over a flat chunk.

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
