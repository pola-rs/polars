#[cfg(feature = "compute_aggregate")]
mod aggregate;
#[cfg(feature = "compute_arithmetics")]
mod arithmetics;
#[cfg(feature = "compute_bitwise")]
mod bitwise;
#[cfg(feature = "compute_boolean")]
mod boolean;
#[cfg(feature = "compute_boolean_kleene")]
mod boolean_kleene;
#[cfg(feature = "compute_cast")]
mod cast;
#[cfg(feature = "compute_comparison")]
mod comparison;
#[cfg(feature = "compute_concatenate")]
mod concatenate;
#[cfg(feature = "compute_contains")]
mod contains;
#[cfg(feature = "compute_filter")]
mod filter;
#[cfg(feature = "compute_hash")]
mod hash;
#[cfg(feature = "compute_if_then_else")]
mod if_then_else;
#[cfg(feature = "compute_length")]
mod length;
#[cfg(feature = "compute_like")]
mod like;
#[cfg(feature = "compute_limit")]
mod limit;
#[cfg(feature = "compute_merge_sort")]
mod merge_sort;
#[cfg(feature = "compute_partition")]
mod partition;
#[cfg(feature = "compute_regex_match")]
mod regex_match;
#[cfg(feature = "compute_sort")]
mod sort;
#[cfg(feature = "compute_substring")]
mod substring;
#[cfg(feature = "compute_take")]
mod take;
#[cfg(feature = "compute_temporal")]
mod temporal;
#[cfg(feature = "compute_utf8")]
mod utf8;
#[cfg(feature = "compute_window")]
mod window;

mod arity_assign;
