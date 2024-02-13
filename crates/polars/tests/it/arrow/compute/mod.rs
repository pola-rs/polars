#[cfg(feature = "compute_aggregate")]
mod aggregate;
#[cfg(feature = "compute_bitwise")]
mod bitwise;
#[cfg(feature = "compute_boolean")]
mod boolean;
#[cfg(feature = "compute_boolean_kleene")]
mod boolean_kleene;
#[cfg(feature = "compute_if_then_else")]
mod if_then_else;

mod arity_assign;
