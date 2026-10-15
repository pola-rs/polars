//! How generated code holds values: one machine word and a validity byte.

use cranelift_codegen::ir::Value;

/// A value and its validity: `1` when it is valid and `0` when it is null.
#[derive(Clone, Copy)]
pub(super) struct JitValue {
    pub(super) data: Value,
    pub(super) valid: Value,
}
