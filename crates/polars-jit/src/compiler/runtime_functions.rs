//! Runtime helper implementations, native ABI signatures, and JIT symbol registration.
//!
//! Generated code calls the `extern "C"` helpers below. They must not unwind across native frames.

use cranelift_codegen::ir::{AbiParam, Signature, types};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use polars_arrow::array::{MutablePlString, Utf8ViewArray};
use polars_error::PolarsResult;

use super::helpers::compile_error;

/// A runtime function. Its signature here must match the Rust definition.
#[derive(Clone, Copy)]
pub(super) enum RuntimeFunction {
    Concat,
    Compare,
    LenChars,
    EndStringRow,
    EndRow,
}

impl RuntimeFunction {
    pub(super) const ALL: [Self; 5] = [
        Self::Concat,
        Self::Compare,
        Self::LenChars,
        Self::EndStringRow,
        Self::EndRow,
    ];

    /// Make every runtime function's address known to the JIT under its name.
    pub(super) fn register_all(jit_builder: &mut JITBuilder) {
        for function in Self::ALL {
            jit_builder.symbol(function.name(), function.address());
        }
    }

    /// Declare every runtime function in the module, indexed like `ALL`.
    pub(super) fn declare_all(module: &mut JITModule) -> PolarsResult<Vec<FuncId>> {
        Self::ALL
            .into_iter()
            .map(|function| {
                module
                    .declare_function(
                        function.name(),
                        Linkage::Import,
                        &function.signature(module),
                    )
                    .map_err(compile_error)
            })
            .collect()
    }

    fn name(self) -> &'static str {
        match self {
            Self::Concat => "polars_jit_string_concat",
            Self::Compare => "polars_jit_string_compare",
            Self::LenChars => "polars_jit_string_len_chars",
            Self::EndStringRow => "polars_jit_end_string_row",
            Self::EndRow => "polars_jit_end_row",
        }
    }

    fn address(self) -> *const u8 {
        match self {
            Self::Concat => concat as *const u8,
            Self::Compare => compare as *const u8,
            Self::LenChars => len_chars as *const u8,
            Self::EndStringRow => end_string_row as *const u8,
            Self::EndRow => end_row as *const u8,
        }
    }

    fn signature(self, module: &JITModule) -> Signature {
        let word = AbiParam::new(types::I64);
        let mut signature = module.make_signature();
        let (params, returns) = match self {
            Self::Concat => (vec![word; 5], vec![word]),
            Self::Compare => (vec![word; 4], vec![AbiParam::new(types::I32)]),
            Self::LenChars => (vec![word; 2], vec![AbiParam::new(types::I32)]),
            Self::EndStringRow => (vec![word, word, word, AbiParam::new(types::I8)], vec![word]),
            Self::EndRow => (vec![word], vec![]),
        };
        signature.params = params;
        signature.returns = returns;
        signature
    }
}

/// Owns string state, per-row scratch storage, and output while generated code executes.
pub(crate) struct StringContext {
    /// The carried string state. Two buffers let a row build its new state from the old one; the
    /// state always points into `states[active]` after a row ends.
    states: [Vec<u8>; 2],
    active: usize,
    /// Strings created while evaluating the current row, freed when the row ends.
    scratch: Vec<Vec<u8>>,
    /// Every row's state, when the state is a string.
    output: MutablePlString,
}

impl StringContext {
    pub(crate) fn new(row_count: usize) -> Self {
        Self {
            states: [Vec::new(), Vec::new()],
            active: 0,
            scratch: Vec::new(),
            output: MutablePlString::with_capacity(row_count),
        }
    }

    /// Store the initial state and return the pointer generated code starts from.
    pub(crate) fn set_initial_state(&mut self, value: &str) -> *const u8 {
        self.states[self.active].extend_from_slice(value.as_bytes());
        self.states[self.active].as_ptr()
    }

    /// Every row's state, once generated code has processed all rows.
    pub(crate) fn into_output(self) -> Utf8ViewArray {
        self.output.freeze()
    }
}

/// The bytes of a string held by generated code.
///
/// # Safety
/// `ptr` must point to `len` readable bytes that outlive `'a`, or `len` must be zero.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        // SAFETY: Guaranteed by the caller.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Concatenate two strings into the current row's scratch storage and return the result's
/// pointer; its length is the sum of both lengths.
///
/// # Safety
/// `context` must be the call's live context and both strings must satisfy [`bytes`].
unsafe extern "C" fn concat(
    context: *mut StringContext,
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> *const u8 {
    // SAFETY: Guaranteed by the caller.
    let (context, left, right) = unsafe {
        (
            &mut *context,
            bytes(left, left_len),
            bytes(right, right_len),
        )
    };
    let mut joined = Vec::with_capacity(left.len() + right.len());
    joined.extend_from_slice(left);
    joined.extend_from_slice(right);
    // Moving the vector into `scratch` does not move its heap buffer, so the pointer stays valid
    // until the row ends.
    let pointer = joined.as_ptr();
    context.scratch.push(joined);
    pointer
}

/// Compare two strings by their UTF-8 bytes, which orders them like Polars: `-1`, `0`, or `1`.
///
/// # Safety
/// Both strings must satisfy [`bytes`].
unsafe extern "C" fn compare(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> i32 {
    // SAFETY: Guaranteed by the caller.
    let (left, right) = unsafe { (bytes(left, left_len), bytes(right, right_len)) };
    left.cmp(right) as i32
}

/// The number of Unicode scalar values in a string, like `str.len_chars()`.
///
/// # Safety
/// The string must satisfy [`bytes`] and be valid UTF-8.
unsafe extern "C" fn len_chars(value: *const u8, len: usize) -> u32 {
    // SAFETY: Guaranteed by the caller.
    let value = unsafe { bytes(value, len) };
    // Every character starts with exactly one byte that is not a continuation byte.
    value.iter().filter(|byte| (**byte as i8) >= -0x40).count() as u32
}

/// Finish a row whose state is a string: append the state to the output, keep it for the next
/// row, and free the row's scratch strings. Returns the pointer to the kept state; a null state
/// keeps no bytes, so generated code continues with a length of zero.
///
/// # Safety
/// `context` must be the call's live context, and the state must satisfy [`bytes`] and be valid
/// UTF-8. It may point into scratch storage, the active state buffer, an input, or a literal.
unsafe extern "C" fn end_string_row(
    context: *mut StringContext,
    state: *const u8,
    len: usize,
    valid: u8,
) -> *const u8 {
    // SAFETY: Guaranteed by the caller.
    let (context, state) = unsafe { (&mut *context, bytes(state, len)) };
    if valid != 0 {
        // SAFETY: The caller guarantees valid UTF-8.
        context
            .output
            .push_value(unsafe { std::str::from_utf8_unchecked(state) });
        let active = &context.states[context.active];
        // An unchanged state already lives in the active buffer and needs no copy.
        if !(active.as_ptr() == state.as_ptr() && active.len() == state.len()) {
            let next = 1 - context.active;
            context.states[next].clear();
            context.states[next].extend_from_slice(state);
            context.active = next;
        }
    } else {
        context.output.push_null();
    }
    context.scratch.clear();
    context.states[context.active].as_ptr()
}

/// Free the row's scratch strings when the state is not a string.
///
/// # Safety
/// `context` must be the call's live context.
unsafe extern "C" fn end_row(context: *mut StringContext) {
    // SAFETY: Guaranteed by the caller.
    unsafe { (*context).scratch.clear() };
}
