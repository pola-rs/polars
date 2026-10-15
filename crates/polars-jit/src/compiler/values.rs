//! How generated code holds values: one machine word for a primitive, or a pointer and a byte
//! length for a string, plus a validity byte. Values cross block boundaries as block parameters.

use cranelift_codegen::ir::{Block, BlockArg, InstBuilder, MemFlagsData, Type, Value, types};
use cranelift_frontend::FunctionBuilder;

use super::helpers::cranelift_type;
use crate::jit_program::ScalarType;

/// A value's machine words.
#[derive(Clone, Copy)]
pub(super) enum JitData {
    /// A primitive held in one word.
    Scalar(Value),
    /// A UTF-8 string: a pointer and a byte length, owned by inputs or the call's StringContext.
    Str { ptr: Value, len: Value },
}

impl JitData {
    /// The single word of a primitive. Translation only gives primitives to the operations that
    /// call this, so any other value is a compiler bug.
    pub(super) fn scalar(self) -> Value {
        match self {
            Self::Scalar(value) => value,
            Self::Str { .. } => unreachable!("translation only gives primitives to this operation"),
        }
    }

    /// The pointer and byte length of a string. Translation only gives strings to the operations
    /// that call this, so any other value is a compiler bug.
    pub(super) fn string(self) -> (Value, Value) {
        match self {
            Self::Str { ptr, len } => (ptr, len),
            Self::Scalar(_) => unreachable!("translation only gives strings to this operation"),
        }
    }

    fn words(self) -> Vec<Value> {
        match self {
            Self::Scalar(value) => vec![value],
            Self::Str { ptr, len } => vec![ptr, len],
        }
    }
}

/// A value and its validity: `1` when it is valid and `0` when it is null.
#[derive(Clone, Copy)]
pub(super) struct JitValue {
    pub(super) data: JitData,
    pub(super) valid: Value,
}

/// The machine types of the words that hold a value of `dtype`.
fn value_types(dtype: ScalarType) -> Vec<Type> {
    match dtype {
        ScalarType::String => vec![types::I64, types::I64],
        dtype => vec![cranelift_type(dtype)],
    }
}

/// Add block parameters for one value: its words followed by its validity.
pub(super) fn append_value_params(
    builder: &mut FunctionBuilder<'_>,
    block: Block,
    dtype: ScalarType,
) {
    for word in value_types(dtype) {
        builder.append_block_param(block, word);
    }
    builder.append_block_param(block, types::I8);
}

/// Block arguments for one value, matching `append_value_params`.
pub(super) fn value_args(value: JitValue) -> Vec<BlockArg> {
    let mut arguments = value
        .data
        .words()
        .into_iter()
        .map(BlockArg::from)
        .collect::<Vec<_>>();
    arguments.push(BlockArg::from(value.valid));
    arguments
}

/// Read one value back from block parameters added by `append_value_params`.
pub(super) fn value_from_params(params: &[Value], dtype: ScalarType) -> JitValue {
    let words = value_types(dtype).len();
    JitValue {
        data: data_from_words(dtype, &params[..words]),
        valid: params[words],
    }
}

fn data_from_words(dtype: ScalarType, words: &[Value]) -> JitData {
    match dtype {
        ScalarType::String => JitData::Str {
            ptr: words[0],
            len: words[1],
        },
        _ => JitData::Scalar(words[0]),
    }
}

/// Load a value's words from consecutive pointer-sized slots at `address`.
pub(super) fn load_words(
    builder: &mut FunctionBuilder<'_>,
    dtype: ScalarType,
    address: Value,
) -> JitData {
    let words = value_types(dtype)
        .into_iter()
        .enumerate()
        .map(|(slot, word)| {
            builder
                .ins()
                .load(word, MemFlagsData::new(), address, (slot * 8) as i32)
        })
        .collect::<Vec<_>>();
    data_from_words(dtype, &words)
}
