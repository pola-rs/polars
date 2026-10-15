use polars_core::frame::column::Column;
use polars_core::prelude::{
    Field, Float32Chunked, Float64Chunked, Int32Chunked, Int64Chunked, UInt32Chunked, UInt64Chunked,
};
use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_plan::dsl::Expr;
use polars_plan::prelude::apply_multiple;

use crate::compiler::CompiledProgram;
use crate::jit_program::ScalarType;

pub(crate) fn into_expr(
    inputs: Box<[Expr]>,
    execute: impl Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync + 'static,
    input_first: bool,
) -> Expr {
    apply_multiple(
        execute,
        inputs.into_vec(),
        move |_schema, fields| {
            // The output is named after the first input, as Polars names it, and has the state's
            // dtype. A method-style fold passes its input before the state.
            let name = fields
                .first()
                .ok_or_else(|| polars_err!(InvalidOperation: "foldv did not receive its inputs"))?
                .name()
                .clone();
            let initial_state = &fields[usize::from(input_first)];
            Ok(Field::new(
                name,
                initial_state.dtype.clone().materialize_unknown(false)?,
            ))
        },
        true,
    )
}

pub(crate) fn execute(
    compiled: CompiledProgram,
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<Column> {
    polars_ensure!(initial_state.len() == 1 && !row_inputs.is_empty(), InvalidOperation:
        "foldv received invalid inputs");
    let state_type = ScalarType::from_dtype(initial_state.dtype())?;
    let initial_state = initial_state_bytes(initial_state, state_type)?;
    let inputs = row_inputs
        .iter()
        .map(RuntimeInput::from_column)
        .collect::<PolarsResult<Vec<_>>>()?;
    let row_count = inputs[0].len();
    polars_ensure!(inputs.iter().all(|input| input.len() == row_count), ShapeMismatch:
        "foldv row inputs must have equal lengths");
    let pointers = inputs
        .iter()
        .map(RuntimeInput::as_ptr)
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut output = vec![0_u8; row_count * state_type.byte_width()];
    call(&compiled, &pointers, row_count, &initial_state, &mut output);
    output_column(row_inputs[0].name().clone(), state_type, &output)
}

enum RuntimeInput {
    Boolean(Vec<u8>),
    Int32(Int32Chunked),
    Int64(Int64Chunked),
    UInt32(UInt32Chunked),
    UInt64(UInt64Chunked),
    Float32(Float32Chunked),
    Float64(Float64Chunked),
}

impl RuntimeInput {
    fn from_column(column: &Column) -> PolarsResult<Self> {
        Ok(match ScalarType::from_dtype(column.dtype())? {
            ScalarType::Boolean => {
                // Arrow Boolean arrays are bit-packed, while the JIT Boolean ABI uses one byte.
                Self::Boolean(
                    column
                        .bool()?
                        .iter()
                        .map(|value| value.map(u8::from))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            polars_err!(InvalidOperation:
                                "foldv currently requires non-null values")
                        })?,
                )
            },
            ScalarType::Int32 => Self::Int32(column.i32()?.rechunk().into_owned()),
            ScalarType::Int64 => Self::Int64(column.i64()?.rechunk().into_owned()),
            ScalarType::UInt32 => Self::UInt32(column.u32()?.rechunk().into_owned()),
            ScalarType::UInt64 => Self::UInt64(column.u64()?.rechunk().into_owned()),
            ScalarType::Float32 => {
                let values = column.f32()?.rechunk();
                polars_ensure!(!values.cont_slice()?.iter().any(|value| value.is_nan()), InvalidOperation:
                    "foldv does not support NaN values yet");
                Self::Float32(values.into_owned())
            },
            ScalarType::Float64 => {
                let values = column.f64()?.rechunk();
                polars_ensure!(!values.cont_slice()?.iter().any(|value| value.is_nan()), InvalidOperation:
                    "foldv does not support NaN values yet");
                Self::Float64(values.into_owned())
            },
        })
    }

    fn as_ptr(&self) -> PolarsResult<*const u8> {
        Ok(match self {
            Self::Boolean(values) => values.as_ptr(),
            Self::Int32(values) => values.cont_slice()?.as_ptr().cast(),
            Self::Int64(values) => values.cont_slice()?.as_ptr().cast(),
            Self::UInt32(values) => values.cont_slice()?.as_ptr().cast(),
            Self::UInt64(values) => values.cont_slice()?.as_ptr().cast(),
            Self::Float32(values) => values.cont_slice()?.as_ptr().cast(),
            Self::Float64(values) => values.cont_slice()?.as_ptr().cast(),
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Boolean(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::UInt32(values) => values.len(),
            Self::UInt64(values) => values.len(),
            Self::Float32(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }
}

fn call(
    compiled: &CompiledProgram,
    pointers: &[*const u8],
    row_count: usize,
    initial_state: &[u8],
    output: &mut [u8],
) {
    // SAFETY: Each pointer refers to row_count values of the dtype used during compilation.
    // The state and output buffers contain at least the compiled state dtype's byte width.
    unsafe {
        (compiled.function)(
            pointers.as_ptr(),
            row_count,
            initial_state.as_ptr(),
            output.as_mut_ptr(),
        )
    }
}

fn initial_state_bytes(column: &Column, dtype: ScalarType) -> PolarsResult<[u8; 8]> {
    let mut bytes = [0_u8; 8];
    match dtype {
        ScalarType::Boolean => bytes[0] = u8::from(required(column.bool()?.get(0))?),
        ScalarType::Int32 => {
            bytes[..4].copy_from_slice(&required(column.i32()?.get(0))?.to_ne_bytes())
        },
        ScalarType::Int64 => bytes.copy_from_slice(&required(column.i64()?.get(0))?.to_ne_bytes()),
        ScalarType::UInt32 => {
            bytes[..4].copy_from_slice(&required(column.u32()?.get(0))?.to_ne_bytes())
        },
        ScalarType::UInt64 => bytes.copy_from_slice(&required(column.u64()?.get(0))?.to_ne_bytes()),
        ScalarType::Float32 => {
            let value = required(column.f32()?.get(0))?;
            polars_ensure!(!value.is_nan(), InvalidOperation: "foldv does not support NaN values yet");
            bytes[..4].copy_from_slice(&value.to_bits().to_ne_bytes());
        },
        ScalarType::Float64 => {
            let value = required(column.f64()?.get(0))?;
            polars_ensure!(!value.is_nan(), InvalidOperation: "foldv does not support NaN values yet");
            bytes.copy_from_slice(&value.to_bits().to_ne_bytes());
        },
    }
    Ok(bytes)
}

fn required<T>(value: Option<T>) -> PolarsResult<T> {
    value.ok_or_else(|| polars_err!(InvalidOperation: "foldv currently requires non-null values"))
}

fn output_column(
    name: polars_utils::pl_str::PlSmallStr,
    dtype: ScalarType,
    bytes: &[u8],
) -> PolarsResult<Column> {
    Ok(match dtype {
        ScalarType::Boolean => Column::new(
            name,
            bytes.iter().map(|value| *value != 0).collect::<Vec<_>>(),
        ),
        ScalarType::Int32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .map(|value| i32::from_ne_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ),
        ScalarType::Int64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .map(|value| i64::from_ne_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ),
        ScalarType::UInt32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .map(|value| u32::from_ne_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ),
        ScalarType::UInt64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .map(|value| u64::from_ne_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ),
        ScalarType::Float32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .map(|value| f32::from_bits(u32::from_ne_bytes(value.try_into().unwrap())))
                .collect::<Vec<_>>(),
        ),
        ScalarType::Float64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .map(|value| f64::from_bits(u64::from_ne_bytes(value.try_into().unwrap())))
                .collect::<Vec<_>>(),
        ),
    })
}
