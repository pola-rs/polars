use polars_arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array,
    UInt64Array,
};
use polars_arrow::bitmap::Bitmap;
use polars_core::frame::column::Column;
use polars_core::prelude::{DataType, Field, IntoColumn, Schema, StringChunked};
use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_plan::dsl::{BaseColumnUdf, Expr, new_column_udf};
use polars_plan::prelude::FunctionOptions;

use crate::compiler::{CastFailure, CompiledProgram, StringContext};
use crate::jit_program::ScalarType;

pub(crate) fn into_expr(
    inputs: Box<[Expr]>,
    execute: impl Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync + 'static,
    state_dtype: impl Fn(&DataType, &[DataType]) -> PolarsResult<DataType> + Send + Sync + 'static,
    input_first: bool,
) -> Expr {
    let function = BaseColumnUdf::new(execute, move |_schema: &Schema, fields: &[Field]| {
        // The output is named after the first input, as Polars names it, and has the state's
        // dtype, which an untyped null initial state takes from the function body.
        let name = fields
            .first()
            .ok_or_else(|| polars_err!(InvalidOperation: "foldv did not receive its inputs"))?
            .name()
            .clone();
        let mut fields = fields.to_vec();
        if input_first {
            fields.swap(0, 1);
        }
        let (initial_state, inputs) = fields.split_first().ok_or_else(
            || polars_err!(InvalidOperation: "foldv did not receive its initial state"),
        )?;
        let initial_dtype = initial_state.dtype.clone().materialize_unknown(false)?;
        let input_dtypes = inputs
            .iter()
            .map(|field| field.dtype.clone())
            .collect::<Vec<_>>();
        Ok(Field::new(
            name,
            state_dtype(&initial_dtype, &input_dtypes)?,
        ))
    });
    // Like `cum_sum`, foldv returns one value per row and folds each group from its own initial
    // state, so `group_by().agg` and `over` apply it per group.
    Expr::AnonymousFunction {
        input: inputs.into_vec(),
        function: new_column_udf(function),
        options: FunctionOptions::length_preserving(),
        fmt_str: Box::new("foldv".into()),
    }
}

pub(crate) fn execute(
    compiled: CompiledProgram,
    initial_state: &Column,
    inputs: &[Column],
) -> PolarsResult<Column> {
    polars_ensure!(initial_state.len() == 1 && !inputs.is_empty(), InvalidOperation:
        "foldv received invalid inputs");
    // Native code reads each buffer as the type it was compiled for, so check them first.
    let state_type = compiled.state_type;
    polars_ensure!(inputs.len() == compiled.input_types.len(), InvalidOperation:
        "foldv program was compiled for {} inputs, but received {}",
        compiled.input_types.len(), inputs.len());
    let input_types = inputs
        .iter()
        .map(|input| ScalarType::from_dtype(input.dtype()))
        .collect::<PolarsResult<Vec<_>>>()?;
    polars_ensure!(*input_types == *compiled.input_types, InvalidOperation:
        "foldv program was compiled for inputs {:?}, but received {input_types:?}",
        compiled.input_types);
    let inputs = inputs.iter().map(Column::rechunk).collect::<Vec<_>>();
    let row_count = inputs[0].len();
    polars_ensure!(inputs.iter().all(|input| input.len() == row_count), ShapeMismatch:
        "foldv inputs must have equal lengths");
    // Owns every string the call creates, including a string state and its output.
    let mut strings = StringContext::new(row_count);
    // An untyped null initial state becomes a null of the type translation inferred for it.
    let initial_state = initial_state.cast(&state_type.dtype())?;
    let (initial_state, initial_valid) =
        initial_state_bytes(&initial_state, state_type, &mut strings)?;
    let all_valid = vec![u8::MAX; row_count.saturating_add(7) / 8];
    // Generated code reads strings as `(pointer, length)` pairs into the inputs' own buffers.
    let decoded_strings = inputs
        .iter()
        .map(decode_strings)
        .collect::<PolarsResult<Vec<_>>>()?;
    let views = inputs
        .iter()
        .zip(&decoded_strings)
        .map(|(input, decoded)| {
            RuntimeInput::from_column(input, decoded.as_deref(), all_valid.as_ptr())
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let pointers = views.iter().map(|input| input.values).collect::<Vec<_>>();
    let value_bit_offsets = views
        .iter()
        .map(|input| input.value_bit_offset)
        .collect::<Vec<_>>();
    let validity_pointers = views.iter().map(|input| input.validity).collect::<Vec<_>>();
    let validity_bit_offsets = views
        .iter()
        .map(|input| input.validity_bit_offset)
        .collect::<Vec<_>>();
    // A string state's output is collected by `strings` instead of this buffer.
    let output_width = if state_type == ScalarType::String {
        0
    } else {
        state_type.byte_width()
    };
    let mut output = vec![0_u8; row_count * output_width];
    let mut output_validity = vec![0_u8; row_count];
    let mut cast_failure = CastFailure::default();
    // SAFETY: Value pointers refer to row_count values of the dtype used during compilation;
    // bitmap pointers and offsets cover row_count bits. `inputs`, `decoded_strings`, and
    // `all_valid` own those buffers until after the call, the state and output buffers have the
    // sizes required by the compiled state dtype, and `strings` holds the initial string state.
    let processed_rows = unsafe {
        (compiled.function)(
            pointers.as_ptr(),
            value_bit_offsets.as_ptr(),
            validity_pointers.as_ptr(),
            validity_bit_offsets.as_ptr(),
            row_count,
            initial_state.as_ptr(),
            initial_valid,
            output.as_mut_ptr(),
            output_validity.as_mut_ptr(),
            &mut cast_failure,
            &mut strings,
        )
    };
    if processed_rows < row_count {
        return Err(strict_cast_error(&compiled, &cast_failure, processed_rows));
    }
    if state_type == ScalarType::String {
        let name = inputs[0].name().clone();
        return Ok(StringChunked::with_chunk(name, strings.into_output()).into_column());
    }
    output_column(
        inputs[0].name().clone(),
        state_type,
        &output,
        &output_validity,
    )
}

struct RuntimeInput {
    values: *const u8,
    value_bit_offset: usize,
    validity: *const u8,
    validity_bit_offset: usize,
}

impl RuntimeInput {
    fn from_column(
        column: &Column,
        decoded_strings: Option<&[[usize; 2]]>,
        all_valid: *const u8,
    ) -> PolarsResult<Self> {
        let array = &**column.as_materialized_series().chunks().first().unwrap();
        let (values, value_bit_offset) = match ScalarType::from_dtype(column.dtype())? {
            ScalarType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .values();
                let (bytes, offset, _) = values.as_slice();
                (bytes.as_ptr(), offset)
            },
            ScalarType::Int32 => primitive_parts::<Int32Array>(array),
            ScalarType::Int64 => primitive_parts::<Int64Array>(array),
            ScalarType::UInt32 => primitive_parts::<UInt32Array>(array),
            ScalarType::UInt64 => primitive_parts::<UInt64Array>(array),
            ScalarType::Float32 => primitive_parts::<Float32Array>(array),
            ScalarType::Float64 => primitive_parts::<Float64Array>(array),
            ScalarType::String => (
                decoded_strings
                    .expect("string inputs are decoded")
                    .as_ptr()
                    .cast(),
                0,
            ),
        };
        let (validity, validity_bit_offset) =
            array.validity().map(bitmap_parts).unwrap_or((all_valid, 0));
        Ok(Self {
            values,
            value_bit_offset,
            validity,
            validity_bit_offset,
        })
    }
}

fn primitive_parts<A>(array: &dyn Array) -> (*const u8, usize)
where
    A: Array + PrimitiveValuesPointer,
    A: 'static,
{
    let array = array.as_any().downcast_ref::<A>().unwrap();
    // Primitive array types all expose their values through the Array trait's buffers, but the
    // concrete pointer is obtained by the specialized helper below.
    primitive_values_pointer(array)
}

trait PrimitiveValuesPointer {
    fn values_pointer(&self) -> *const u8;
}

macro_rules! impl_primitive_values_pointer {
    ($($array:ty),+ $(,)?) => {$(
        impl PrimitiveValuesPointer for $array {
            fn values_pointer(&self) -> *const u8 {
                self.values().as_ptr().cast()
            }
        }
    )+};
}

impl_primitive_values_pointer!(
    Int32Array,
    Int64Array,
    UInt32Array,
    UInt64Array,
    Float32Array,
    Float64Array,
);

fn primitive_values_pointer<A: PrimitiveValuesPointer>(array: &A) -> (*const u8, usize) {
    (array.values_pointer(), 0)
}

/// For a string column, each row's `(pointer, length)` into the column's buffers; a null row is
/// `(0, 0)`, which generated code treats as an empty string.
fn decode_strings(column: &Column) -> PolarsResult<Option<Vec<[usize; 2]>>> {
    if column.dtype() != &DataType::String {
        return Ok(None);
    }
    Ok(Some(
        column
            .str()?
            .iter()
            .map(|value| value.map_or([0, 0], |value| [value.as_ptr() as usize, value.len()]))
            .collect(),
    ))
}

fn bitmap_parts(bitmap: &Bitmap) -> (*const u8, usize) {
    let (bytes, offset, _) = bitmap.as_slice();
    (bytes.as_ptr(), offset)
}

/// Describe a failed strict cast like Polars' own strict cast error.
fn strict_cast_error(
    compiled: &CompiledProgram,
    failure: &CastFailure,
    row: usize,
) -> polars_error::PolarsError {
    let (source, target) = compiled.strict_casts[failure.cast as usize];
    let bits = failure.value;
    let value = match source {
        ScalarType::Boolean => (bits != 0).to_string(),
        ScalarType::Int32 => (bits as u32 as i32).to_string(),
        ScalarType::Int64 => (bits as i64).to_string(),
        ScalarType::UInt32 => (bits as u32).to_string(),
        ScalarType::UInt64 => bits.to_string(),
        ScalarType::Float32 => f32::from_bits(bits as u32).to_string(),
        ScalarType::Float64 => f64::from_bits(bits).to_string(),
        ScalarType::String => unreachable!("translation rejects casts from strings"),
    };
    polars_err!(InvalidOperation:
        "conversion from `{}` to `{}` failed in foldv at row {row}: [{value}]; \
        use `strict=False` to produce null instead",
        source.dtype(), target.dtype())
}

/// The initial state's words, laid out as generated code loads them, and its validity. A string
/// is copied into `strings` and passed as a `(pointer, length)` pair; a null state is all zeros.
fn initial_state_bytes(
    column: &Column,
    dtype: ScalarType,
    strings: &mut StringContext,
) -> PolarsResult<([u8; 16], u8)> {
    let mut bytes = [0_u8; 16];
    if column.get(0)?.is_null() {
        return Ok((bytes, 0));
    }
    match dtype {
        ScalarType::Boolean => bytes[0] = u8::from(required(column.bool()?.get(0))?),
        ScalarType::Int32 => {
            bytes[..4].copy_from_slice(&required(column.i32()?.get(0))?.to_ne_bytes())
        },
        ScalarType::Int64 => {
            bytes[..8].copy_from_slice(&required(column.i64()?.get(0))?.to_ne_bytes())
        },
        ScalarType::UInt32 => {
            bytes[..4].copy_from_slice(&required(column.u32()?.get(0))?.to_ne_bytes())
        },
        ScalarType::UInt64 => {
            bytes[..8].copy_from_slice(&required(column.u64()?.get(0))?.to_ne_bytes())
        },
        ScalarType::Float32 => {
            let value = required(column.f32()?.get(0))?;
            bytes[..4].copy_from_slice(&value.to_bits().to_ne_bytes());
        },
        ScalarType::Float64 => {
            let value = required(column.f64()?.get(0))?;
            bytes[..8].copy_from_slice(&value.to_bits().to_ne_bytes());
        },
        ScalarType::String => {
            let value = required(column.str()?.get(0))?;
            let pointer = strings.set_initial_state(value);
            bytes[..8].copy_from_slice(&(pointer as usize).to_ne_bytes());
            bytes[8..].copy_from_slice(&value.len().to_ne_bytes());
        },
    }
    Ok((bytes, 1))
}

fn required<T>(value: Option<T>) -> PolarsResult<T> {
    value.ok_or_else(|| polars_err!(InvalidOperation: "foldv currently requires non-null values"))
}

fn output_column(
    name: polars_utils::pl_str::PlSmallStr,
    dtype: ScalarType,
    bytes: &[u8],
    validity: &[u8],
) -> PolarsResult<Column> {
    Ok(match dtype {
        ScalarType::Boolean => Column::new(
            name,
            bytes
                .iter()
                .zip(validity)
                .map(|(value, valid)| (*valid != 0).then_some(*value != 0))
                .collect::<Vec<_>>(),
        ),
        ScalarType::Int32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0).then(|| i32::from_ne_bytes(value.try_into().unwrap()))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::Int64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0).then(|| i64::from_ne_bytes(value.try_into().unwrap()))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::UInt32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0).then(|| u32::from_ne_bytes(value.try_into().unwrap()))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::UInt64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0).then(|| u64::from_ne_bytes(value.try_into().unwrap()))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::Float32 => Column::new(
            name,
            bytes
                .chunks_exact(4)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0)
                        .then(|| f32::from_bits(u32::from_ne_bytes(value.try_into().unwrap())))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::Float64 => Column::new(
            name,
            bytes
                .chunks_exact(8)
                .zip(validity)
                .map(|(value, valid)| {
                    (*valid != 0)
                        .then(|| f64::from_bits(u64::from_ne_bytes(value.try_into().unwrap())))
                })
                .collect::<Vec<_>>(),
        ),
        ScalarType::String => unreachable!("a string state's output is collected separately"),
    })
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{Column, DataType};
    use polars_plan::dsl::Expr;
    use polars_plan::prelude::col;

    use crate::{compiler, translate};

    #[test]
    fn execute_rejects_a_different_number_of_inputs_than_compiled() {
        for count in [1, 3] {
            let function = Expr::Element + col("a") + col("b");
            let names = ["a".into(), "b".into()];
            let state = Column::new("s".into(), [0_i64]);
            let compiled_for = [
                Column::new("a".into(), [1_i64, 2]),
                Column::new("b".into(), [3_i64, 4]),
            ];
            let translated =
                translate::translate(&function, &names, &state, &compiled_for).unwrap();
            let compiled = compiler::compile(
                &translated.program,
                compiler::CompilationShape::Lists(translated.bindings),
            )
            .unwrap();
            let other = vec![compiled_for[0].clone(); count];
            let error = super::execute(compiled, &state, &other).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("compiled for 2 inputs, but received {count}")),
                "{error}"
            );
        }
    }

    #[test]
    fn execute_rejects_inputs_of_another_type_than_compiled() {
        // A program compiled for an Int64 input must not read an Int32 buffer.
        let function = Expr::Element + col("l");
        let names = ["l".into()];
        let state = Column::new("s".into(), [0_i64]);
        let compiled_for = [Column::new("l".into(), [1_i64, 2])];
        let translated = translate::translate(&function, &names, &state, &compiled_for).unwrap();
        let compiled = compiler::compile(
            &translated.program,
            compiler::CompilationShape::Lists(translated.bindings),
        )
        .unwrap();

        let other = [Column::new("l".into(), [1_i32, 2])];
        assert_eq!(other[0].dtype(), &DataType::Int32);
        let error = super::execute(compiled, &state, &other).unwrap_err();
        assert!(error.to_string().contains("compiled for inputs"), "{error}");
    }
}
