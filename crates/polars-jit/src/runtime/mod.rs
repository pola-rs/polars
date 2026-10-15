use polars_arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array,
    UInt64Array,
};
use polars_arrow::bitmap::Bitmap;
use polars_core::frame::column::Column;
use polars_core::prelude::{DataType, Field};
use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_plan::dsl::Expr;
use polars_plan::prelude::apply_multiple;

use crate::compiler::{CastFailure, CompiledProgram};
use crate::jit_program::ScalarType;

pub(crate) fn into_expr(
    inputs: Box<[Expr]>,
    execute: impl Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync + 'static,
    state_dtype: impl Fn(&DataType, &[DataType]) -> PolarsResult<DataType> + Send + Sync + 'static,
    input_first: bool,
) -> Expr {
    apply_multiple(
        execute,
        inputs.into_vec(),
        move |_schema, fields| {
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
            let (initial_state, row_inputs) = fields.split_first().ok_or_else(
                || polars_err!(InvalidOperation: "foldv did not receive its initial state"),
            )?;
            let initial_dtype = initial_state.dtype.clone().materialize_unknown(false)?;
            let row_dtypes = row_inputs
                .iter()
                .map(|field| field.dtype.clone())
                .collect::<Vec<_>>();
            Ok(Field::new(name, state_dtype(&initial_dtype, &row_dtypes)?))
        },
        true,
    )
}

pub(crate) fn execute(
    compiled: CompiledProgram,
    state_type: ScalarType,
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<Column> {
    polars_ensure!(initial_state.len() == 1 && !row_inputs.is_empty(), InvalidOperation:
        "foldv received invalid inputs");
    // An untyped null initial state becomes a null of the type translation inferred for it.
    let initial_state = initial_state.cast(&state_type.dtype())?;
    let (initial_state, initial_valid) = initial_state_bytes(&initial_state, state_type)?;
    let inputs = row_inputs.iter().map(Column::rechunk).collect::<Vec<_>>();
    let row_count = inputs[0].len();
    polars_ensure!(inputs.iter().all(|input| input.len() == row_count), ShapeMismatch:
        "foldv row inputs must have equal lengths");
    let all_valid = vec![u8::MAX; row_count.saturating_add(7) / 8];
    let views = inputs
        .iter()
        .map(|input| RuntimeInput::from_column(input, all_valid.as_ptr()))
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
    let mut output = vec![0_u8; row_count * state_type.byte_width()];
    let mut output_validity = vec![0_u8; row_count];
    let mut cast_failure = CastFailure::default();
    // SAFETY: Value pointers refer to row_count values of the dtype used during compilation;
    // bitmap pointers and offsets cover row_count bits. `inputs` and `all_valid` own those
    // buffers until after the call, and the state and output buffers have the sizes required
    // by the compiled state dtype.
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
        )
    };
    if processed_rows < row_count {
        return Err(strict_cast_error(&compiled, &cast_failure, processed_rows));
    }
    output_column(
        row_inputs[0].name().clone(),
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
    fn from_column(column: &Column, all_valid: *const u8) -> PolarsResult<Self> {
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
    };
    polars_err!(InvalidOperation:
        "conversion from `{}` to `{}` failed in foldv at row {row}: [{value}]; \
        use `strict=False` to produce null instead",
        source.dtype(), target.dtype())
}

fn initial_state_bytes(column: &Column, dtype: ScalarType) -> PolarsResult<([u8; 8], u8)> {
    let mut bytes = [0_u8; 8];
    if column.get(0)?.is_null() {
        return Ok((bytes, 0));
    }
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
            bytes[..4].copy_from_slice(&value.to_bits().to_ne_bytes());
        },
        ScalarType::Float64 => {
            let value = required(column.f64()?.get(0))?;
            bytes.copy_from_slice(&value.to_bits().to_ne_bytes());
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
    })
}
