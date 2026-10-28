use std::borrow::Cow;

use polars_row::{
    RowEncodingContext, RowEncodingField, RowEncodingLayout, RowEncodingOptions, RowEncodingOutput,
    convert_columns,
};
use polars_utils::itertools::Itertools;
use rayon::prelude::*;

use crate::prelude::*;
use crate::runtime::RAYON;
use crate::utils::_split_offsets;

pub fn encode_rows_vertical_par_unordered(by: &[Column]) -> PolarsResult<BinaryOffsetChunked> {
    encode_binary_offset_par(by, &unordered_opts(by.len()), false)
}

pub fn encode_rows_vertical_par_unordered_broadcast_nulls(
    by: &[Column],
) -> PolarsResult<BinaryOffsetChunked> {
    encode_binary_offset_par(by, &unordered_opts(by.len()), true)
}

/// Get the [`RowEncodingContext`] for a certain [`DataType`].
///
/// This should be given the logical type in order to communicate Polars datatype information down
/// into the row encoding / decoding.
pub fn get_row_encoding_context(dtype: &DataType) -> Option<RowEncodingContext> {
    match dtype {
        DataType::Boolean
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::UInt128
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::String
        | DataType::Binary
        | DataType::BinaryOffset
        | DataType::Null
        | DataType::Time
        | DataType::Date
        | DataType::Datetime(_, _)
        | DataType::Duration(_) => None,

        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, mapping) | DataType::Enum(_, mapping) => {
            use polars_row::RowEncodingCategoricalContext;

            Some(RowEncodingContext::Categorical(
                RowEncodingCategoricalContext {
                    is_enum: matches!(dtype, DataType::Enum(_, _)),
                    mapping: mapping.clone(),
                },
            ))
        },

        DataType::Unknown(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "object")]
        DataType::Object(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, _) => Some(RowEncodingContext::Decimal(*precision)),

        #[cfg(feature = "dtype-array")]
        DataType::Array(dtype, _) => get_row_encoding_context(dtype),
        DataType::List(dtype) => get_row_encoding_context(dtype),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fs) => {
            let mut ctxts = Vec::new();

            for (i, f) in fs.iter().enumerate() {
                if let Some(ctxt) = get_row_encoding_context(f.dtype()) {
                    ctxts.reserve(fs.len());
                    ctxts.extend(std::iter::repeat_n(None, i));
                    ctxts.push(Some(ctxt));
                    break;
                }
            }

            if ctxts.is_empty() {
                return None;
            }

            ctxts.extend(
                fs[ctxts.len()..]
                    .iter()
                    .map(|f| get_row_encoding_context(f.dtype())),
            );

            Some(RowEncodingContext::Struct(ctxts))
        },

        #[cfg(feature = "dtype-map")]
        DataType::Map(key, value) => {
            let ctxts = vec![
                get_row_encoding_context(key),
                get_row_encoding_context(value),
            ];

            if ctxts.iter().all(Option::is_none) {
                return None;
            }

            Some(RowEncodingContext::Struct(ctxts))
        },

        #[cfg(feature = "dtype-extension")]
        DataType::Extension(_, storage) => get_row_encoding_context(storage),
    }
}

/// The layout for row encoding columns of these logical dtypes. `encode_nulls` says per column
/// whether its nulls are encoded, `None` means for all columns.
pub fn get_row_encoding_layout(
    dtypes: &[DataType],
    opts: &[RowEncodingOptions],
    encode_nulls: Option<&[bool]>,
    output: RowEncodingOutput,
) -> RowEncodingLayout {
    assert_eq!(dtypes.len(), opts.len());
    assert!(encode_nulls.is_none_or(|n| n.len() == dtypes.len()));
    let fields = dtypes
        .iter()
        .zip(opts)
        .enumerate()
        .map(|(i, (dtype, opt))| RowEncodingField {
            dtype: dtype.to_physical().to_arrow(CompatLevel::newest()),
            opt: *opt,
            ctx: get_row_encoding_context(dtype),
            encode_nulls: encode_nulls.is_none_or(|n| n[i]),
        })
        .collect();
    RowEncodingLayout::new(fields, output)
}

/// The dtype of the rows of `layout`.
pub fn rows_dtype(layout: &RowEncodingLayout) -> DataType {
    match layout.dtype() {
        ArrowDataType::LargeBinary => DataType::BinaryOffset,
        dtype => DataType::from_arrow_dtype(dtype),
    }
}

fn encode_columns(
    by: &[Column],
    layout: &RowEncodingLayout,
    propagate_nulls: bool,
) -> PolarsResult<ArrayRef> {
    // Since ZFS exists, we might not actually have any arrays and need to get the length from the
    // columns.
    let num_rows = by.first().map_or(0, |c| c.len());

    let mut cols = Vec::with_capacity(by.len());
    for (by, field) in by.iter().zip(layout.fields()) {
        polars_ensure!(
            field.encode_nulls || propagate_nulls || !by.has_nulls(),
            ComputeError: "row encoding: column '{}' contains nulls, but its nulls are not encoded", by.name()
        );

        let by = by
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(by), Cow::Owned);
        let by = by.propagate_nulls().map_or(by, Cow::Owned);
        let by = by.as_materialized_series();
        cols.push(by.to_physical_repr().rechunk().chunks()[0].to_boxed());
    }
    Ok(convert_columns(num_rows, &cols, layout, propagate_nulls))
}

fn unordered_opts(n: usize) -> Vec<RowEncodingOptions> {
    vec![RowEncodingOptions::new_unsorted(); n]
}

fn ordered_opts(descending: &[bool], nulls_last: &[bool]) -> Vec<RowEncodingOptions> {
    debug_assert_eq!(descending.len(), nulls_last.len());
    descending
        .iter()
        .zip(nulls_last)
        .map(|(desc, null_last)| RowEncodingOptions::new_sorted(*desc, *null_last))
        .collect_vec()
}

/// The [`RowEncodingOutput::AlwaysBinaryOffset`] layout for these columns.
pub fn get_binary_offset_row_encoding_layout(
    by: &[Column],
    opts: &[RowEncodingOptions],
) -> RowEncodingLayout {
    let dtypes = by.iter().map(|c| c.dtype().clone()).collect_vec();
    get_row_encoding_layout(&dtypes, opts, None, RowEncodingOutput::AlwaysBinaryOffset)
}

fn encode_binary_offset(
    by: &[Column],
    opts: &[RowEncodingOptions],
    propagate_nulls: bool,
) -> PolarsResult<BinaryArray<i64>> {
    let rows = encode_columns(
        by,
        &get_binary_offset_row_encoding_layout(by, opts),
        propagate_nulls,
    )?;
    Ok(rows
        .as_any()
        .downcast_ref::<BinaryArray<i64>>()
        .unwrap()
        .clone())
}

fn encode_binary_offset_par(
    by: &[Column],
    opts: &[RowEncodingOptions],
    propagate_nulls: bool,
) -> PolarsResult<BinaryOffsetChunked> {
    let rows = encode_rows_with_layout_par(
        by,
        &get_binary_offset_row_encoding_layout(by, opts),
        propagate_nulls,
    )?;
    Ok(rows.binary_offset()?.clone())
}

/// Encodes `by` into rows of `layout`, which must be made for the dtypes of `by`. With
/// `propagate_nulls` a row is null if any of its values is null.
pub fn encode_rows_with_layout(
    by: &[Column],
    layout: &RowEncodingLayout,
    propagate_nulls: bool,
) -> PolarsResult<Series> {
    let rows = encode_columns(by, layout, propagate_nulls)?;
    // SAFETY: the rows have the dtype of the layout.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![rows], &rows_dtype(layout))
    })
}

/// Below this number of rows, encoding rows in parallel isn't worth it.
pub const ROW_ENCODE_PAR_MIN_ROWS: usize = 1 << 16;

/// [`encode_rows_with_layout`], encoding a slice of the rows per thread.
pub fn encode_rows_with_layout_par(
    by: &[Column],
    layout: &RowEncodingLayout,
    propagate_nulls: bool,
) -> PolarsResult<Series> {
    let len = by.first().map_or(0, |c| c.len());
    let n_splits = RAYON.current_num_threads().min(len).max(1);
    let chunks = _split_offsets(len, n_splits)
        .into_par_iter()
        .map(|(offset, len)| {
            let sliced = by
                .iter()
                .map(|c| c.slice(offset as i64, len))
                .collect::<Vec<_>>();
            encode_columns(&sliced, layout, propagate_nulls)
        });
    let chunks = RAYON.install(|| chunks.collect::<PolarsResult<Vec<_>>>())?;
    // SAFETY: the rows have the dtype of the layout.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, chunks, &rows_dtype(layout))
    })
}

/// The [`RowEncodingOutput::Optimal`] layout for these columns, with each column encoding nulls
/// if it has nulls in `by` or in any of `others`. Falls back to the
/// [`RowEncodingOutput::AlwaysBinaryOffset`] layout if the rows are not packed or passed through,
/// or if this build does not support the dtype of the rows.
pub fn get_optimal_row_encoding_layout(
    by: &[Column],
    others: &[&[Column]],
    opts: &[RowEncodingOptions],
) -> RowEncodingLayout {
    let dtypes = by.iter().map(|c| c.dtype().clone()).collect_vec();
    let encode_nulls = (0..by.len())
        .map(|i| by[i].has_nulls() || others.iter().any(|o| o[i].has_nulls()))
        .collect_vec();
    let layout = |output| get_row_encoding_layout(&dtypes, opts, Some(&encode_nulls), output);
    let optimal = layout(RowEncodingOutput::Optimal);
    let unsupported_u128 =
        *optimal.dtype() == ArrowDataType::UInt128 && !cfg!(feature = "dtype-u128");
    if unsupported_u128 || !(optimal.is_packed() || optimal.is_pass_through()) {
        return layout(RowEncodingOutput::AlwaysBinaryOffset);
    }
    optimal
}

pub fn encode_rows_unordered(by: &[Column]) -> PolarsResult<BinaryOffsetChunked> {
    let rows = _get_rows_encoded_unordered(by)?;
    Ok(BinaryOffsetChunked::with_chunk(PlSmallStr::EMPTY, rows))
}

pub fn _get_rows_encoded_unordered(by: &[Column]) -> PolarsResult<BinaryArray<i64>> {
    encode_binary_offset(by, &unordered_opts(by.len()), false)
}

pub fn _get_rows_encoded(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryArray<i64>> {
    _get_rows_encoded_arr(by, descending, nulls_last, false)
}

pub fn _get_rows_encoded_ca(
    name: PlSmallStr,
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
    propagate_nulls: bool,
) -> PolarsResult<BinaryOffsetChunked> {
    let rows_arr = _get_rows_encoded_arr(by, descending, nulls_last, propagate_nulls)?;
    Ok(BinaryOffsetChunked::with_chunk(name, rows_arr))
}

pub fn _get_rows_encoded_arr(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
    propagate_nulls: bool,
) -> PolarsResult<BinaryArray<i64>> {
    debug_assert_eq!(by.len(), descending.len());
    encode_binary_offset(by, &ordered_opts(descending, nulls_last), propagate_nulls)
}

pub fn _get_rows_encoded_ca_unordered(
    name: PlSmallStr,
    by: &[Column],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded_unordered(by).map(|rows| BinaryOffsetChunked::with_chunk(name, rows))
}

/// Decodes rows of `layout` into a struct with these fields.
#[cfg(feature = "dtype-struct")]
pub fn row_encoding_decode(
    rows: &Series,
    fields: &[Field],
    layout: &RowEncodingLayout,
) -> PolarsResult<StructChunked> {
    let expected = rows_dtype(layout);
    polars_ensure!(
        rows.dtype() == &expected,
        SchemaMismatch: "row decoding expected rows of type {}, got {}", expected, rows.dtype()
    );

    let struct_arrow_dtype = ArrowDataType::Struct(
        fields
            .iter()
            .map(|v| v.to_physical().to_arrow(CompatLevel::newest()))
            .collect(),
    );

    let chunks = rows
        .chunks()
        .iter()
        .map(|array| {
            let decoded_arrays =
                unsafe { polars_row::decode::decode_rows_from_array(array.as_ref(), layout) };
            assert_eq!(decoded_arrays.len(), fields.len());

            StructArray::new(
                struct_arrow_dtype.clone(),
                array.len(),
                decoded_arrays,
                None,
            )
            .to_boxed()
        })
        .collect::<Vec<_>>();

    Ok(unsafe {
        StructChunked::from_chunks_and_dtype(
            rows.name().clone(),
            chunks,
            DataType::Struct(fields.to_vec()),
        )
    })
}
