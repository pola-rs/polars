use arrow::compute::utils::combine_validities_and_many;
use polars_row::{convert_columns, EncodingField, RowsEncoded};
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::_split_offsets;
use crate::POOL;

pub(crate) fn convert_series_for_row_encoding(s: &Series) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        Categorical(_, _) | Enum(_, _) => s.rechunk(),
        Binary | Boolean => s.clone(),
        BinaryOffset => s.clone(),
        String => s.str().unwrap().as_binary().into_series(),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let ca = s.struct_().unwrap();
            let new_fields = ca
                .fields_as_series()
                .iter()
                .map(convert_series_for_row_encoding)
                .collect::<PolarsResult<Vec<_>>>()?;
            let mut out =
                StructChunked::from_series(ca.name().clone(), ca.len(), new_fields.iter())?;
            out.zip_outer_validity(ca);
            out.into_series()
        },
        // we could fallback to default branch, but decimal is not numeric dtype for now, so explicit here
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => s.clone(),
        List(inner) if !inner.is_nested() => s.clone(),
        Null => s.clone(),
        _ => {
            let phys = s.to_physical_repr().into_owned();
            polars_ensure!(
                phys.dtype().is_numeric(),
                InvalidOperation: "cannot sort column of dtype `{}`", s.dtype()
            );
            phys
        },
    };
    Ok(out)
}

pub fn _get_rows_encoded_compat_array(by: &Series) -> PolarsResult<ArrayRef> {
    let by = convert_series_for_row_encoding(by)?;
    let by = by.rechunk();

    let out = match by.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let ca = by.categorical().unwrap();
            if ca.uses_lexical_ordering() {
                by.to_arrow(0, CompatLevel::newest())
            } else {
                ca.physical().chunks[0].clone()
            }
        },
        // Take physical
        _ => by.chunks()[0].clone(),
    };
    Ok(out)
}

pub fn encode_rows_vertical_par_unordered(by: &[Series]) -> PolarsResult<BinaryOffsetChunked> {
    let n_threads = POOL.current_num_threads();
    let len = by[0].len();
    let splits = _split_offsets(len, n_threads);

    let chunks = splits.into_par_iter().map(|(offset, len)| {
        let sliced = by
            .iter()
            .map(|s| s.slice(offset as i64, len))
            .collect::<Vec<_>>();
        let rows = _get_rows_encoded_unordered(&sliced)?;
        Ok(rows.into_array())
    });
    let chunks = POOL.install(|| chunks.collect::<PolarsResult<Vec<_>>>());

    Ok(BinaryOffsetChunked::from_chunk_iter(
        PlSmallStr::EMPTY,
        chunks?,
    ))
}

// Almost the same but broadcast nulls to the row-encoded array.
pub fn encode_rows_vertical_par_unordered_broadcast_nulls(
    by: &[Series],
) -> PolarsResult<BinaryOffsetChunked> {
    let n_threads = POOL.current_num_threads();
    let len = by[0].len();
    let splits = _split_offsets(len, n_threads);

    let chunks = splits.into_par_iter().map(|(offset, len)| {
        let sliced = by
            .iter()
            .map(|s| s.slice(offset as i64, len))
            .collect::<Vec<_>>();
        let rows = _get_rows_encoded_unordered(&sliced)?;

        let validities = sliced
            .iter()
            .flat_map(|s| {
                let s = s.rechunk();
                #[allow(clippy::unnecessary_to_owned)]
                s.chunks()
                    .to_vec()
                    .into_iter()
                    .map(|arr| arr.validity().cloned())
            })
            .collect::<Vec<_>>();

        let validity = combine_validities_and_many(&validities);
        Ok(rows.into_array().with_validity_typed(validity))
    });
    let chunks = POOL.install(|| chunks.collect::<PolarsResult<Vec<_>>>());

    Ok(BinaryOffsetChunked::from_chunk_iter(
        PlSmallStr::EMPTY,
        chunks?,
    ))
}

pub fn encode_rows_unordered(by: &[Series]) -> PolarsResult<BinaryOffsetChunked> {
    let rows = _get_rows_encoded_unordered(by)?;
    Ok(BinaryOffsetChunked::with_chunk(
        PlSmallStr::EMPTY,
        rows.into_array(),
    ))
}

pub fn _get_rows_encoded_unordered(by: &[Series]) -> PolarsResult<RowsEncoded> {
    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());
    for by in by {
        let arr = _get_rows_encoded_compat_array(by)?;
        let field = EncodingField::new_unsorted();
        match arr.dtype() {
            // Flatten the struct fields.
            ArrowDataType::Struct(_) => {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                for arr in arr.values() {
                    cols.push(arr.clone() as ArrayRef);
                    fields.push(field)
                }
            },
            _ => {
                cols.push(arr);
                fields.push(field)
            },
        }
    }
    Ok(convert_columns(&cols, &fields))
}

pub fn _get_rows_encoded(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<RowsEncoded> {
    debug_assert_eq!(by.len(), descending.len());
    debug_assert_eq!(by.len(), nulls_last.len());

    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());

    for ((by, desc), null_last) in by.iter().zip(descending).zip(nulls_last) {
        let by = by.as_materialized_series();
        let arr = _get_rows_encoded_compat_array(by)?;
        let sort_field = EncodingField {
            descending: *desc,
            nulls_last: *null_last,
            no_order: false,
        };
        match arr.dtype() {
            // Flatten the struct fields.
            ArrowDataType::Struct(_) => {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                let arr = arr.propagate_nulls();
                for value_arr in arr.values() {
                    cols.push(value_arr.clone() as ArrayRef);
                    fields.push(sort_field);
                }
            },
            _ => {
                cols.push(arr);
                fields.push(sort_field);
            },
        }
    }
    Ok(convert_columns(&cols, &fields))
}

pub fn _get_rows_encoded_ca(
    name: PlSmallStr,
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded(by, descending, nulls_last)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}

pub fn _get_rows_encoded_arr(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryArray<i64>> {
    _get_rows_encoded(by, descending, nulls_last).map(|rows| rows.into_array())
}

pub fn _get_rows_encoded_ca_unordered(
    name: PlSmallStr,
    by: &[Series],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded_unordered(by)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}
