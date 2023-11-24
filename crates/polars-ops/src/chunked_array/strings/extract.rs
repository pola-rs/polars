#[cfg(feature = "extract_groups")]
use arrow::array::{Array, StructArray};
use arrow::array::{MutableArray, MutableUtf8Array, Utf8Array, ArrayRef};
use polars_core::export::regex::Regex;

use super::*;

#[cfg(feature = "extract_groups")]
fn extract_groups_array(
    arr: &Utf8Array<i64>,
    reg: &Regex,
    names: &[&str],
    data_type: ArrowDataType,
) -> PolarsResult<ArrayRef> {
    let mut builders = (0..names.len())
        .map(|_| MutableUtf8Array::<i64>::with_capacity(arr.len()))
        .collect::<Vec<_>>();

    let mut locs = reg.capture_locations();
    for opt_v in arr {
        if let Some(s) = opt_v {
            if reg.captures_read(&mut locs, s).is_some() {
                for (i, builder) in builders.iter_mut().enumerate() {
                    builder.push(locs.get(i + 1).map(|(start, stop)| &s[start..stop]));
                }
                continue;
            }
        }

        // Push nulls if either the string is null or there was no match. We
        // distinguish later between the two by copying arr's validity mask.
        builders.iter_mut().for_each(|arr| arr.push_null());
    }

    let values = builders
        .into_iter()
        .map(|a| {
            let immutable_a: Utf8Array<i64> = a.into();
            immutable_a.to_boxed()
        })
        .collect();
    Ok(StructArray::new(data_type.clone(), values, arr.validity().cloned()).boxed())
}

#[cfg(feature = "extract_groups")]
pub(super) fn extract_groups(
    ca: &Utf8Chunked,
    pat: &str,
    dtype: &DataType,
) -> PolarsResult<Series> {
    let reg = Regex::new(pat)?;
    let n_fields = reg.captures_len();
    if n_fields == 1 {
        return StructChunked::new(ca.name(), &[Series::new_null(ca.name(), ca.len())])
            .map(|ca| ca.into_series());
    }

    let data_type = dtype.to_arrow();
    let DataType::Struct(fields) = dtype else {
        unreachable!() // Implementation error if it isn't a struct.
    };
    let names = fields
        .iter()
        .map(|fld| fld.name.as_str())
        .collect::<Vec<_>>();

    let chunks = ca
        .downcast_iter()
        .map(|array| extract_groups_array(array, &reg, &names, data_type.clone()))
        .collect::<PolarsResult<Vec<_>>>()?;

    Series::try_from((ca.name(), chunks))
}

fn extract_group_array(
    arr: &Utf8Array<i64>,
    reg: &Regex,
    group_index: usize,
) -> PolarsResult<Utf8Array<i64>> {
    let mut builder = MutableUtf8Array::<i64>::with_capacity(arr.len());

    let mut locs = reg.capture_locations();
    for opt_v in arr {
        if let Some(s) = opt_v {
            if reg.captures_read(&mut locs, s).is_some() {
                builder.push(locs.get(group_index).map(|(start, stop)| &s[start..stop]));
                continue;
            }
        }

        // Push null if either the string is null or there was no match.
        builder.push_null();
    }

    Ok(builder.into())
}

pub(super) fn extract_group(
    ca: &Utf8Chunked,
    pat: &str,
    group_index: usize,
) -> PolarsResult<Utf8Chunked> {
    let reg = Regex::new(pat)?;
    let chunks = ca
        .downcast_iter()
        .map(|array| extract_group_array(array, &reg, group_index));
    ChunkedArray::try_from_chunk_iter(ca.name(), chunks)
}
