use std::iter::zip;

use polars_array::builder::StaticArrayBuilder as _;
use polars_core::prelude::arity::{try_binary_mut_with_options, try_unary_mut_with_options};
use regex::Regex;

use super::*;

#[cfg(feature = "extract_groups")]
fn extract_groups_array(
    arr: &PlUtf8ViewArray,
    reg: &Regex,
    names: &[&str],
) -> PolarsResult<PlArrayRef> {
    let mut builders = (0..names.len())
        .map(|_| PlUtf8ViewArrayBuilder::with_capacity(arr.len()))
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
        .map(|builder| builder.freeze().into_boxed())
        .collect();
    // The mask of a struct array holds one bit per element, which is what a scalar one stands
    // for; the field names live in the `DataType` of the `Series` this becomes a chunk of.
    let validity = arr.validity().map(|v| v.to_flat_or_scalar());
    Ok(PlStructArray::new_broadcast(values, arr.len(), validity).into_boxed())
}

#[cfg(feature = "extract_groups")]
pub(super) fn extract_groups(
    ca: &StringChunked,
    pat: &str,
    dtype: &DataType,
) -> PolarsResult<Series> {
    let reg = polars_utils::regex_cache::compile_regex(pat)?;
    let n_fields = reg.captures_len();
    if n_fields == 1 {
        return StructChunked::from_series(ca.name().clone(), ca.len(), [].iter())
            .map(|ca| ca.into_series());
    }

    let DataType::Struct(fields) = dtype else {
        unreachable!() // Implementation error if it isn't a struct.
    };
    let names = fields
        .iter()
        .map(|fld| fld.name.as_str())
        .collect::<Vec<_>>();

    let chunks = ca
        .downcast_iter()
        .map(|array| extract_groups_array(array, &reg, &names))
        .collect::<PolarsResult<Vec<_>>>()?;

    // SAFETY: one field of strings per capture group, which is what `dtype` names.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, dtype) })
}

fn extract_group_reg_lit(
    arr: &PlUtf8ViewArray,
    reg: &Regex,
    group_index: usize,
) -> PolarsResult<PlUtf8ViewArray> {
    let mut builder = PlUtf8ViewArrayBuilder::with_capacity(arr.len());

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

    Ok(builder.freeze())
}

fn extract_group_array_lit(
    s: &str,
    pat: &PlUtf8ViewArray,
    group_index: usize,
) -> PolarsResult<PlUtf8ViewArray> {
    let mut builder = PlUtf8ViewArrayBuilder::with_capacity(pat.len());

    for opt_pat in pat {
        if let Some(pat) = opt_pat {
            let reg = polars_utils::regex_cache::compile_regex(pat)?;
            let mut locs = reg.capture_locations();
            if reg.captures_read(&mut locs, s).is_some() {
                builder.push(locs.get(group_index).map(|(start, stop)| &s[start..stop]));
                continue;
            }
        }

        // Push null if either the pat is null or there was no match.
        builder.push_null();
    }

    Ok(builder.freeze())
}

fn extract_group_binary(
    arr: &PlUtf8ViewArray,
    pat: &PlUtf8ViewArray,
    group_index: usize,
) -> PolarsResult<PlUtf8ViewArray> {
    let mut builder = PlUtf8ViewArrayBuilder::with_capacity(arr.len());

    for (opt_s, opt_pat) in zip(arr, pat) {
        match (opt_s, opt_pat) {
            (Some(s), Some(pat)) => {
                let reg = polars_utils::regex_cache::compile_regex(pat)?;
                let mut locs = reg.capture_locations();
                if reg.captures_read(&mut locs, s).is_some() {
                    builder.push(locs.get(group_index).map(|(start, stop)| &s[start..stop]));
                    continue;
                }
                // Push null if there was no match.
                builder.push_null()
            },
            _ => builder.push_null(),
        }
    }

    Ok(builder.freeze())
}

pub(super) fn extract_group(
    ca: &StringChunked,
    pat: &StringChunked,
    group_index: usize,
) -> PolarsResult<StringChunked> {
    match (ca.len(), pat.len()) {
        (_, 1) => {
            if let Some(pat) = pat.get(0) {
                let reg = polars_utils::regex_cache::compile_regex(pat)?;
                try_unary_mut_with_options(ca, |arr| extract_group_reg_lit(arr, &reg, group_index))
            } else {
                Ok(StringChunked::full_null(ca.name().clone(), ca.len()))
            }
        },
        (1, _) => {
            if let Some(s) = ca.get(0) {
                try_unary_mut_with_options(pat, |pat| extract_group_array_lit(s, pat, group_index))
            } else {
                Ok(StringChunked::full_null(ca.name().clone(), pat.len()))
            }
        },
        (len_ca, len_pat) if len_ca == len_pat => try_binary_mut_with_options(
            ca,
            pat,
            |ca, pat| extract_group_binary(ca, pat, group_index),
            ca.name().clone(),
        ),
        _ => {
            polars_bail!(ComputeError: "ca(len: {}) and pat(len: {}) should either broadcast or have the same length", ca.len(), pat.len())
        },
    }
}
