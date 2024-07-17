use std::iter::zip;

#[cfg(feature = "extract_groups")]
use arrow::array::{Array, StructArray};
use arrow::array::{MutablePlString, Utf8ViewArray};
use polars_core::export::regex::Regex;
use polars_core::prelude::arity::{try_binary_mut_with_options, try_unary_mut_with_options};

use super::*;

#[cfg(feature = "extract_groups")]
fn extract_groups_array(
    arr: &Utf8ViewArray,
    reg: &Regex,
    names: &[&str],
    data_type: ArrowDataType,
) -> PolarsResult<ArrayRef> {
    let mut builders = (0..names.len())
        .map(|_| MutablePlString::with_capacity(arr.len()))
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

    let values = builders.into_iter().map(|a| a.freeze().boxed()).collect();
    Ok(StructArray::new(data_type.clone(), values, arr.validity().cloned()).boxed())
}

#[cfg(feature = "extract_groups")]
pub(super) fn extract_groups(
    ca: &StringChunked,
    pat: &str,
    dtype: &DataType,
) -> PolarsResult<Series> {
    let reg = Regex::new(pat)?;
    let n_fields = reg.captures_len();
    if n_fields == 1 {
        return StructChunked::from_series(ca.name(), &[Series::new_null(ca.name(), ca.len())])
            .map(|ca| ca.into_series());
    }

    let data_type = dtype.try_to_arrow(CompatLevel::newest())?;
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

fn extract_group_reg_lit(
    arr: &Utf8ViewArray,
    reg: &Regex,
    group_index: usize,
) -> PolarsResult<Utf8ViewArray> {
    let mut builder = MutablePlString::with_capacity(arr.len());

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

fn extract_group_array_lit(
    s: &str,
    pat: &Utf8ViewArray,
    group_index: usize,
) -> PolarsResult<Utf8ViewArray> {
    let mut builder = MutablePlString::with_capacity(pat.len());

    for opt_pat in pat {
        if let Some(pat) = opt_pat {
            let reg = Regex::new(pat)?;
            let mut locs = reg.capture_locations();
            if reg.captures_read(&mut locs, s).is_some() {
                builder.push(locs.get(group_index).map(|(start, stop)| &s[start..stop]));
                continue;
            }
        }

        // Push null if either the pat is null or there was no match.
        builder.push_null();
    }

    Ok(builder.into())
}

fn extract_group_binary(
    arr: &Utf8ViewArray,
    pat: &Utf8ViewArray,
    group_index: usize,
) -> PolarsResult<Utf8ViewArray> {
    let mut builder = MutablePlString::with_capacity(arr.len());

    for (opt_s, opt_pat) in zip(arr, pat) {
        match (opt_s, opt_pat) {
            (Some(s), Some(pat)) => {
                let reg = Regex::new(pat)?;
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

    Ok(builder.into())
}

pub(super) fn extract_group(
    ca: &StringChunked,
    pat: &StringChunked,
    group_index: usize,
) -> PolarsResult<StringChunked> {
    match (ca.len(), pat.len()) {
        (_, 1) => {
            if let Some(pat) = pat.get(0) {
                let reg = Regex::new(pat)?;
                try_unary_mut_with_options(ca, |arr| extract_group_reg_lit(arr, &reg, group_index))
            } else {
                Ok(StringChunked::full_null(ca.name(), ca.len()))
            }
        },
        (1, _) => {
            if let Some(s) = ca.get(0) {
                try_unary_mut_with_options(pat, |pat| extract_group_array_lit(s, pat, group_index))
            } else {
                Ok(StringChunked::full_null(ca.name(), pat.len()))
            }
        },
        (len_ca, len_pat) if len_ca == len_pat => try_binary_mut_with_options(
            ca,
            pat,
            |ca, pat| extract_group_binary(ca, pat, group_index),
            ca.name(),
        ),
        _ => {
            polars_bail!(ComputeError: "ca(len: {}) and pat(len: {}) should either broadcast or have the same length", ca.len(), pat.len())
        },
    }
}
