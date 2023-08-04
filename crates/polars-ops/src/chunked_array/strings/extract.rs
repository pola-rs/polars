use arrow::array::{Array, MutableArray, MutableUtf8Array, StructArray, Utf8Array};
use polars_arrow::utils::combine_validities_and;
use polars_core::export::regex::Regex;

use super::*;

fn extract_groups_array(
    arr: &Utf8Array<i64>,
    reg: &Regex,
    names: &[String],
    data_type: ArrowDataType,
) -> PolarsResult<ArrayRef> {
    let mut builders = (0..names.len())
        .map(|_| MutableUtf8Array::<i64>::with_capacity(arr.len()))
        .collect::<Vec<_>>();

    arr.into_iter().for_each(|opt_v| {
        // we combine the null validity later
        if let Some(value) = opt_v {
            let caps = reg.captures(value);
            match caps {
                None => builders.iter_mut().for_each(|arr| arr.push_null()),
                Some(caps) => {
                    caps.iter()
                        .skip(1) // skip 0th group
                        .zip(builders.iter_mut())
                        .for_each(|(m, builder)| builder.push(m.map(|m| m.as_str())))
                }
            }
        }
    });

    let values = builders
        .into_iter()
        .map(|group_array| {
            let group_array: Utf8Array<i64> = group_array.into();
            let final_validity = combine_validities_and(group_array.validity(), arr.validity());
            group_array.with_validity(final_validity).to_boxed()
        })
        .collect();

    Ok(StructArray::new(data_type.clone(), values, None).boxed())
}

pub(super) fn extract_groups(ca: &Utf8Chunked, pat: &str) -> PolarsResult<Series> {
    let reg = Regex::new(pat)?;
    let n_fields = reg.captures_len();

    if n_fields == 1 {
        return StructChunked::new(ca.name(), &[Series::new_null(ca.name(), ca.len())])
            .map(|ca| ca.into_series());
    }

    let names = reg
        .capture_names()
        .enumerate()
        .skip(1)
        .map(|(idx, opt_name)| {
            opt_name
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("{idx}"))
        })
        .collect::<Vec<_>>();
    let data_type = ArrowDataType::Struct(
        names
            .iter()
            .map(|name| ArrowField::new(name.as_str(), ArrowDataType::LargeUtf8, true))
            .collect(),
    );

    let chunks = ca
        .downcast_iter()
        .map(|array| extract_groups_array(array, &reg, &names, data_type.clone()))
        .collect::<PolarsResult<Vec<_>>>()?;

    Series::try_from((ca.name(), chunks))
}
