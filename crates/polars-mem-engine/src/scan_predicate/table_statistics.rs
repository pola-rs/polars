use std::borrow::Cow;

use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;

/// Supplied string bounds use lexical order (including Parquet ENUM statistics),
/// whereas enum bounds use declaration order. Normalize before evaluating a
/// skip-batch predicate, whose statistics schema uses the scan's logical dtypes.
pub(super) fn normalize_enum_statistics<'a>(
    statistics: &'a DataFrame,
    schema: &Schema,
) -> PolarsResult<Cow<'a, DataFrame>> {
    let mut out = Cow::Borrowed(statistics);
    for (name, dtype) in schema.iter() {
        if !dtype.contains_enums() {
            continue;
        }
        let min_name = format_pl_smallstr!("{name}_min");
        let max_name = format_pl_smallstr!("{name}_max");
        let (Ok(min), Ok(max)) = (statistics.column(&min_name), statistics.column(&max_name))
        else {
            continue;
        };
        if let Some((min, max)) = normalize_bounds(
            min.as_materialized_series(),
            max.as_materialized_series(),
            dtype,
        )? {
            out.to_mut().with_column(min.into_column())?;
            out.to_mut().with_column(max.into_column())?;
        }
    }
    Ok(out)
}

fn normalize_bounds(
    min: &Series,
    max: &Series,
    dtype: &DataType,
) -> PolarsResult<Option<(Series, Series)>> {
    match dtype {
        DataType::Enum(categories, _) if min.dtype().is_string() || max.dtype().is_string() => {
            let mut lower = Vec::with_capacity(min.len());
            let mut upper = Vec::with_capacity(max.len());
            let min_values = min.str().ok();
            let max_values = max.str().ok();
            for i in 0..min.len() {
                let bounds = min_values
                    .and_then(|s| s.get(i))
                    .zip(max_values.and_then(|s| s.get(i)));
                let bounds = bounds.and_then(|(lo, hi)| {
                    // Casting just the endpoints is unsound: a category between
                    // them lexically may lie outside their enum-code interval.
                    // Iteration is in enum order, so the first and last possible
                    // categories bound every value the file could contain.
                    let mut possible = categories
                        .categories()
                        .values_iter()
                        .filter(|&s| lo <= s && s <= hi);
                    let lower = possible.next()?;
                    let upper = possible.next_back().unwrap_or(lower);
                    Some((lower, upper))
                });
                // Missing, reversed, or empty intervals become unknown. Bounds
                // need not themselves be categories (e.g. truncated statistics).
                lower.push(bounds.map(|(lo, _)| lo));
                upper.push(bounds.map(|(_, hi)| hi));
            }
            Ok(Some((
                Series::new(min.name().clone(), lower).cast(dtype)?,
                Series::new(max.name().clone(), upper).cast(dtype)?,
            )))
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fields) => {
            let (Ok(min_ca), Ok(max_ca)) = (min.struct_(), max.struct_()) else {
                return Ok(None);
            };
            let mut mins = min_ca.fields_as_series();
            let mut maxs = max_ca.fields_as_series();
            let mut changed = false;
            for field in fields {
                let min = mins.iter_mut().find(|s| s.name() == field.name());
                let max = maxs.iter_mut().find(|s| s.name() == field.name());
                let (Some(min), Some(max)) = (min, max) else {
                    continue;
                };
                if let Some((new_min, new_max)) = normalize_bounds(min, max, field.dtype())? {
                    *min = new_min;
                    *max = new_max;
                    changed = true;
                }
            }
            if !changed {
                return Ok(None);
            }
            let mut lower = StructChunked::from_series(min.name().clone(), min.len(), mins.iter())?;
            let mut upper = StructChunked::from_series(max.name().clone(), max.len(), maxs.iter())?;
            lower.zip_outer_validity(min_ca);
            upper.zip_outer_validity(max_ca);
            Ok(Some((lower.into_series(), upper.into_series())))
        },
        // Enum-typed bounds already use enum order and must be left unchanged.
        _ => Ok(None),
    }
}
