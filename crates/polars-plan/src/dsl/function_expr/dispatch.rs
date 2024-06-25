use super::*;

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_n_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::approx_n_unique(s)
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    polars_ops::prelude::diff(s, n, null_behavior)
}

#[cfg(feature = "pct_change")]
pub(super) fn pct_change(s: &[Series]) -> PolarsResult<Series> {
    polars_ops::prelude::pct_change(&s[0], &s[1])
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Series, method: InterpolationMethod) -> PolarsResult<Series> {
    Ok(polars_ops::prelude::interpolate(s, method))
}

#[cfg(feature = "interpolate_by")]
pub(super) fn interpolate_by(s: &[Series]) -> PolarsResult<Series> {
    let by = &s[1];
    let by_is_sorted = by.is_sorted(Default::default())?;
    polars_ops::prelude::interpolate_by(&s[0], by, by_is_sorted)
}

pub(super) fn to_physical(s: &Series) -> PolarsResult<Series> {
    Ok(s.to_physical_repr().into_owned())
}

pub(super) fn set_sorted_flag(s: &Series, sorted: IsSorted) -> PolarsResult<Series> {
    let mut s = s.clone();
    s.set_sorted_flag(sorted);
    Ok(s)
}

#[cfg(feature = "timezones")]
pub(super) fn replace_time_zone(
    s: &[Series],
    time_zone: Option<&str>,
    non_existent: NonExistent,
) -> PolarsResult<Series> {
    let s1 = &s[0];
    let ca = s1.datetime().unwrap();
    let s2 = &s[1].str()?;
    Ok(polars_ops::prelude::replace_time_zone(ca, time_zone, s2, non_existent)?.into_series())
}

#[cfg(feature = "dtype-struct")]
pub(super) fn value_counts(
    s: &Series,
    sort: bool,
    parallel: bool,
    name: String,
    normalize: bool,
) -> PolarsResult<Series> {
    s.value_counts(sort, parallel, name, normalize)
        .map(|df| df.into_struct(s.name()).into_series())
}

#[cfg(feature = "unique_counts")]
pub(super) fn unique_counts(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::unique_counts(s)
}

pub(super) fn reshape(s: &Series, dimensions: &[i64], nested: &NestedType) -> PolarsResult<Series> {
    match nested {
        NestedType::List => s.reshape_list(dimensions),
        #[cfg(feature = "dtype-array")]
        NestedType::Array => s.reshape_array(dimensions),
    }
}

#[cfg(feature = "repeat_by")]
pub(super) fn repeat_by(s: &[Series]) -> PolarsResult<Series> {
    let by = &s[1];
    let s = &s[0];
    let by = by.cast(&IDX_DTYPE)?;
    polars_ops::chunked_array::repeat_by(s, by.idx()?).map(|ok| ok.into_series())
}

pub(super) fn backward_fill(s: &Series, limit: FillNullLimit) -> PolarsResult<Series> {
    s.fill_null(FillNullStrategy::Backward(limit))
}

pub(super) fn forward_fill(s: &Series, limit: FillNullLimit) -> PolarsResult<Series> {
    s.fill_null(FillNullStrategy::Forward(limit))
}

pub(super) fn max_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::max_horizontal(s)
}

pub(super) fn min_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::min_horizontal(s)
}

pub(super) fn sum_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::sum_horizontal(s)
}

pub(super) fn mean_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    polars_ops::prelude::mean_horizontal(s)
}

pub(super) fn drop_nulls(s: &Series) -> PolarsResult<Series> {
    Ok(s.drop_nulls())
}

#[cfg(feature = "mode")]
pub(super) fn mode(s: &Series) -> PolarsResult<Series> {
    mode::mode(s)
}

#[cfg(feature = "moment")]
pub(super) fn skew(s: &Series, bias: bool) -> PolarsResult<Series> {
    s.skew(bias).map(|opt_v| Series::new(s.name(), &[opt_v]))
}

#[cfg(feature = "moment")]
pub(super) fn kurtosis(s: &Series, fisher: bool, bias: bool) -> PolarsResult<Series> {
    s.kurtosis(fisher, bias)
        .map(|opt_v| Series::new(s.name(), &[opt_v]))
}

pub(super) fn arg_unique(s: &Series) -> PolarsResult<Series> {
    s.arg_unique().map(|ok| ok.into_series())
}

#[cfg(feature = "rank")]
pub(super) fn rank(s: &Series, options: RankOptions, seed: Option<u64>) -> PolarsResult<Series> {
    Ok(s.rank(options, seed))
}

#[cfg(feature = "hist")]
pub(super) fn hist(
    s: &[Series],
    bin_count: Option<usize>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Series> {
    let bins = if s.len() == 2 {
        Some(s[1].clone())
    } else {
        None
    };
    let s = &s[0];
    hist_series(s, bin_count, bins, include_category, include_breakpoint)
}

#[cfg(feature = "replace")]
pub(super) fn replace(s: &[Series]) -> PolarsResult<Series> {
    polars_ops::series::replace(&s[0], &s[1], &s[2])
}

#[cfg(feature = "replace")]
pub(super) fn replace_strict(s: &[Series], return_dtype: Option<DataType>) -> PolarsResult<Series> {
    match s.get(3) {
        Some(default) => {
            polars_ops::series::replace_or_default(&s[0], &s[1], &s[2], default, return_dtype)
        },
        None => polars_ops::series::replace_strict(&s[0], &s[1], &s[2], return_dtype),
    }
}

pub(super) fn fill_null_with_strategy(
    s: &Series,
    strategy: FillNullStrategy,
) -> PolarsResult<Series> {
    s.fill_null(strategy)
}

pub(super) fn gather_every(s: &Series, n: usize, offset: usize) -> PolarsResult<Series> {
    polars_ensure!(n > 0, InvalidOperation: "gather_every(n): n should be positive");
    Ok(s.gather_every(n, offset))
}

#[cfg(feature = "reinterpret")]
pub(super) fn reinterpret(s: &Series, signed: bool) -> PolarsResult<Series> {
    polars_ops::series::reinterpret(s, signed)
}

pub(super) fn negate(s: &Series) -> PolarsResult<Series> {
    polars_ops::series::negate(s)
}

pub(super) fn extend_constant(s: &[Series]) -> PolarsResult<Series> {
    let value = &s[1];
    let n = &s[2];
    polars_ensure!(value.len() == 1 && n.len() == 1, ComputeError: "value and n should have unit length.");
    let n = n.strict_cast(&DataType::UInt64)?;
    let v = value.get(0)?;
    let s = &s[0];
    match n.u64()?.get(0) {
        Some(n) => s.extend_constant(v, n as usize),
        None => {
            polars_bail!(ComputeError: "n can not be None for extend_constant.")
        },
    }
}
