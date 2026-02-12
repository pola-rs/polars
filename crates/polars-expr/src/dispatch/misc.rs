use polars_core::error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_core::prelude::row_encode::{_get_rows_encoded_ca, _get_rows_encoded_ca_unordered};
use polars_core::prelude::*;
use polars_core::scalar::Scalar;
use polars_core::series::ops::NullBehavior;
use polars_core::series::{IsSorted, Series};
use polars_core::utils::try_get_supertype;
#[cfg(feature = "interpolate")]
use polars_ops::series::InterpolationMethod;
#[cfg(feature = "rank")]
use polars_ops::series::RankOptions;
use polars_ops::series::{ArgAgg, NullStrategy, SeriesMethods};
#[cfg(feature = "dtype-array")]
use polars_plan::dsl::ReshapeDimension;
#[cfg(feature = "fused")]
use polars_plan::plans::FusedOperator;
#[cfg(feature = "cov")]
use polars_plan::plans::IRCorrelationMethod;
use polars_plan::plans::{DynamicPred, RowEncodingVariant};
use polars_row::RowEncodingOptions;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

#[cfg(feature = "abs")]
pub(super) fn abs(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::abs(s.as_materialized_series()).map(Column::from)
}

pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    Ok(s.reverse())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_n_unique(s: &Column) -> PolarsResult<Column> {
    s.approx_n_unique()
        .map(|v| Column::new_scalar(s.name().clone(), Scalar::new(IDX_DTYPE, v.into()), 1))
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &[Column], null_behavior: NullBehavior) -> PolarsResult<Column> {
    let s1 = s[0].as_materialized_series();
    let n = &s[1];

    polars_ensure!(
        n.len() == 1,
        ComputeError: "n must be a single value."
    );
    let n = n.strict_cast(&DataType::Int64)?;
    match n.i64()?.get(0) {
        Some(n) => polars_ops::prelude::diff(s1, n, null_behavior).map(Column::from),
        None => polars_bail!(ComputeError: "'n' can not be None for diff"),
    }
}

#[cfg(feature = "pct_change")]
pub(super) fn pct_change(s: &[Column]) -> PolarsResult<Column> {
    polars_ops::prelude::pct_change(s[0].as_materialized_series(), s[1].as_materialized_series())
        .map(Column::from)
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Column, method: InterpolationMethod) -> PolarsResult<Column> {
    Ok(polars_ops::prelude::interpolate(s.as_materialized_series(), method).into())
}

#[cfg(feature = "interpolate_by")]
pub(super) fn interpolate_by(s: &[Column]) -> PolarsResult<Column> {
    use polars_ops::series::SeriesMethods;

    let by = &s[1];
    let by_is_sorted = by.as_materialized_series().is_sorted(Default::default())?;
    polars_ops::prelude::interpolate_by(&s[0], by, by_is_sorted)
}

pub(super) fn to_physical(s: &Column) -> PolarsResult<Column> {
    Ok(s.to_physical_repr())
}

pub(super) fn set_sorted_flag(s: &Column, sorted: IsSorted) -> PolarsResult<Column> {
    let mut s = s.clone();
    s.set_sorted_flag(sorted);
    Ok(s)
}

#[cfg(feature = "timezones")]
pub(super) fn replace_time_zone(
    s: &[Column],
    time_zone: Option<&TimeZone>,
    non_existent: NonExistent,
) -> PolarsResult<Column> {
    let s1 = &s[0];
    let ca = s1.datetime().unwrap();
    let s2 = &s[1].str()?;
    Ok(polars_ops::prelude::replace_time_zone(ca, time_zone, s2, non_existent)?.into_column())
}

#[cfg(feature = "dtype-struct")]
pub(super) fn value_counts(
    s: &Column,
    sort: bool,
    parallel: bool,
    name: PlSmallStr,
    normalize: bool,
) -> PolarsResult<Column> {
    use polars_ops::series::SeriesMethods;

    s.as_materialized_series()
        .value_counts(sort, parallel, name, normalize)
        .map(|df| df.into_struct(s.name().clone()).into_column())
}

#[cfg(feature = "unique_counts")]
pub(super) fn unique_counts(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::unique_counts(s.as_materialized_series()).map(Column::from)
}

#[cfg(feature = "dtype-array")]
pub(super) fn reshape(c: &Column, dimensions: &[ReshapeDimension]) -> PolarsResult<Column> {
    c.reshape_array(dimensions)
}

#[cfg(feature = "repeat_by")]
pub(super) fn repeat_by(s: &[Column]) -> PolarsResult<Column> {
    let by = &s[1];
    let s = &s[0];
    let by = by.strict_cast(&IDX_DTYPE)?;
    polars_ops::chunked_array::repeat_by(s.as_materialized_series(), by.idx()?)
        .map(|ok| ok.into_column())
}

pub(super) fn max_horizontal(s: &mut [Column]) -> PolarsResult<Column> {
    polars_ops::prelude::max_horizontal(s).map(Option::unwrap)
}

pub(super) fn min_horizontal(s: &mut [Column]) -> PolarsResult<Column> {
    polars_ops::prelude::min_horizontal(s).map(Option::unwrap)
}

pub(super) fn sum_horizontal(s: &mut [Column], ignore_nulls: bool) -> PolarsResult<Column> {
    let null_strategy = if ignore_nulls {
        NullStrategy::Ignore
    } else {
        NullStrategy::Propagate
    };
    polars_ops::prelude::sum_horizontal(s, null_strategy).map(Option::unwrap)
}

pub(super) fn mean_horizontal(s: &mut [Column], ignore_nulls: bool) -> PolarsResult<Column> {
    let null_strategy = if ignore_nulls {
        NullStrategy::Ignore
    } else {
        NullStrategy::Propagate
    };
    polars_ops::prelude::mean_horizontal(s, null_strategy).map(Option::unwrap)
}

pub(super) fn drop_nulls(s: &Column) -> PolarsResult<Column> {
    Ok(s.drop_nulls())
}

pub fn rechunk(s: &Column) -> PolarsResult<Column> {
    Ok(s.rechunk())
}

pub fn append(s: &[Column], upcast: bool) -> PolarsResult<Column> {
    assert_eq!(s.len(), 2);

    let a = &s[0];
    let b = &s[1];

    if upcast {
        let dtype = try_get_supertype(a.dtype(), b.dtype())?;
        let mut a = a.cast(&dtype)?;
        a.append_owned(b.cast(&dtype)?)?;
        Ok(a)
    } else {
        let mut a = a.clone();
        a.append(b)?;
        Ok(a)
    }
}

#[cfg(feature = "mode")]
pub(super) fn mode(s: &Column, maintain_order: bool) -> PolarsResult<Column> {
    polars_ops::prelude::mode::mode(s.as_materialized_series(), maintain_order).map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn skew(s: &Column, bias: bool) -> PolarsResult<Column> {
    // @scalar-opt

    use polars_ops::series::MomentSeries;
    s.as_materialized_series()
        .skew(bias)
        .map(|opt_v| Column::new(s.name().clone(), &[opt_v]))
}

#[cfg(feature = "moment")]
pub(super) fn kurtosis(s: &Column, fisher: bool, bias: bool) -> PolarsResult<Column> {
    // @scalar-opt

    use polars_ops::series::MomentSeries;
    s.as_materialized_series()
        .kurtosis(fisher, bias)
        .map(|opt_v| Column::new(s.name().clone(), &[opt_v]))
}

pub(super) fn arg_unique(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .arg_unique()
        .map(|ok| ok.into_column())
}

pub(super) fn arg_min(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    Ok(s.as_materialized_series()
        .arg_min()
        .map_or(Scalar::null(IDX_DTYPE), |v| {
            Scalar::from(IdxSize::try_from(v).expect("idxsize"))
        })
        .into_column(s.name().clone()))
}

pub(super) fn arg_max(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    Ok(s.as_materialized_series()
        .arg_max()
        .map_or(Scalar::null(IDX_DTYPE), |v| {
            Scalar::from(IdxSize::try_from(v).expect("idxsize"))
        })
        .into_column(s.name().clone()))
}

pub(super) fn arg_sort(s: &Column, descending: bool, nulls_last: bool) -> PolarsResult<Column> {
    // @scalar-opt
    Ok(s.as_materialized_series()
        .arg_sort(SortOptions {
            descending,
            nulls_last,
            multithreaded: true,
            maintain_order: false,
            limit: None,
        })
        .into_column())
}

pub(super) fn min_by(s: &[Column]) -> PolarsResult<Column> {
    assert!(s.len() == 2);
    let input = &s[0];
    let by = &s[1];
    if input.len() != by.len() {
        polars_bail!(ShapeMismatch: "'by' column in `min_by` operation has incorrect length (got {}, expected {})", by.len(), input.len());
    }
    match by.as_materialized_series().arg_min() {
        Some(idx) => Ok(input.new_from_index(idx, 1)),
        None => Ok(Series::new_null(input.name().clone(), 1).into_column()),
    }
}

pub(super) fn max_by(s: &[Column]) -> PolarsResult<Column> {
    assert!(s.len() == 2);
    let input = &s[0];
    let by = &s[1];
    if input.len() != by.len() {
        polars_bail!(ShapeMismatch: "'by' column in `max_by` operation has incorrect length (got {}, expected {})", by.len(), input.len());
    }
    match by.as_materialized_series().arg_max() {
        Some(idx) => Ok(input.new_from_index(idx, 1)),
        None => Ok(Series::new_null(input.name().clone(), 1).into_column()),
    }
}

pub(super) fn product(s: &Column) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .product()
        .map(|sc| sc.into_column(s.name().clone()))
}

#[cfg(feature = "rank")]
pub(super) fn rank(s: &Column, options: RankOptions, seed: Option<u64>) -> PolarsResult<Column> {
    use polars_ops::series::SeriesRank;

    Ok(s.as_materialized_series().rank(options, seed).into_column())
}

#[cfg(feature = "hist")]
pub(super) fn hist(
    s: &[Column],
    bin_count: Option<usize>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Column> {
    let bins = if s.len() == 2 { Some(&s[1]) } else { None };
    let s = s[0].as_materialized_series();
    polars_ops::prelude::hist_series(
        s,
        bin_count,
        bins.map(|b| b.as_materialized_series().clone()),
        include_category,
        include_breakpoint,
    )
    .map(Column::from)
}

#[cfg(feature = "replace")]
pub(super) fn replace(s: &[Column]) -> PolarsResult<Column> {
    polars_ops::series::replace(s[0].as_materialized_series(), s[1].list()?, s[2].list()?)
        .map(Column::from)
}

#[cfg(feature = "replace")]
pub(super) fn replace_strict(s: &[Column], return_dtype: Option<DataType>) -> PolarsResult<Column> {
    match s.get(3) {
        Some(default) => polars_ops::series::replace_or_default(
            s[0].as_materialized_series(),
            s[1].list()?,
            s[2].list()?,
            default.as_materialized_series(),
            return_dtype,
        ),
        None => polars_ops::series::replace_strict(
            s[0].as_materialized_series(),
            s[1].list()?,
            s[2].list()?,
            return_dtype,
        ),
    }
    .map(Column::from)
}

pub(super) fn fill_null_with_strategy(
    s: &Column,
    strategy: FillNullStrategy,
) -> PolarsResult<Column> {
    s.fill_null(strategy)
}

pub(super) fn gather_every(s: &Column, n: usize, offset: usize) -> PolarsResult<Column> {
    s.gather_every(n, offset)
}

#[cfg(feature = "reinterpret")]
pub(super) fn reinterpret(s: &Column, signed: bool) -> PolarsResult<Column> {
    polars_ops::series::reinterpret(s.as_materialized_series(), signed).map(Column::from)
}

pub(super) fn negate(s: &Column) -> PolarsResult<Column> {
    polars_ops::series::negate(s.as_materialized_series()).map(Column::from)
}

pub(super) fn extend_constant(s: &[Column]) -> PolarsResult<Column> {
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

#[cfg(feature = "row_hash")]
pub(super) fn row_hash(c: &Column, k0: u64, k1: u64, k2: u64, k3: u64) -> PolarsResult<Column> {
    use std::hash::BuildHasher;

    use polars_utils::aliases::{
        PlFixedStateQuality, PlSeedableRandomStateQuality, SeedableFromU64SeedExt,
    };

    // TODO: don't expose all these seeds.
    let seed = PlFixedStateQuality::default().hash_one((k0, k1, k2, k3));

    // @scalar-opt
    Ok(c.as_materialized_series()
        .hash(PlSeedableRandomStateQuality::seed_from_u64(seed))
        .into_column())
}

#[cfg(feature = "arg_where")]
pub(super) fn arg_where(s: &mut [Column]) -> PolarsResult<Column> {
    use polars_core::utils::arrow::bitmap::utils::SlicesIterator;

    let predicate = s[0].bool()?;

    if predicate.is_empty() {
        Ok(Column::full_null(predicate.name().clone(), 0, &IDX_DTYPE))
    } else {
        use arrow::datatypes::IdxArr;
        use polars_core::prelude::IdxCa;

        let capacity = predicate.sum().unwrap();
        let mut out = Vec::with_capacity(capacity as usize);
        let mut total_offset = 0;

        predicate.downcast_iter().for_each(|arr| {
            let values = match arr.validity() {
                Some(validity) if validity.unset_bits() > 0 => validity & arr.values(),
                _ => arr.values().clone(),
            };

            for (offset, len) in SlicesIterator::new(&values) {
                // law of small numbers optimization
                if len == 1 {
                    out.push((total_offset + offset) as IdxSize)
                } else {
                    let offset = (offset + total_offset) as IdxSize;
                    let len = len as IdxSize;
                    let iter = offset..offset + len;
                    out.extend(iter)
                }
            }

            total_offset += arr.len();
        });
        let ca = IdxCa::with_chunk(predicate.name().clone(), IdxArr::from_vec(out));
        Ok(ca.into_column())
    }
}

#[cfg(feature = "index_of")]
/// Given two columns, find the index of a value (the second column) within the
/// first column. Will use binary search if possible, as an optimization.
pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Column> {
    use polars_core::series::IsSorted;
    use polars_ops::series::index_of as index_of_op;
    let series = if let Column::Scalar(ref sc) = s[0] {
        // We only care about the first value:
        &sc.as_single_value_series()
    } else {
        s[0].as_materialized_series()
    };

    let needle_s = &s[1];
    polars_ensure!(
        needle_s.len() == 1,
        InvalidOperation: "needle of `index_of` can only contain a single value, found {} values",
        needle_s.len()
    );
    let needle = Scalar::new(
        needle_s.dtype().clone(),
        needle_s.get(0).unwrap().into_static(),
    );

    let is_sorted_flag = series.is_sorted_flag();
    let result = match is_sorted_flag {
        // If the Series is sorted, we can use an optimized binary search to
        // find the value.
        IsSorted::Ascending | IsSorted::Descending if !needle.is_null() => {
            use polars_ops::series::SearchSortedSide;

            polars_ops::series::search_sorted(
                series,
                needle_s.as_materialized_series(),
                SearchSortedSide::Left,
                IsSorted::Descending == is_sorted_flag,
            )?
            .get(0)
            .and_then(|idx| {
                // search_sorted() gives an index even if it's not an exact
                // match! So we want to make sure it actually found the value.
                if series.get(idx as usize).ok()? == needle.as_any_value() {
                    Some(idx as usize)
                } else {
                    None
                }
            })
        },
        _ => index_of_op(series, needle)?,
    };

    let av = match result {
        None => AnyValue::Null,
        Some(idx) => AnyValue::from(idx as IdxSize),
    };
    let scalar = Scalar::new(IDX_DTYPE, av);
    Ok(Column::new_scalar(series.name().clone(), scalar, 1))
}

#[cfg(feature = "search_sorted")]
pub(super) fn search_sorted_impl(
    s: &mut [Column],
    side: polars_ops::series::SearchSortedSide,
    descending: bool,
) -> PolarsResult<Column> {
    let sorted_array = &s[0];
    let search_value = &s[1];

    polars_ops::series::search_sorted(
        sorted_array.as_materialized_series(),
        search_value.as_materialized_series(),
        side,
        descending,
    )
    .map(|ca| ca.into_column())
}

#[cfg(feature = "sign")]
pub(super) fn sign(s: &Column) -> PolarsResult<Column> {
    use num_traits::{One, Zero};
    use polars_core::prelude::{ChunkedArray, PolarsNumericType};
    use polars_core::with_match_physical_numeric_polars_type;

    fn sign_impl<T>(ca: &ChunkedArray<T>) -> Column
    where
        T: PolarsNumericType,
        ChunkedArray<T>: IntoColumn,
    {
        ca.apply_values(|x| {
            if x < T::Native::zero() {
                T::Native::zero() - T::Native::one()
            } else if x > T::Native::zero() {
                T::Native::one()
            } else {
                // Returning x here ensures we return NaN for NaN input, and
                // maintain the sign for signed zeroes (although we don't really
                // care about the latter).
                x
            }
        })
        .into_column()
    }

    let s = s.as_materialized_series();
    let dtype = s.dtype();
    use polars_core::datatypes::*;
    match dtype {
        _ if dtype.is_primitive_numeric() => with_match_physical_numeric_polars_type!(dtype, |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
            Ok(sign_impl(ca))
        }),
        DataType::Decimal(_, scale) => {
            use polars_core::prelude::ChunkApply;

            let ca = s.decimal()?;
            let out = ca
                .physical()
                .apply_values(|x| polars_compute::decimal::dec128_sign(x, *scale))
                .into_column();
            unsafe { out.from_physical_unchecked(dtype) }
        },
        _ => polars_bail!(opq = sign, dtype),
    }
}

pub(super) fn fill_null(s: &[Column]) -> PolarsResult<Column> {
    match (s[0].len(), s[1].len()) {
        (a, b) if a == b || b == 1 => {
            let series = s[0].clone();

            // Nothing to fill, so return early
            // this is done after casting as the output type must be correct
            if series.null_count() == 0 {
                return Ok(series);
            }

            let fill_value = s[1].clone();

            // default branch
            fn default(series: Column, fill_value: Column) -> PolarsResult<Column> {
                let mask = series.is_not_null();
                series.zip_with_same_type(&mask, &fill_value)
            }

            let fill_value = if series.dtype().is_categorical() && fill_value.dtype().is_string() {
                fill_value.cast(series.dtype()).unwrap()
            } else {
                fill_value
            };
            default(series, fill_value)
        },
        (1, other_len) => {
            if s[0].has_nulls() {
                Ok(s[1].clone())
            } else {
                Ok(s[0].new_from_index(0, other_len))
            }
        },
        (self_len, other_len) => polars_bail!(length_mismatch = "fill_null", self_len, other_len),
    }
}

pub(super) fn coalesce(s: &mut [Column]) -> PolarsResult<Column> {
    polars_ops::series::coalesce_columns(s)
}

pub(super) fn drop_nans(s: Column) -> PolarsResult<Column> {
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = s.f16()?;
            let mask = ca.is_not_nan() | ca.is_null();
            ca.filter(&mask).map(|ca| ca.into_column())
        },
        DataType::Float32 => {
            let ca = s.f32()?;
            let mask = ca.is_not_nan() | ca.is_null();
            ca.filter(&mask).map(|ca| ca.into_column())
        },
        DataType::Float64 => {
            let ca = s.f64()?;
            let mask = ca.is_not_nan() | ca.is_null();
            ca.filter(&mask).map(|ca| ca.into_column())
        },
        _ => Ok(s),
    }
}

#[cfg(feature = "round_series")]
pub(super) fn clip(s: &[Column], has_min: bool, has_max: bool) -> PolarsResult<Column> {
    match (has_min, has_max) {
        (true, true) => polars_ops::series::clip(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
            s[2].as_materialized_series(),
        ),
        (true, false) => polars_ops::series::clip_min(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
        ),
        (false, true) => polars_ops::series::clip_max(
            s[0].as_materialized_series(),
            s[1].as_materialized_series(),
        ),
        _ => unreachable!(),
    }
    .map(Column::from)
}

#[cfg(feature = "dtype-struct")]
pub fn as_struct(cols: &[Column]) -> PolarsResult<Column> {
    use polars_core::prelude::StructChunked;

    let Some(fst) = cols.first() else {
        polars_bail!(nyi = "turning no columns as_struct");
    };

    let mut min_length = usize::MAX;
    let mut max_length = usize::MIN;

    for col in cols {
        let len = col.len();

        min_length = min_length.min(len);
        max_length = max_length.max(len);
    }

    // @NOTE: Any additional errors should be handled by the StructChunked::from_columns
    let length = if min_length == 0 { 0 } else { max_length };

    Ok(StructChunked::from_columns(fst.name().clone(), length, cols)?.into_column())
}

#[cfg(feature = "log")]
pub(super) fn entropy(s: &Column, base: f64, normalize: bool) -> PolarsResult<Column> {
    use polars_ops::series::LogSeries;

    let out = s.as_materialized_series().entropy(base, normalize)?;
    if matches!(s.dtype(), DataType::Float32) {
        let out = out as f32;
        Ok(Column::new(s.name().clone(), [out]))
    } else {
        Ok(Column::new(s.name().clone(), [out]))
    }
}

#[cfg(feature = "log")]
pub(super) fn log(columns: &[Column]) -> PolarsResult<Column> {
    use polars_ops::series::LogSeries;

    assert_eq!(columns.len(), 2);
    Column::apply_broadcasting_binary_elementwise(&columns[0], &columns[1], Series::log)
}

#[cfg(feature = "log")]
pub(super) fn log1p(s: &Column) -> PolarsResult<Column> {
    use polars_ops::series::LogSeries;

    Ok(s.as_materialized_series().log1p().into())
}

#[cfg(feature = "log")]
pub(super) fn exp(s: &Column) -> PolarsResult<Column> {
    use polars_ops::series::LogSeries;

    Ok(s.as_materialized_series().exp().into())
}

pub(super) fn unique(s: &Column, stable: bool) -> PolarsResult<Column> {
    if stable {
        s.unique_stable()
    } else {
        s.unique()
    }
}

#[cfg(feature = "fused")]
pub(super) fn fused(input: &[Column], op: FusedOperator) -> PolarsResult<Column> {
    use polars_plan::plans::FusedOperator;

    let s0 = &input[0];
    let s1 = &input[1];
    let s2 = &input[2];
    match op {
        FusedOperator::MultiplyAdd => Ok(polars_ops::series::fma_columns(s0, s1, s2)),
        FusedOperator::SubMultiply => Ok(polars_ops::series::fsm_columns(s0, s1, s2)),
        FusedOperator::MultiplySub => Ok(polars_ops::series::fms_columns(s0, s1, s2)),
    }
}

pub(super) fn concat_expr(s: &[Column], rechunk: bool) -> PolarsResult<Column> {
    let mut first = s[0].clone();

    for s in &s[1..] {
        first.append(s)?;
    }
    if rechunk {
        first = first.rechunk()
    }
    Ok(first)
}

#[cfg(feature = "cov")]
pub(super) fn corr(s: &[Column], method: IRCorrelationMethod) -> PolarsResult<Column> {
    use polars_plan::plans::IRCorrelationMethod;

    fn covariance(s: &[Column], ddof: u8) -> PolarsResult<Column> {
        let a = &s[0];
        let b = &s[1];
        let name = PlSmallStr::from_static("cov");

        use polars_ops::chunked_array::cov::cov;
        let ret = match a.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                use num_traits::AsPrimitive;
                use polars_utils::float16::pf16;

                let ret =
                    cov(a.f16().unwrap(), b.f16().unwrap(), ddof).map(AsPrimitive::<pf16>::as_);
                return Ok(Column::new(name, &[ret]));
            },
            DataType::Float32 => {
                let ret = cov(a.f32().unwrap(), b.f32().unwrap(), ddof).map(|v| v as f32);
                return Ok(Column::new(name, &[ret]));
            },
            DataType::Float64 => cov(a.f64().unwrap(), b.f64().unwrap(), ddof),
            DataType::Int32 => cov(a.i32().unwrap(), b.i32().unwrap(), ddof),
            DataType::Int64 => cov(a.i64().unwrap(), b.i64().unwrap(), ddof),
            DataType::UInt32 => cov(a.u32().unwrap(), b.u32().unwrap(), ddof),
            DataType::UInt64 => cov(a.u64().unwrap(), b.u64().unwrap(), ddof),
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                cov(a.f64().unwrap(), b.f64().unwrap(), ddof)
            },
        };
        Ok(Column::new(name, &[ret]))
    }

    fn pearson_corr(s: &[Column]) -> PolarsResult<Column> {
        let a = &s[0];
        let b = &s[1];
        let name = PlSmallStr::from_static("pearson_corr");

        use polars_ops::chunked_array::cov::pearson_corr;
        let ret = match a.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                use num_traits::AsPrimitive;
                use polars_utils::float16::pf16;

                let ret =
                    pearson_corr(a.f16().unwrap(), b.f16().unwrap()).map(AsPrimitive::<pf16>::as_);
                return Ok(Column::new(name, &[ret]));
            },
            DataType::Float32 => {
                let ret = pearson_corr(a.f32().unwrap(), b.f32().unwrap()).map(|v| v as f32);
                return Ok(Column::new(name, &[ret]));
            },
            DataType::Float64 => pearson_corr(a.f64().unwrap(), b.f64().unwrap()),
            DataType::Int32 => pearson_corr(a.i32().unwrap(), b.i32().unwrap()),
            DataType::Int64 => pearson_corr(a.i64().unwrap(), b.i64().unwrap()),
            DataType::UInt32 => pearson_corr(a.u32().unwrap(), b.u32().unwrap()),
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                pearson_corr(a.f64().unwrap(), b.f64().unwrap())
            },
        };
        Ok(Column::new(name, &[ret]))
    }

    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
    fn spearman_rank_corr(s: &[Column], propagate_nans: bool) -> PolarsResult<Column> {
        use polars_core::utils::coalesce_nulls_columns;
        use polars_ops::chunked_array::nan_propagating_aggregate::nan_max_s;
        use polars_ops::series::{RankMethod, SeriesRank};
        let a = &s[0];
        let b = &s[1];

        let (a, b) = coalesce_nulls_columns(a, b);

        let name = PlSmallStr::from_static("spearman_rank_correlation");
        if propagate_nans && a.dtype().is_float() {
            for s in [&a, &b] {
                let max = nan_max_s(s.as_materialized_series(), PlSmallStr::EMPTY);
                if max.get(0).is_ok_and(|m| m.is_nan()) {
                    return Ok(Column::new(name, &[f64::NAN]));
                }
            }
        }

        // drop nulls so that they are excluded
        let a = a.drop_nulls();
        let b = b.drop_nulls();

        let a_rank = a.as_materialized_series().rank(
            RankOptions {
                method: RankMethod::Average,
                ..Default::default()
            },
            None,
        );
        let b_rank = b.as_materialized_series().rank(
            RankOptions {
                method: RankMethod::Average,
                ..Default::default()
            },
            None,
        );

        // Because rank results in f64, we may need to restore the dtype
        let a_rank = if a.dtype().is_float() {
            a_rank.cast(a.dtype())?.into()
        } else {
            a_rank.into()
        };
        let b_rank = if b.dtype().is_float() {
            b_rank.cast(b.dtype())?.into()
        } else {
            b_rank.into()
        };

        pearson_corr(&[a_rank, b_rank])
    }

    polars_ensure!(
        s[0].len() == s[1].len() || s[0].len() == 1 || s[1].len() == 1,
        length_mismatch = "corr",
        s[0].len(),
        s[1].len()
    );

    match method {
        IRCorrelationMethod::Pearson => pearson_corr(s),
        #[cfg(all(feature = "rank", feature = "propagate_nans"))]
        IRCorrelationMethod::SpearmanRank(propagate_nans) => spearman_rank_corr(s, propagate_nans),
        IRCorrelationMethod::Covariance(ddof) => covariance(s, ddof),
    }
}

#[cfg(feature = "peaks")]
pub(super) fn peak_min(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::peaks::peak_min_max(s, &AnyValue::Int8(0), &AnyValue::Int8(0), false)
        .map(IntoColumn::into_column)
}

#[cfg(feature = "peaks")]
pub(super) fn peak_max(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::peaks::peak_min_max(s, &AnyValue::Int8(0), &AnyValue::Int8(0), true)
        .map(IntoColumn::into_column)
}

#[cfg(feature = "cutqcut")]
pub(super) fn cut(
    s: &Column,
    breaks: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Column> {
    polars_ops::prelude::cut(
        s.as_materialized_series(),
        breaks,
        labels,
        left_closed,
        include_breaks,
    )
    .map(Column::from)
}

#[cfg(feature = "cutqcut")]
pub(super) fn qcut(
    s: &Column,
    probs: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    allow_duplicates: bool,
    include_breaks: bool,
) -> PolarsResult<Column> {
    polars_ops::prelude::qcut(
        s.as_materialized_series(),
        probs,
        labels,
        left_closed,
        allow_duplicates,
        include_breaks,
    )
    .map(Column::from)
}

#[cfg(feature = "ewma")]
pub(super) fn ewm_mean(
    s: &Column,
    options: polars_ops::series::EWMOptions,
) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_mean(s.as_materialized_series(), options).map(Column::from)
}

#[cfg(feature = "ewma")]
pub(super) fn ewm_std(s: &Column, options: polars_ops::series::EWMOptions) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_std(s.as_materialized_series(), options).map(Column::from)
}

#[cfg(feature = "ewma")]
pub(super) fn ewm_var(s: &Column, options: polars_ops::series::EWMOptions) -> PolarsResult<Column> {
    polars_ops::prelude::ewm_var(s.as_materialized_series(), options).map(Column::from)
}

#[cfg(feature = "ewma_by")]
pub(super) fn ewm_mean_by(s: &[Column], half_life: polars_time::Duration) -> PolarsResult<Column> {
    use polars_ops::series::SeriesMethods;

    let time_zone = match s[1].dtype() {
        DataType::Datetime(_, Some(time_zone)) => Some(time_zone),
        _ => None,
    };
    polars_ensure!(!half_life.negative(), InvalidOperation: "half_life cannot be negative");
    polars_time::prelude::ensure_is_constant_duration(half_life, time_zone, "half_life")?;
    // `half_life` is a constant duration so we can safely use `duration_ns()`.
    let half_life = half_life.duration_ns();
    let values = &s[0];
    let times = &s[1];
    let times_is_sorted = times
        .as_materialized_series()
        .is_sorted(Default::default())?;
    polars_ops::prelude::ewm_mean_by(
        values.as_materialized_series(),
        times.as_materialized_series(),
        half_life,
        times_is_sorted,
    )
    .map(Column::from)
}

pub fn row_encode(
    c: &mut [Column],
    dts: Vec<DataType>,
    variant: RowEncodingVariant,
) -> PolarsResult<Column> {
    assert_eq!(c.len(), dts.len());

    // We need to make sure that the output types are correct or we will get wrong results or even
    // segfaults when decoding.
    for (dt, c) in dts.iter().zip(c.iter_mut()) {
        if c.dtype().matches_schema_type(dt)? {
            *c = c.cast(dt)?;
        }
    }

    let name = PlSmallStr::from_static("row_encoded");
    match variant {
        RowEncodingVariant::Unordered => _get_rows_encoded_ca_unordered(name, c),
        RowEncodingVariant::Ordered {
            descending,
            nulls_last,
            broadcast_nulls,
        } => {
            let descending = descending.unwrap_or_else(|| vec![false; c.len()]);
            let nulls_last = nulls_last.unwrap_or_else(|| vec![false; c.len()]);
            let broadcast_nulls = broadcast_nulls.unwrap_or(false);

            assert_eq!(c.len(), descending.len());
            assert_eq!(c.len(), nulls_last.len());

            _get_rows_encoded_ca(name, c, &descending, &nulls_last, broadcast_nulls)
        },
    }
    .map(IntoColumn::into_column)
}

#[cfg(feature = "dtype-struct")]
pub fn row_decode(
    c: &mut [Column],
    fields: Vec<Field>,
    variant: RowEncodingVariant,
) -> PolarsResult<Column> {
    use polars_core::prelude::row_encode::row_encoding_decode;

    assert_eq!(c.len(), 1);
    let ca = c[0].binary_offset()?;

    let mut opts = Vec::with_capacity(fields.len());
    match variant {
        RowEncodingVariant::Unordered => opts.extend(std::iter::repeat_n(
            RowEncodingOptions::new_unsorted(),
            fields.len(),
        )),
        RowEncodingVariant::Ordered {
            descending,
            nulls_last,
            broadcast_nulls,
        } => {
            let descending = descending.unwrap_or_else(|| vec![false; fields.len()]);
            let nulls_last = nulls_last.unwrap_or_else(|| vec![false; fields.len()]);
            if broadcast_nulls.is_some() {
                polars_bail!(InvalidOperation: "broadcast_nulls is not supported for row_decode.");
            }

            assert_eq!(fields.len(), descending.len());
            assert_eq!(fields.len(), nulls_last.len());

            opts.extend(
                descending
                    .into_iter()
                    .zip(nulls_last)
                    .map(|(d, n)| RowEncodingOptions::new_sorted(d, n)),
            )
        },
    }

    row_encoding_decode(ca, &fields, &opts).map(IntoColumn::into_column)
}

pub fn repeat(args: &[Column]) -> PolarsResult<Column> {
    let c = &args[0];
    let n = &args[1];

    polars_ensure!(
        n.dtype().is_integer(),
        SchemaMismatch: "expected expression of dtype 'integer', got '{}'", n.dtype()
    );

    let first_value = n.get(0)?;
    let n = first_value.extract::<usize>().ok_or_else(
        || polars_err!(ComputeError: "could not parse value '{}' as a size.", first_value),
    )?;

    Ok(c.new_from_index(0, n))
}

pub fn dynamic_pred(columns: &[Column], pred: &DynamicPred) -> PolarsResult<Column> {
    pred.evaluate(columns)
}
