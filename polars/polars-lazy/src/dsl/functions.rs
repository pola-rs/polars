//! # Functions
//!
//! Functions on expressions that might be useful.
//!
use crate::prelude::*;
use crate::utils::has_wildcard;
use polars_core::prelude::*;
use polars_core::utils::get_supertype;
use rayon::prelude::*;
use std::ops::{BitAnd, BitOr};

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr) -> Expr {
    let name = "cov";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "cov";
        options
    })
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr(a: Expr, b: Expr) -> Expr {
    let name = "pearson_corr";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_f(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_f(ca_a, ca_b)])
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_i(ca_a, ca_b)])
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_i(ca_a, ca_b)])
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_i(ca_a, ca_b)])
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_i(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr_f(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "pearson_corr";
        options
    })
}

/// Compute the spearman rank correlation between two columns.
#[cfg(feature = "rank")]
#[cfg_attr(docsrs, doc(cfg(feature = "rank")))]
pub fn spearman_rank_corr(a: Expr, b: Expr) -> Expr {
    pearson_corr(
        a.rank(RankOptions {
            method: RankMethod::Min,
            ..Default::default()
        }),
        b.rank(RankOptions {
            method: RankMethod::Min,
            ..Default::default()
        }),
    )
    .with_fmt("spearman_rank_correlation")
}

/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
pub fn argsort_by<E: AsRef<[Expr]>>(by: E, reverse: &[bool]) -> Expr {
    let reverse = reverse.to_vec();
    let function = NoEq::new(Arc::new(move |by: &mut [Series]| {
        polars_core::functions::argsort_by(by, &reverse).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);

    Expr::Function {
        input: by.as_ref().to_vec(),
        function,
        output_type: GetOutput::from_type(DataType::UInt32),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: false,
            auto_explode: true,
            fmt_str: "argsort_by",
        },
    }
}

#[cfg(feature = "concat_str")]
#[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
/// Horizontally concat string columns in linear time
pub fn concat_str(s: Vec<Expr>, sep: &str) -> Expr {
    let sep = sep.to_string();
    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        polars_core::functions::concat_str(s, &sep).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: s,
        function,
        output_type: GetOutput::from_type(DataType::Utf8),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: true,
            auto_explode: true,
            fmt_str: "concat_by",
        },
    }
}

/// Concat lists entries.
#[cfg(feature = "list")]
#[cfg_attr(docsrs, doc(cfg(feature = "list")))]
pub fn concat_lst(s: Vec<Expr>) -> Expr {
    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        let mut first = std::mem::take(&mut s[0]);
        let other = &s[1..];

        let first_ca = match first.list().ok() {
            Some(ca) => ca,
            None => {
                first = first.reshape(&[-1, 1]).unwrap();
                first.list().unwrap()
            }
        };
        first_ca.lst_concat(other).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: s,
        function,
        output_type: GetOutput::from_type(DataType::Utf8),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            auto_explode: false,
            fmt_str: "concat_list",
        },
    }
}

/// Create list entries that are range arrays
/// - if `low` and `high` are a column, every element will expand into an array in a list column.
/// - if `low` and `high` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
#[cfg_attr(docsrs, doc(cfg(feature = "arange")))]
pub fn arange(low: Expr, high: Expr, step: usize) -> Expr {
    if (matches!(low, Expr::Literal(_)) && !matches!(low, Expr::Literal(LiteralValue::Series(_))))
        || matches!(high, Expr::Literal(_))
            && !matches!(high, Expr::Literal(LiteralValue::Series(_)))
    {
        let f = move |sa: Series, sb: Series| {
            let sa = sa.cast(&DataType::Int64)?;
            let sb = sb.cast(&DataType::Int64)?;
            let low = sa
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `low` evaluation".into()))?;
            let high = sb
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `high` evaluation".into()))?;

            if step > 1 {
                Ok(
                    Int64Chunked::from_iter_values("arange", (low..high).step_by(step))
                        .into_series(),
                )
            } else {
                Ok(Int64Chunked::from_iter_values("arange", low..high).into_series())
            }
        };
        map_binary(
            low,
            high,
            f,
            GetOutput::map_field(|_| Field::new("arange", DataType::Int64)),
        )
    } else {
        let f = move |sa: Series, sb: Series| {
            let sa = sa.cast(&DataType::Int64)?;
            let sb = sb.cast(&DataType::Int64)?;
            let low = sa.i64()?;
            let high = sb.i64()?;
            let mut builder = ListPrimitiveChunkedBuilder::<i64>::new(
                "arange",
                low.len(),
                low.len() * 3,
                DataType::Int64,
            );

            low.into_iter()
                .zip(high.into_iter())
                .for_each(|(opt_l, opt_h)| match (opt_l, opt_h) {
                    (Some(l), Some(r)) => {
                        if step > 1 {
                            builder.append_iter_values((l..r).step_by(step));
                        } else {
                            builder.append_iter_values(l..r);
                        }
                    }
                    _ => builder.append_null(),
                });

            Ok(builder.finish().into_series())
        };
        map_binary(
            low,
            high,
            f,
            GetOutput::map_field(|_| Field::new("arange", DataType::List(DataType::Int64.into()))),
        )
    }
}

#[cfg(feature = "temporal")]
pub fn datetime(
    year: Expr,
    month: Expr,
    day: Expr,
    hour: Option<Expr>,
    minute: Option<Expr>,
    second: Option<Expr>,
    millisecond: Option<Expr>,
) -> Expr {
    use polars_core::export::chrono::NaiveDate;
    use polars_core::utils::CustomIterTools;

    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 7);
        let max_len = s.iter().map(|s| s.len()).max().unwrap();
        let mut year = s[0].cast(&DataType::Int32)?;
        if year.len() < max_len {
            year = year.expand_at_index(0, max_len)
        }
        let year = year.i32()?;
        let mut month = s[1].cast(&DataType::UInt32)?;
        if month.len() < max_len {
            month = month.expand_at_index(0, max_len);
        }
        let month = month.u32()?;
        let mut day = s[2].cast(&DataType::UInt32)?;
        if day.len() < max_len {
            day = day.expand_at_index(0, max_len);
        }
        let day = day.u32()?;
        let mut hour = s[3].cast(&DataType::UInt32)?;
        if hour.len() < max_len {
            hour = hour.expand_at_index(0, max_len);
        }
        let hour = hour.u32()?;

        let mut minute = s[4].cast(&DataType::UInt32)?;
        if minute.len() < max_len {
            minute = minute.expand_at_index(0, max_len);
        }
        let minute = minute.u32()?;

        let mut second = s[5].cast(&DataType::UInt32)?;
        if second.len() < max_len {
            second = second.expand_at_index(0, max_len);
        }
        let second = second.u32()?;

        let mut millisecond = s[6].cast(&DataType::UInt32)?;
        if millisecond.len() < max_len {
            millisecond = millisecond.expand_at_index(0, max_len);
        }
        let millisecond = millisecond.u32()?;

        let ca: Int64Chunked = year
            .into_iter()
            .zip(month.into_iter())
            .zip(day.into_iter())
            .zip(hour.into_iter())
            .zip(minute.into_iter())
            .zip(second.into_iter())
            .zip(millisecond.into_iter())
            .map(|((((((y, m), d), h), mnt), s), ms)| {
                if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(ms)) =
                    (y, m, d, h, mnt, s, ms)
                {
                    Some(
                        NaiveDate::from_ymd(y, m, d)
                            .and_hms_milli(h, mnt, s, ms)
                            .timestamp_millis(),
                    )
                } else {
                    None
                }
            })
            .collect_trusted();

        Ok(ca.into_datetime(TimeUnit::Milliseconds, None).into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: vec![
            year,
            month,
            day,
            hour.unwrap_or_else(|| lit(0)),
            minute.unwrap_or_else(|| lit(0)),
            second.unwrap_or_else(|| lit(0)),
            millisecond.unwrap_or_else(|| lit(0)),
        ],
        function,
        output_type: GetOutput::from_type(DataType::Datetime(TimeUnit::Milliseconds, None)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            auto_explode: false,
            fmt_str: "datetime",
        },
    }
    .alias("datetime")
}

/// Concat multiple
pub fn concat<L: AsRef<[LazyFrame]>>(inputs: L, rechunk: bool) -> Result<LazyFrame> {
    let mut inputs = inputs.as_ref().to_vec();
    let lf = std::mem::take(
        inputs
            .get_mut(0)
            .ok_or_else(|| PolarsError::ComputeError("empty container given".into()))?,
    );
    let opt_state = lf.opt_state;
    let mut lps = Vec::with_capacity(inputs.len());
    lps.push(lf.logical_plan);

    for lf in &mut inputs[1..] {
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp)
    }

    let lp = LogicalPlan::Union {
        inputs: lps,
        options: Default::default(),
    };
    let mut lf = LazyFrame::from(lp);
    lf.opt_state = opt_state;

    if rechunk {
        Ok(lf.map(
            |mut df: DataFrame| {
                df.rechunk();
                Ok(df)
            },
            Some(AllowedOptimizations::default()),
            None,
            Some("RECHUNK"),
        ))
    } else {
        Ok(lf)
    }
}

/// Collect all `LazyFrame` computations.
pub fn collect_all<I>(lfs: I) -> Result<Vec<DataFrame>>
where
    I: IntoParallelIterator<Item = LazyFrame>,
{
    let iter = lfs.into_par_iter();

    polars_core::POOL.install(|| iter.map(|lf| lf.collect()).collect())
}

/// Create a Column Expression based on a column name.
///
/// # Arguments
///
/// * `name` - A string slice that holds the name of the column
///
/// # Examples
///
/// ```ignore
/// // select a column name
/// col("foo")
/// ```
///
/// ```ignore
/// // select all columns by using a wildcard
/// col("*")
/// ```
///
/// ```ignore
/// // select specific column by writing a regular expression that starts with `^` and ends with `$`
/// // only if regex features is activated
/// col("^foo.*$")
/// ```
pub fn col(name: &str) -> Expr {
    match name {
        "*" => Expr::Wildcard,
        _ => Expr::Column(Arc::from(name)),
    }
}

/// Selects all columns
pub fn all() -> Expr {
    Expr::Wildcard
}

/// Select multiple columns by name
pub fn cols<I: IntoVec<String>>(names: I) -> Expr {
    let names = names.into_vec();
    Expr::Columns(names)
}

/// Select multiple columns by dtype.
pub fn dtype_col(dtype: &DataType) -> Expr {
    Expr::DtypeColumn(vec![dtype.clone()])
}

/// Select multiple columns by dtype.
pub fn dtype_cols<DT: AsRef<[DataType]>>(dtype: DT) -> Expr {
    let dtypes = dtype.as_ref().to_vec();
    Expr::DtypeColumn(dtypes)
}

/// Sum all the values in this Expression.
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

/// Find the minimum of all the values in this Expression.
pub fn min(name: &str) -> Expr {
    col(name).min()
}

/// Find the maximum of all the values in this Expression.
pub fn max(name: &str) -> Expr {
    col(name).max()
}

/// Find the mean of all the values in this Expression.
pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

/// Find the mean of all the values in this Expression.
pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

/// Find the median of all the values in this Expression.
pub fn median(name: &str) -> Expr {
    col(name).median()
}

/// Find a specific quantile of all the values in this Expression.
pub fn quantile(name: &str, quantile: f64, interpol: QuantileInterpolOptions) -> Expr {
    col(name).quantile(quantile, interpol)
}

macro_rules! prepare_binary_function {
    ($f:ident) => {
        move |s: &mut [Series]| {
            let s0 = std::mem::take(&mut s[0]);
            let s1 = std::mem::take(&mut s[1]);

            $f(s0, s1)
        }
    };
}

/// Apply a closure on the two columns that are evaluated from `Expr` a and `Expr` b.
pub fn map_binary<F: 'static>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.map_many(function, &[b], output_type)
}

pub fn apply_binary<F: 'static>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.apply_many(function, &[b], output_type)
}

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<F: 'static, E: AsRef<[Expr]>>(mut acc: Expr, f: F, exprs: E) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync + Clone,
{
    let mut exprs = exprs.as_ref().to_vec();
    if exprs.iter().any(has_wildcard) {
        exprs.push(acc);

        let function = NoEq::new(Arc::new(move |series: &mut [Series]| {
            let mut series = series.to_vec();
            let mut acc = series.pop().unwrap();

            for s in series {
                acc = f(acc, s)?;
            }
            Ok(acc)
        }) as Arc<dyn SeriesUdf>);

        // Todo! make sure that output type is correct
        Expr::Function {
            input: exprs,
            function,
            output_type: GetOutput::same_type(),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "",
            },
        }
    } else {
        for e in exprs {
            acc = map_binary(
                acc,
                e,
                f.clone(),
                GetOutput::map_dtypes(|dt| get_supertype(dt[0], dt[1]).unwrap()),
            );
        }
        acc
    }
}

/// Get the the sum of the values per row
pub fn sum_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1, s2| Ok(&s1 + &s2);
    fold_exprs(lit(0), func, exprs)
}

/// Get the the maximum value per row
pub fn max_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    max_exprs_impl(exprs)
}

fn max_exprs_impl(mut exprs: Vec<Expr>) -> Expr {
    if exprs.len() == 1 {
        return std::mem::take(&mut exprs[0]);
    }

    let first = std::mem::take(&mut exprs[0]);
    first
        .map_many(
            |s| {
                let s = s.to_vec();
                let df = DataFrame::new_no_checks(s);
                df.hmax().map(|s| s.unwrap())
            },
            &exprs[1..],
            GetOutput::super_type(),
        )
        .alias("max")
}

/// Get the the minimum value per row
pub fn min_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    min_exprs_impl(exprs)
}

fn min_exprs_impl(mut exprs: Vec<Expr>) -> Expr {
    if exprs.len() == 1 {
        return std::mem::take(&mut exprs[0]);
    }

    let first = std::mem::take(&mut exprs[0]);
    first
        .map_many(
            |s| {
                let s = s.to_vec();
                let df = DataFrame::new_no_checks(s);
                df.hmin().map(|s| s.unwrap())
            },
            &exprs[1..],
            GetOutput::super_type(),
        )
        .alias("min")
}

/// Evaluate all the expressions with a bitwise or
pub fn any_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(s1.bool()?.bitor(s2.bool()?).into_series());
    fold_exprs(lit(false), func, exprs)
}

/// Evaluate all the expressions with a bitwise and
pub fn all_exprs<E: AsRef<[Expr]>>(exprs: E) -> Expr {
    let exprs = exprs.as_ref().to_vec();
    let func = |s1: Series, s2: Series| Ok(s1.bool()?.bitand(s2.bool()?).into_series());
    fold_exprs(lit(true), func, exprs)
}

/// [Not](Expr::Not) expression.
pub fn not(expr: Expr) -> Expr {
    Expr::Not(Box::new(expr))
}

/// [IsNull](Expr::IsNotNull) expression
pub fn is_null(expr: Expr) -> Expr {
    Expr::IsNull(Box::new(expr))
}

/// [IsNotNull](Expr::IsNotNull) expression.
pub fn is_not_null(expr: Expr) -> Expr {
    Expr::IsNotNull(Box::new(expr))
}

/// [Cast](Expr::Cast) expression.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
        strict: false,
    }
}

pub trait Range<T> {
    fn into_range(self, high: T) -> Expr;
}

macro_rules! impl_into_range {
    ($dt: ty) => {
        impl Range<$dt> for $dt {
            fn into_range(self, high: $dt) -> Expr {
                Expr::Literal(LiteralValue::Range {
                    low: self as i64,
                    high: high as i64,
                    data_type: DataType::Int32,
                })
            }
        }
    };
}

impl_into_range!(i32);
impl_into_range!(i64);
impl_into_range!(u32);

/// Create a range literal.
pub fn range<T: Range<T>>(low: T, high: T) -> Expr {
    low.into_range(high)
}

/// Take several expressions and collect them into a [`StructChunked`].
#[cfg(feature = "dtype-struct")]
pub fn as_struct(exprs: &[Expr]) -> Expr {
    map_multiple(
        |s| StructChunked::new("", s).map(|ca| ca.into_series()),
        exprs,
        GetOutput::map_fields(|fld| Field::new("", DataType::Struct(fld.to_vec()))),
    )
    .with_function_options(|mut options| {
        options.input_wildcard_expansion = true;
        options.fmt_str = "as_struct";
        options
    })
}
