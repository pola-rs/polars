//! # Functions
//!
//! Functions on expressions that might be useful.
//!
use crate::prelude::*;
use polars_core::prelude::*;

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr) -> Expr {
    let name = "cov";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    map_binary(a, b, function, Some(Field::new(name, DataType::Float32))).alias(name)
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr(a: Expr, b: Expr) -> Expr {
    let name = "pearson_corr";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    map_binary(a, b, function, Some(Field::new(name, DataType::Float32))).alias(name)
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
    .alias("spearman_rank_corr")
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
        },
    }
}

#[cfg(feature = "concat_str")]
#[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
/// Concat string columns in linear time
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
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
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
        },
    }
}

/// Create list entries that are range arrays
/// - if `low` and `high` are a column, every element will expand into an array in a list column.
/// - if `low` and `high` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
#[cfg_attr(docsrs, doc(cfg(feature = "arange")))]
pub fn arange(low: Expr, high: Expr, step: usize) -> Expr {
    if matches!(low, Expr::Literal(_)) || matches!(high, Expr::Literal(_)) {
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
                Ok(Int64Chunked::new_from_iter("arange", (low..high).step_by(step)).into_series())
            } else {
                Ok(Int64Chunked::new_from_iter("arange", low..high).into_series())
            }
        };
        map_binary(low, high, f, Some(Field::new("arange", DataType::Int64)))
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
            Some(Field::new("arange", DataType::List(DataType::Int64.into()))),
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
                            .timestamp_nanos(),
                    )
                } else {
                    None
                }
            })
            .trust_my_length(max_len)
            .collect_trusted();

        Ok(ca.into_date().into_series())
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
        output_type: GetOutput::from_type(DataType::Datetime),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
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
            .ok_or_else(|| PolarsError::ValueError("empty container given".into()))?,
    );
    let opt_state = lf.opt_state;
    let mut lps = Vec::with_capacity(inputs.len());
    lps.push(lf.logical_plan);

    for lf in &mut inputs[1..] {
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp)
    }

    let lp = LogicalPlan::Union { inputs: lps };
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
        ))
    } else {
        Ok(lf)
    }
}
