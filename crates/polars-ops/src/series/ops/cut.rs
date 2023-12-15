use std::cmp::PartialOrd;
use std::iter::once;

use polars_core::prelude::*;

fn map_cats(
    s: &Series,
    cutlabs: &[String],
    sorted_breaks: &[f64],
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let cl: Vec<&str> = cutlabs.iter().map(String::as_str).collect();

    let out_name = format!("{}_bin", s.name());
    let mut bld = CategoricalChunkedBuilder::new(&out_name, s.len(), Default::default());
    let s2 = s.cast(&DataType::Float64)?;
    // It would be nice to parallelize this
    let s_iter = s2.f64()?.into_iter();

    let op = if left_closed {
        PartialOrd::ge
    } else {
        PartialOrd::gt
    };

    if include_breaks {
        // This is to replicate the behavior of the old buggy version that only worked on series and
        // returned a dataframe. That included a column of the right endpoint of the interval. So we
        // return a struct series instead which can be turned into a dataframe later.
        let right_ends = [sorted_breaks, &[f64::INFINITY]].concat();
        let mut brk_vals = PrimitiveChunkedBuilder::<Float64Type>::new("brk", s.len());
        s_iter
            .map(|opt| {
                opt.filter(|x| !x.is_nan())
                    .map(|x| sorted_breaks.partition_point(|v| op(&x, v)))
            })
            .for_each(|idx| match idx {
                None => {
                    bld.append_null();
                    brk_vals.append_null();
                },
                Some(idx) => unsafe {
                    bld.append_value(cl.get_unchecked(idx));
                    brk_vals.append_value(*right_ends.get_unchecked(idx));
                },
            });

        let outvals = vec![brk_vals.finish().into_series(), bld.finish().into_series()];
        Ok(StructChunked::new(&out_name, &outvals)?.into_series())
    } else {
        bld.drain_iter(s_iter.map(|opt| {
            opt.filter(|x| !x.is_nan())
                .map(|x| unsafe { *cl.get_unchecked(sorted_breaks.partition_point(|v| op(&x, v))) })
        }));
        Ok(bld.finish().into_series())
    }
}

pub fn cut(
    s: &Series,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    polars_ensure!(!breaks.iter().any(|x| x.is_nan()), ComputeError: "Breaks cannot be NaN");
    // Breaks must be sorted to cut inputs properly.
    let mut breaks = breaks;
    let sorted_breaks = breaks.as_mut_slice();
    sorted_breaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    polars_ensure!(sorted_breaks.windows(2).all(|x| x[0] != x[1]), Duplicate: "Breaks are not unique");
    if !sorted_breaks.is_empty() {
        polars_ensure!(sorted_breaks[0] > f64::NEG_INFINITY, ComputeError: "Don't include -inf in breaks");
        polars_ensure!(sorted_breaks[sorted_breaks.len() - 1] < f64::INFINITY, ComputeError: "Don't include inf in breaks");
    }

    let cutlabs = match labels {
        Some(ll) => {
            polars_ensure!(ll.len() == sorted_breaks.len() + 1, ShapeMismatch: "Provide nbreaks + 1 labels");
            ll
        },
        None => (once(&f64::NEG_INFINITY).chain(sorted_breaks.iter()))
            .zip(sorted_breaks.iter().chain(once(&f64::INFINITY)))
            .map(|v| {
                if left_closed {
                    format!("[{}, {})", v.0, v.1)
                } else {
                    format!("({}, {}]", v.0, v.1)
                }
            })
            .collect::<Vec<String>>(),
    };

    map_cats(s, &cutlabs, sorted_breaks, left_closed, include_breaks)
}

pub fn qcut(
    s: &Series,
    probs: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
    allow_duplicates: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let s = s.cast(&DataType::Float64)?;
    let s2 = s.sort(false).unwrap();
    let ca = s2.f64()?;
    let f = |&p| {
        ca.quantile(p, QuantileInterpolOptions::Linear)
            .unwrap()
            .unwrap()
    };
    let mut qbreaks: Vec<_> = probs.iter().map(f).collect();
    qbreaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // When probs are spaced too closely for the number of repeated values in the distribution
    // some quantiles may be duplicated. The only thing to do if we want to go on, is to drop
    // the repeated values and live with some bins being larger than intended.
    if allow_duplicates {
        let lfilt = match labels {
            None => None,
            Some(ll) => {
                polars_ensure!(ll.len() == qbreaks.len() + 1,
                    ShapeMismatch: "Wrong number of labels");
                let blen = ll.len();
                Some(
                    ll.into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i == 0 || *i == blen - 1 || qbreaks[*i] != qbreaks[i - 1])
                        .unzip::<_, _, Vec<_>, Vec<_>>()
                        .1,
                )
            },
        };
        qbreaks.dedup();
        return cut(&s, qbreaks, lfilt, left_closed, include_breaks);
    }
    cut(&s, qbreaks, labels, left_closed, include_breaks)
}
