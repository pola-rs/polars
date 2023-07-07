use std::{iter::once, cmp::PartialOrd};

use polars_core::prelude::*;
use polars_utils::aliases::PlHashMap;

fn map_cats(
    s: &Series,
    cutlabs: &Vec<String>,
    sorted_breaks: &[f64],
    left_closed: bool,
    breaks_struct: bool
) -> PolarsResult<Series> {
    let cl: Vec<&str> = cutlabs.iter().map(String::as_str).collect();

    let out_name = format!("{}_bin", s.name());
    let mut bld = CategoricalChunkedBuilder::new(&out_name, s.len());
    let s2 = s.cast(&DataType::Float64);
    //let s_iter = s2?.f64()?.into_iter();

    let op = if left_closed { PartialOrd::ge } else { PartialOrd::gt };
    unsafe {
        bld.drain_iter(s2?.f64()?.into_iter().map(|opt| {
            opt
                .filter(|x| !x.is_nan())
                .map(|x| *cl.get_unchecked(sorted_breaks.partition_point(|v| op(&x, v))))
        }));
    }
    let res = bld.finish();
    if breaks_struct {
        let mut mapper = PlHashMap::<&str, f64>::with_capacity(cl.len());
        mapper.extend(cl.iter().zip(sorted_breaks.iter().chain(once(&f64::INFINITY))));
        let brk_vals = res
            .iter_str()
            .map(|k| k.map(|l| mapper[l])).collect::<Float64Chunked>();
        let outvals = vec![res.into_series(), brk_vals.into_series()];
        Ok(StructChunked::new("cut", &outvals)?.into_series())
    } else {
        Ok(res.into_series())
    }
}

// fn cut_with_breaks() {

// }

pub fn cut(
    s: &Series,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
) -> PolarsResult<Series> {
    polars_ensure!(!breaks.is_empty(), ShapeMismatch: "Breaks are empty");
    polars_ensure!(!breaks.iter().any(|x| x.is_nan()), ComputeError: "Breaks cannot be NaN");
    // Breaks must be sorted to cut inputs properly.
    let mut breaks = breaks;
    let sorted_breaks = breaks.as_mut_slice();
    sorted_breaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    polars_ensure!(sorted_breaks.windows(2).all(|x| x[0] != x[1]), Duplicate: "Breaks are not unique");

    polars_ensure!(sorted_breaks[0] > f64::NEG_INFINITY, ComputeError: "Don't include -inf in breaks");
    polars_ensure!(sorted_breaks[sorted_breaks.len() - 1] < f64::INFINITY, ComputeError: "Don't include inf in breaks");

    let cutlabs = match labels {
        Some(ll) => {
            polars_ensure!(ll.len() == sorted_breaks.len() + 1, ShapeMismatch: "Provide nbreaks + 1 labels");
            ll
        }
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


    map_cats(s, &cutlabs, sorted_breaks, left_closed, true)
}

pub fn qcut(
    s: &Series,
    probs: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
    allow_duplicates: bool,
) -> PolarsResult<Series> {
    let s = s.cast(&DataType::Float64)?;
    let s2 = s.sort(false);
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
                        .filter(|(i, _)| *i == 0 || *i == blen || qbreaks[*i] != qbreaks[i - 1])
                        .unzip::<_, _, Vec<_>, Vec<_>>()
                        .1,
                )
            }
        };
        qbreaks.dedup();
        return cut(&s, qbreaks, lfilt, left_closed);
    }
    cut(&s, qbreaks, labels, left_closed)
}
