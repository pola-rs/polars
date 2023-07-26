use std::cmp::PartialOrd;
use std::iter::once;

use polars_core::export::regex::Regex;
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
    let mut bld = CategoricalChunkedBuilder::new(&out_name, s.len());
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
                }
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

const MAX_PRECISION: usize = 32;

fn find_label_precision(sorted_breaks: &[f64], precision: usize, scientific: bool) -> usize {
    // Find a level of precision such that the numbers printed in labels are distinct.
    if sorted_breaks.is_empty() {
        return precision;
    }
    // The following finds a level of precision such that every break has a unique string
    // representation. While a little half-assed, it's also what everyone else does. We could
    // possibly speed this up with "clever" floating point operations, but that's more complex and
    // error prone. And unless you have millions of categories it isn't remotely an issue.
    let format_fn = match scientific {
        false => |b, p| format!("{0:.1$}", b, p),
        true => |b, p| format!("{0:.1$e}", b, p),
    };
    let mut precision_out = precision;
    'outer: while precision_out < MAX_PRECISION {
        let mut last_str = format_fn(sorted_breaks[0], precision_out);
        for b in sorted_breaks.iter().skip(1) {
            let b_str: String = format_fn(*b, precision_out);
            if b_str == last_str {
                precision_out += 1;
                continue 'outer;
            }
            last_str = b_str;
        }
        break;
    }
    precision_out
}

fn make_labels(
    sorted_breaks: &[f64],
    left_closed: bool,
    precision: usize,
    scientific: bool,
) -> Vec<String> {
    let precision = find_label_precision(sorted_breaks, precision, scientific);
    // This seems to be the most straightforward way to get rid of trailing zeros.
    let (re, replacement) = match scientific {
        true => (Regex::new(r"(\.?0+)e").unwrap(), "e"),
        false => (Regex::new(r"(\.?0+)$").unwrap(), ""),
    };
    // Rust doesn't like returning these closures as part of the tuple above.
    let format_fn = match scientific {
        true => |v, p| format!("{0:.1$e}", v, p),
        false => |v, p| format!("{0:.1$}", v, p),
    };

    let mut out = Vec::with_capacity(sorted_breaks.len() + 2);
    let mut last_break_str = format!("{}", f64::NEG_INFINITY);
    for v in sorted_breaks.iter().chain(once(&f64::INFINITY)) {
        let raw_break_str = format_fn(v, precision);
        let break_str = re.replace(&raw_break_str, replacement).to_string();
        if left_closed {
            out.push(format!("[{}, {})", last_break_str, break_str));
        } else {
            out.push(format!("({}, {}]", last_break_str, break_str))
        }
        last_break_str = break_str;
    }
    out
}

pub fn cut(
    s: &Series,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
    include_breaks: bool,
    precision: usize,
    scientific: bool,
) -> PolarsResult<Series> {
    polars_ensure!(breaks.iter().all(|x| x.is_finite()), ComputeError: "Don't include NaN, Inf, or -Inf in breaks");
    // Breaks must be sorted to cut inputs properly.
    let mut breaks = breaks;
    let sorted_breaks = breaks.as_mut_slice();
    sorted_breaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    polars_ensure!(sorted_breaks.windows(2).all(|x| x[0] != x[1]), Duplicate: "Breaks are not unique");

    let cutlabs = match labels {
        Some(ll) => {
            polars_ensure!(ll.len() == sorted_breaks.len() + 1, ShapeMismatch: "Provide nbreaks + 1 labels");
            ll
        }
        None => make_labels(&sorted_breaks, left_closed, precision, scientific),
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
    precision: usize,
    scientific: bool,
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
        return cut(
            &s,
            qbreaks,
            lfilt,
            left_closed,
            include_breaks,
            precision,
            scientific,
        );
    }
    cut(
        &s,
        qbreaks,
        labels,
        left_closed,
        include_breaks,
        precision,
        scientific,
    )
}
