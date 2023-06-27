use std::iter::once;

use crate::prelude::*;

impl Series {
    pub fn cut(
        &self,
        breaks: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
    ) -> PolarsResult<Series> {
        polars_ensure!(breaks.len() > 0, ShapeMismatch: "Breaks are empty");
        polars_ensure!(!breaks.iter().any(|x| x.is_nan()), ComputeError: "Breaks cannot be NaN");
        // Breaks must be sorted to cut inputs properly.
        let mut breaks = breaks.clone();
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

        let cl: Vec<&str> = cutlabs.iter().map(String::as_str).collect();
        let s_flt = self.cast(&DataType::Float64)?;
        let bin_iter = s_flt.f64()?.into_iter();

        let out_name = format!("{}_bin", self.name());
        let mut bld = CategoricalChunkedBuilder::new(&out_name, self.len());
        unsafe {
            if left_closed {
                bld.drain_iter(bin_iter.map(|opt| {
                    opt.map(|x| *cl.get_unchecked(sorted_breaks.partition_point(|&v| v < x)))
                }));
            } else {
                bld.drain_iter(bin_iter.map(|opt| {
                    opt.map(|x| *cl.get_unchecked(sorted_breaks.partition_point(|&v| v <= x)))
                }));
            }
        }
        Ok(bld.finish().into_series())
    }

    pub fn qcut(
        &self,
        probs: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
    ) -> PolarsResult<Series> {
        let s = self.sort(false);
        let s = s.f64()?;
        let f = |&p| {
            s.quantile(p, QuantileInterpolOptions::Linear)
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
            return self.cut(qbreaks, lfilt, left_closed);
        }
        self.cut(qbreaks, labels, left_closed)
    }
}
