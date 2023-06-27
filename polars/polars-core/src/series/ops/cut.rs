use crate::prelude::*;

impl Series {
    pub fn cut(
        &self,
        breaks: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
    ) -> PolarsResult<Series> {
        polars_ensure!(breaks.len() > 0, ShapeMismatch: "Breaks are empty");

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
            },
            None => (std::iter::once(&f64::NEG_INFINITY).chain(sorted_breaks.iter()))
                .zip(sorted_breaks.iter().chain(std::iter::once(&f64::INFINITY)))
                .map(|v| if left_closed { format!("[{}, {})", v.0, v.1) } else { format!("({}, {}]", v.0, v.1) })
                .collect::<Vec<String>>()
            //defaultlabs.iter().map(String::as_str).collect(),
        };
        
        let s_flt = self.cast(&DataType::Float64)?;
        let bin_iter = s_flt.f64()?.into_iter();

        let out_name = format!("{}_bin", self.name());
        let mut bld = CategoricalChunkedBuilder::new(&out_name, self.len());
        if left_closed {
            bld.drain_iter(bin_iter.map(|opt| {
                opt.map(|x| cutlabs[sorted_breaks.partition_point(|&v| v < x)].as_str())
            }));
        } else {
            bld.drain_iter(bin_iter.map(|opt| {
                opt.map(|x| cutlabs[sorted_breaks.partition_point(|&v| v <= x)].as_str())
            }));
        }
        Ok(bld.finish().into_series())
    }
}