use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::builder::CategoricalChunkedBuilder;
use polars_core::chunked_array::ops::binning::quantiles_without_nan;
use polars_core::prelude::*;
use polars_core::utils::cut::compute_cut_labels;

fn map_cats<C: PolarsCategoricalType>(
    s: &Series,
    cat_dtype: DataType,
    labels: &[PlSmallStr],
    sorted_breaks: &[f64],
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let out_name = PlSmallStr::from_static("category");

    let s2 = s.cast(&DataType::Float64)?;
    let s_iter = s2.f64()?.iter();

    let op: fn(&f64, &f64) -> bool = if left_closed {
        PartialOrd::ge
    } else {
        PartialOrd::gt
    };

    let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = &cat_dtype else {
        unreachable!()
    };
    let mapping = mapping.clone();
    let is_enum = cat_dtype.is_enum();

    // Labels are resolved to category ids on first use.
    let mut label_cats: Vec<Option<CatSize>> = vec![None; labels.len()];
    let mut bld = CategoricalChunkedBuilder::<C>::new(out_name.clone(), cat_dtype);
    bld.reserve(s.len());
    let mut brk_vals = include_breaks.then(|| {
        PrimitiveChunkedBuilder::<Float64Type>::new(PlSmallStr::from_static("breakpoint"), s.len())
    });
    let right_ends = if include_breaks {
        [sorted_breaks, &[f64::INFINITY]].concat()
    } else {
        Vec::new()
    };

    for opt in s_iter {
        let idx = opt
            .filter(|x| !x.is_nan())
            .map(|x| sorted_breaks.partition_point(|v| op(&x, v)));
        match idx {
            None => {
                bld.append_null();
                if let Some(brk_vals) = &mut brk_vals {
                    brk_vals.append_null();
                }
            },
            Some(idx) => unsafe {
                let slot = label_cats.get_unchecked_mut(idx);
                let cat = match *slot {
                    Some(cat) => cat,
                    None => {
                        let label = labels.get_unchecked(idx).as_str();
                        let cat = if is_enum {
                            mapping.get_cat(label).ok_or_else(|| {
                                polars_err!(ComputeError: "attempted to insert '{label}' into Enum which does not contain this string")
                            })?
                        } else {
                            mapping.insert_cat(label)?
                        };
                        *slot = Some(cat);
                        cat
                    },
                };
                bld.append_cat(cat, &mapping)?;
                if let Some(brk_vals) = &mut brk_vals {
                    brk_vals.append_value(*right_ends.get_unchecked(idx));
                }
            },
        }
    }

    let cats = bld.finish().into_series();
    match brk_vals {
        // The breakpoint column is the right endpoint of each interval. A struct is returned
        // so it can be turned into a dataframe later.
        Some(brk_vals) => {
            let outvals = [brk_vals.finish().into_series(), cats];
            Ok(
                StructChunked::from_series(out_name, outvals[0].len(), outvals.iter())?
                    .into_series(),
            )
        },
        None => Ok(cats),
    }
}

fn map_enum_cats(
    s: &Series,
    labels: &[PlSmallStr],
    sorted_breaks: &[f64],
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let fcats = FrozenCategories::new(labels.iter().map(|s| s.as_str()))?;
    let enum_dtype = DataType::from_frozen_categories(fcats.clone());

    with_match_categorical_physical_type!(fcats.physical(), |$C| {
        map_cats::<$C>(s, enum_dtype, labels, sorted_breaks, left_closed, include_breaks)
    })
}

fn map_categorical_cats(
    s: &Series,
    labels: &[PlSmallStr],
    sorted_breaks: &[f64],
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let cat_dtype = DataType::from_categories(Categories::global());
    map_cats::<Categorical32Type>(
        s,
        cat_dtype,
        labels,
        sorted_breaks,
        left_closed,
        include_breaks,
    )
}

pub fn cut(
    s: &Series,
    mut breaks: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    // Breaks must be sorted to cut inputs properly.
    polars_ensure!(!breaks.iter().any(|x| x.is_nan()), ComputeError: "breaks cannot be NaN");
    breaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    polars_ensure!(breaks.windows(2).all(|x| x[0] != x[1]), Duplicate: "breaks are not unique");
    if !breaks.is_empty() {
        polars_ensure!(breaks[0] > f64::NEG_INFINITY, ComputeError: "don't include -inf in breaks");
        polars_ensure!(breaks[breaks.len() - 1] < f64::INFINITY, ComputeError: "don't include inf in breaks");
    }

    let cut_labels = if let Some(l) = labels {
        polars_ensure!(
            l.len() == breaks.len() + 1,
            ShapeMismatch: "expected {} labels (len(breaks) + 1), got {}",
            breaks.len() + 1, l.len()
        );
        l
    } else {
        compute_cut_labels(&breaks, left_closed)?
    };
    map_enum_cats(s, &cut_labels, &breaks, left_closed, include_breaks)
}

pub fn qcut(
    s: &Series,
    probs: Vec<f64>,
    labels: Option<Vec<PlSmallStr>>,
    left_closed: bool,
    allow_duplicates: bool,
    include_breaks: bool,
) -> PolarsResult<Series> {
    let s = s.cast(&DataType::Float64)?;
    let Some(mut qbreaks) = quantiles_without_nan(s.f64()?, &probs, QuantileMethod::Linear)? else {
        // No usable values (all null/NaN/empty): no breakpoints. With
        // `include_breaks` the output must still be a Struct to match the schema.
        let cat_dtype = DataType::from_categories(Categories::global());
        if include_breaks {
            let brk = Series::full_null(
                PlSmallStr::from_static("breakpoint"),
                s.len(),
                &DataType::Float64,
            );
            let cat = Series::full_null(PlSmallStr::from_static("category"), s.len(), &cat_dtype);
            return Ok(StructChunked::from_series(
                s.name().clone(),
                s.len(),
                [&brk, &cat].into_iter(),
            )?
            .into_series());
        }
        return Ok(Series::full_null(s.name().clone(), s.len(), &cat_dtype));
    };

    // Interpolating across an infinity gives a NaN breakpoint that would panic
    // the sort below; reject it, as `cut` already does.
    polars_ensure!(
        !qbreaks.iter().any(|x| x.is_nan()),
        ComputeError: "quantile breakpoint is NaN (the input may contain infinite values)"
    );

    qbreaks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    if !allow_duplicates {
        polars_ensure!(qbreaks.windows(2).all(|x| x[0] != x[1]), Duplicate: "quantiles are not unique while allow_duplicates=False");
    }

    let cut_labels = if let Some(l) = labels {
        polars_ensure!(
            l.len() == qbreaks.len() + 1,
            ShapeMismatch: "expected {} labels (len(breaks) + 1), got {}",
            qbreaks.len() + 1, l.len()
        );
        l
    } else {
        compute_cut_labels(&qbreaks, left_closed)?
    };

    map_categorical_cats(&s, &cut_labels, &qbreaks, left_closed, include_breaks)
}

mod test {
    // This need metadata in fields
    #[ignore]
    #[test]
    fn test_map_cats_fast_unique() {
        // This test is here to check the fast unique flag is set when it can be
        // as it is not visible to Python.
        use polars_core::prelude::*;

        use super::{map_categorical_cats, map_enum_cats};

        let s = Series::new("x".into(), &[1, 2, 3, 4, 5]);

        let labels = &["a", "b", "c"].map(PlSmallStr::from_static);
        let breaks = &[2.0, 4.0];
        let left_closed = false;

        let include_breaks = false;
        let out = map_enum_cats(&s, labels, breaks, left_closed, include_breaks).unwrap();
        out.cat8().unwrap();

        let include_breaks = true;
        let out = map_enum_cats(&s, labels, breaks, left_closed, include_breaks).unwrap();
        let out = out.struct_().unwrap().fields_as_series()[1].clone();
        out.cat8().unwrap();

        let include_breaks = false;
        let out = map_categorical_cats(&s, labels, breaks, left_closed, include_breaks).unwrap();
        out.cat32().unwrap();

        let include_breaks = true;
        let out = map_categorical_cats(&s, labels, breaks, left_closed, include_breaks).unwrap();
        let out = out.struct_().unwrap().fields_as_series()[1].clone();
        out.cat32().unwrap();
    }
}
