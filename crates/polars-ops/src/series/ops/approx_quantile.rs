use std::fmt;

use polars_compute::approx_quantile::{ApproxQuantileMethod, FinalizedSketch, Sketch};
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::_split_offsets;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::pl_serialize;
use polars_utils::total_ord::TotalOrd;
use rayon::prelude::*;

/// Encode finalized `sketches` as one opaque blob per row.
pub fn sketches_to_series<T: fmt::Debug + Clone + TotalOrd + serde::Serialize>(
    sketches: &[FinalizedSketch<T>],
) -> PolarsResult<Series> {
    let mut builder = BinaryChunkedBuilder::new(PlSmallStr::EMPTY, sketches.len());
    let mut blob = Vec::new();
    for sketch in sketches {
        blob.clear();
        pl_serialize::serialize_into_writer::<_, _, false>(&mut blob, sketch)?;
        builder.append_value(&blob);
    }
    Ok(builder.finish().into_series())
}

fn build_sketch<T>(items: &[T], error: f64, method: &ApproxQuantileMethod) -> FinalizedSketch<T>
where
    T: fmt::Debug + Clone + TotalOrd + Send + Sync,
{
    const THREAD_BOUNDARY: usize = if cfg!(debug_assertions) { 1 } else { 100_000 };

    let build = |items: &[T]| {
        let mut sketch = Sketch::new(method, error);
        for item in items {
            sketch.update_owned(item.clone());
        }
        sketch
    };

    let sketch = if items.len() < THREAD_BOUNDARY
        || RAYON.current_num_threads() == 1
        || RAYON.current_thread_has_pending_tasks().unwrap_or(false)
    {
        build(items)
    } else {
        let splits = _split_offsets(items.len(), RAYON.current_num_threads());
        RAYON
            .install(|| {
                splits
                    .into_par_iter()
                    .map(|(offset, len)| build(&items[offset..offset + len]))
                    .reduce_with(|mut acc, sketch| {
                        acc.merge(&sketch);
                        acc
                    })
            })
            .unwrap()
    };
    sketch.finalize()
}

/// Summarize `s` into a single-row sketch column.
pub fn approx_quantile_sketch(
    s: &Series,
    error: f64,
    method: &ApproxQuantileMethod,
) -> PolarsResult<Series> {
    let out = match s.dtype() {
        dt if dt.is_primitive_numeric() || dt.is_temporal() || dt.is_decimal() => {
            let physical = s.to_physical_repr();
            let physical: &Series = physical.as_ref();
            with_match_physical_numeric_polars_type!(physical.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = physical.as_ref().as_ref();
                let ca = ca.drop_nulls();
                let ca = ca.rechunk();
                let sketch = build_sketch(ca.cont_slice()?, error, method);
                sketches_to_series(&[sketch])
            })
        },
        DataType::Boolean => {
            let items: Vec<bool> = s.bool()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, error, method)])
        },
        DataType::String => {
            let items: Vec<&str> = s.str()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, error, method)])
        },
        dt => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{dt}`")
        },
    }?;
    Ok(out.with_name(s.name().clone()))
}

/// Estimate `quantiles` from every sketch of `sketch`.
pub fn approx_quantile_estimate(
    sketch: &Series,
    quantiles: &Series,
    values_dtype: &DataType,
) -> PolarsResult<Series> {
    let sketch = sketch.binary().expect("incorrect dtype");
    let quantiles_is_list = quantiles.dtype().is_list();
    if sketch.is_empty() {
        let dtype = if quantiles_is_list {
            DataType::List(Box::new(values_dtype.clone()))
        } else {
            values_dtype.clone()
        };
        return Ok(Series::new_empty(sketch.name().clone(), &dtype));
    }

    polars_ensure!(
        !quantiles.is_empty(),
        ComputeError:
            "the 'quantile' expression input should produce a single quantile or a list of quantiles, \
            got an empty input"
    );
    polars_ensure!(
        quantiles.len() == 1 || quantiles.len() == sketch.len(),
        ComputeError:
            "polars does not support varying approximate quantiles, \
            make sure the 'quantile' expression input produces a single quantile or a list of quantiles"
    );

    let quantiles = if quantiles_is_list {
        quantiles.strict_cast(&DataType::List(Box::new(DataType::Float64)))?
    } else {
        quantiles
            .strict_cast(&DataType::Float64)?
            .to_unit_list()
            .into_series()
    };
    let quantiles = quantiles
        .broadcast_to(sketch.len())?
        .list()?
        .rechunk()
        .into_owned();
    polars_ensure!(
        !quantiles.has_nulls(),
        ComputeError: "`quantile` should not be null",
    );

    let quantiles_inner = quantiles.get_inner();
    let quantiles_inner = quantiles_inner.f64()?;
    polars_ensure!(
        !quantiles_inner.has_nulls(),
        ComputeError: "`quantile` should not contain null values",
    );
    let values = quantiles_inner.cont_slice().expect("rechunk");

    let estimates = match values_dtype {
        _ if values_dtype.is_primitive_numeric()
            || values_dtype.is_temporal()
            || values_dtype.is_decimal() =>
        {
            let physical = values_dtype.to_physical();
            let estimates = with_match_physical_numeric_polars_type!(physical, |$T| {
                let estimates = approx_quantile_estimate_inner::<<$T as PolarsNumericType>::Native>(sketch, &quantiles, values)?;
                ChunkedArray::<$T>::from_iter_options(PlSmallStr::EMPTY, estimates.into_iter())
                    .into_series()
            });
            // SAFETY: the estimates are items the input itself held.
            unsafe { estimates.from_physical_unchecked(values_dtype)? }
        },
        DataType::Boolean => {
            let estimates = approx_quantile_estimate_inner::<bool>(sketch, &quantiles, values)?;
            BooleanChunked::from_iter_options(PlSmallStr::EMPTY, estimates.into_iter())
                .into_series()
        },
        DataType::String => {
            let estimates = approx_quantile_estimate_inner::<String>(sketch, &quantiles, values)?;
            StringChunked::from_iter_options(
                PlSmallStr::EMPTY,
                estimates.iter().map(|v| v.as_deref()),
            )
            .into_series()
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{values_dtype}`")
        },
    };

    let estimates = quantiles.with_inner_values(&estimates);
    let out = match quantiles_is_list {
        true => estimates.into_series(),
        false => estimates.get_inner(),
    };
    Ok(out.with_name(sketch.name().clone()))
}

/// Estimate every quantile of every sketch, as one estimate per output slot.
fn approx_quantile_estimate_inner<
    T: fmt::Debug + Clone + TotalOrd + serde::de::DeserializeOwned,
>(
    sketch: &BinaryChunked,
    quantiles: &ListChunked,
    values: &[f64],
) -> PolarsResult<Vec<Option<T>>> {
    let mut out = Vec::with_capacity(values.len());
    let offsets = quantiles.downcast_as_array().offsets();
    for ((offset, len), blob) in Iterator::zip(offsets.offset_and_length_iter(), sketch.iter()) {
        let sketch: Option<FinalizedSketch<T>> = blob
            .map(pl_serialize::deserialize_from_reader::<_, _, false>)
            .transpose()?;
        for quantile in &values[offset..offset + len] {
            let estimate = match &sketch {
                Some(sketch) => sketch.estimate_quantile(*quantile)?.cloned(),
                None => None,
            };
            out.push(estimate);
        }
    }
    Ok(out)
}
