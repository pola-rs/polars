use std::borrow::Cow;
use std::fmt;

use polars_arrow::legacy::utils::CustomIterTools;
use polars_compute::approx_quantile::{ApproxQuantileMethod, FinalizedSketch, Sketch};
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::_split_offsets;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::pl_serialize;
use polars_utils::total_ord::TotalOrd;
use rayon::prelude::*;

/// Encode `sketches` (finalized, or ingesting states) as one opaque blob per row.
pub fn sketches_to_series<S: serde::Serialize>(sketches: &[S]) -> PolarsResult<Series> {
    let mut builder = BinaryChunkedBuilder::new(PlSmallStr::EMPTY, sketches.len());
    let mut blob = Vec::new();
    for sketch in sketches {
        blob.clear();
        pl_serialize::serialize_into_writer::<_, _, false>(&mut blob, sketch)?;
        builder.append_value(&blob);
    }
    Ok(builder.finish().into_series())
}

fn build_sketch<T>(
    items: &[T],
    method: &ApproxQuantileMethod,
    error: f64,
    failure_probability: f64,
) -> FinalizedSketch<T>
where
    T: fmt::Debug + Clone + TotalOrd + Send + Sync,
{
    const THREAD_BOUNDARY: usize = if cfg!(debug_assertions) { 1 } else { 100_000 };

    let build = |items: &[T]| {
        let mut sketch = Sketch::new(method, error, failure_probability);
        for item in items {
            sketch.update(item);
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
    method: &ApproxQuantileMethod,
    error: f64,
    failure_probability: f64,
) -> PolarsResult<Series> {
    let items = sketch_repr(s)?;
    let items: &Series = items.as_ref();
    let out = match items.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = items.as_ref().as_ref();
                let ca = ca.drop_nulls();
                let ca = ca.rechunk();
                let sketch = build_sketch(ca.cont_slice()?, method, error, failure_probability);
                sketches_to_series(&[sketch])
            })
        },
        DataType::Boolean => {
            let items: Vec<bool> = items.bool()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, method, error, failure_probability)])
        },
        DataType::String => {
            let items: Vec<&str> = items.str()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, method, error, failure_probability)])
        },
        DataType::Binary => {
            let items: Vec<&[u8]> = items.binary()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, method, error, failure_probability)])
        },
        DataType::BinaryOffset => {
            let items: Vec<&[u8]> = items.binary_offset()?.iter().flatten().collect();
            sketches_to_series(&[build_sketch(&items, method, error, failure_probability)])
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{}`", s.dtype())
        },
    }?;
    Ok(out.with_name(s.name().clone()))
}

/// Convert `s` to sketch items that compare as `s` sorts: the physical repr of `s`, or its
/// ordered row encoding if the physical repr compares differently.
pub fn sketch_repr(s: &Series) -> PolarsResult<Cow<'_, Series>> {
    if !sketch_items_are_row_encoded(s.dtype()) {
        return Ok(s.to_physical_repr());
    }
    // TODO: See if we can row-encode lazily.
    let rows = _get_rows_encoded_ca(
        s.name().clone(),
        &[s.clone().into_column()],
        &[false],
        &[false],
        true,
    )?;
    Ok(Cow::Owned(rows.into_series()))
}

/// Return the dtype of [`sketch_repr`] for values of `dtype`.
pub fn sketch_dtype(dtype: &DataType) -> DataType {
    if sketch_items_are_row_encoded(dtype) {
        DataType::BinaryOffset
    } else {
        dtype.to_physical()
    }
}

fn sketch_items_are_row_encoded(dtype: &DataType) -> bool {
    match dtype {
        #[cfg(feature = "dtype-extension")]
        DataType::Extension(_, storage) => sketch_items_are_row_encoded(storage),
        // Categoricals sort lexically, not by their physical id.
        dt => dt.is_categorical() || dt.is_nested() || dt.is_null(),
    }
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
    let quantiles = quantiles.broadcast_to(sketch.len())?.list()?.to_owned();
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
    let estimates = match sketch_dtype(values_dtype) {
        physical if physical.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(physical, |$T| {
                let estimates = approx_quantile_estimate_inner::<<$T as PolarsNumericType>::Native>(sketch, &quantiles, quantiles_inner)?;
                ChunkedArray::<$T>::from_iter_options(PlSmallStr::EMPTY, estimates.into_iter())
                    .into_series()
            })
        },
        DataType::Boolean => {
            let estimates =
                approx_quantile_estimate_inner::<bool>(sketch, &quantiles, quantiles_inner)?;
            BooleanChunked::from_iter_options(PlSmallStr::EMPTY, estimates.into_iter())
                .into_series()
        },
        DataType::String => {
            let estimates =
                approx_quantile_estimate_inner::<String>(sketch, &quantiles, quantiles_inner)?;
            StringChunked::from_iter_options(
                PlSmallStr::EMPTY,
                estimates.iter().map(|v| v.as_deref()),
            )
            .into_series()
        },
        DataType::Binary => {
            let estimates =
                approx_quantile_estimate_inner::<Vec<u8>>(sketch, &quantiles, quantiles_inner)?;
            BinaryChunked::from_iter_options(
                PlSmallStr::EMPTY,
                estimates.iter().map(|v| v.as_deref()),
            )
            .into_series()
        },
        DataType::BinaryOffset => {
            let estimates =
                approx_quantile_estimate_inner::<Vec<u8>>(sketch, &quantiles, quantiles_inner)?;
            let estimates: BinaryOffsetChunked =
                estimates.iter().map(|v| v.as_deref()).collect_trusted();
            estimates.into_series()
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{values_dtype}`")
        },
    };
    let estimates = if sketch_items_are_row_encoded(values_dtype) {
        decode_rows(estimates.binary_offset()?, values_dtype)?
    } else {
        // SAFETY: the estimates are items the input itself held.
        unsafe { estimates.from_physical_unchecked(values_dtype)? }
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
    values: &Float64Chunked,
) -> PolarsResult<Vec<Option<T>>> {
    let mut out = Vec::with_capacity(values.len());
    let mut values = values.no_null_iter();
    let lengths = quantiles
        .downcast_iter()
        .flat_map(|arr| arr.offsets().lengths());
    for (blob, len) in Iterator::zip(sketch.iter(), lengths) {
        let sketch: Option<FinalizedSketch<T>> = blob
            .map(pl_serialize::deserialize_from_reader::<_, _, false>)
            .transpose()?;
        for quantile in values.by_ref().take(len) {
            let estimate = match &sketch {
                Some(sketch) => sketch.estimate_quantile(quantile)?.cloned(),
                None => None,
            };
            out.push(estimate);
        }
    }
    Ok(out)
}

/// Decode the ordered row encodings `rows` of values of `dtype`.
fn decode_rows(rows: &BinaryOffsetChunked, dtype: &DataType) -> PolarsResult<Series> {
    // A null has no row to decode, so decode the valid rows and gather them back in place.
    let field = Field::new(rows.name().clone(), dtype.clone());
    let decoded = rows
        .drop_nulls()
        .row_decode_ordered(&[field], &[false], &[false])?;
    let decoded = decoded.fields_as_series().pop().unwrap();
    let mut next: IdxSize = 0;
    let idx = IdxCa::from_iter_options(
        PlSmallStr::EMPTY,
        rows.iter().map(|row| {
            row.map(|_| {
                next += 1;
                next - 1
            })
        }),
    );
    decoded.take(&idx)
}
