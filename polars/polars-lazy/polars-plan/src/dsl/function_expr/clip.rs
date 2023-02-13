use polars_core::utils::CustomIterTools;
use DataType::*;

use super::*;

pub(super) fn clip(args: &[Series]) -> PolarsResult<Series> {
    let ca = &args[0];
    let ca_type = ca.dtype();

    let min = &args[1].cast(ca_type)?;
    let max = &args[2].cast(ca_type)?;

    let mut out = match ca_type {
        Date | Time | Datetime(_, _) | DataType::Duration(_) => {
            let ca_physical = ca.to_physical_repr();
            let ca = ca_physical.i64().unwrap();
            let min_physical = min.to_physical_repr();
            let min = min_physical.i64().unwrap();
            let max_physical = max.to_physical_repr();
            let max = max_physical.i64().unwrap();
            clip_ca(ca, min, max).into_series().cast(ca_type).unwrap()
        }
        Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 => {
            let ca = ca.i64().unwrap();
            let min = min.i64().unwrap();
            let max = max.i64().unwrap();
            clip_ca(ca, min, max).into_series()
        }
        UInt64 => {
            let ca = ca.u64().unwrap();
            let min = min.u64().unwrap();
            let max = max.u64().unwrap();
            clip_ca(ca, min, max).into_series()
        }
        Float32 => {
            let ca = ca.f32().unwrap();
            let min = min.f32().unwrap();
            let max = max.f32().unwrap();
            clip_ca(ca, min, max).into_series()
        }
        Float64 => {
            let ca = ca.f64().unwrap();
            let min = min.f64().unwrap();
            let max = max.f64().unwrap();
            clip_ca(ca, min, max).into_series()
        }
        dt => panic!("clip not supported for dtype: {dt:?}"),
    };

    out.rename(ca.name());
    Ok(out)
}

pub(super) fn clip_min(args: &[Series]) -> PolarsResult<Series> {
    let ca = &args[0];
    let ca_type = ca.dtype();

    let min = &args[1].cast(ca_type)?;

    let mut out = match ca_type {
        Date | Time | Datetime(_, _) | DataType::Duration(_) => {
            let ca_physical = ca.to_physical_repr();
            let ca = ca_physical.i64().unwrap();
            let min_physical = min.to_physical_repr();
            let min = min_physical.i64().unwrap();
            clip_min_ca(ca, min).into_series().cast(ca_type).unwrap()
        }
        Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 => {
            let ca = ca.i64().unwrap();
            let min = min.i64().unwrap();
            clip_min_ca(ca, min).into_series()
        }
        UInt64 => {
            let ca = ca.u64().unwrap();
            let min = min.u64().unwrap();
            clip_min_ca(ca, min).into_series()
        }
        Float32 => {
            let ca = ca.f32().unwrap();
            let min = min.f32().unwrap();
            clip_min_ca(ca, min).into_series()
        }
        Float64 => {
            let ca = ca.f64().unwrap();
            let min = min.f64().unwrap();
            clip_min_ca(ca, min).into_series()
        }
        dt => panic!("clip_min not supported for dtype: {dt:?}"),
    };

    out.rename(ca.name());
    Ok(out)
}

pub(super) fn clip_max(args: &[Series]) -> PolarsResult<Series> {
    let ca = &args[0];
    let ca_type = ca.dtype();

    let max = &args[1].cast(ca_type)?;

    let mut out = match ca_type {
        Date | Time | Datetime(_, _) | DataType::Duration(_) => {
            let ca_physical = ca.to_physical_repr();
            let ca = ca_physical.i64().unwrap();
            let max_physical = max.to_physical_repr();
            let max = max_physical.i64().unwrap();
            clip_max_ca(ca, max).into_series().cast(ca_type).unwrap()
        }
        Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 => {
            let ca = ca.i64().unwrap();
            let max = max.i64().unwrap();
            clip_max_ca(ca, max).into_series()
        }
        UInt64 => {
            let ca = ca.u64().unwrap();
            let max = max.u64().unwrap();
            clip_max_ca(ca, max).into_series()
        }
        Float32 => {
            let ca = ca.f32().unwrap();
            let max = max.f32().unwrap();
            clip_max_ca(ca, max).into_series()
        }
        Float64 => {
            let ca = ca.f64().unwrap();
            let max = max.f64().unwrap();
            clip_max_ca(ca, max).into_series()
        }
        dt => panic!("clip_max not supported for dtype: {dt:?}"),
    };

    out.rename(ca.name());
    Ok(out)
}

fn clip_min_ca<T>(ca: &ChunkedArray<T>, min: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let f = |(src, min)| match (src, min) {
        (Some(_), Some(_)) => {
            if src < min {
                min
            } else {
                src
            }
        }
        (Some(_), _) => src,
        _ => None,
    };

    let out: ChunkedArray<T> = match min.len() {
        1 => ca.into_iter().map(|x| f((x, min.get(0)))).collect_trusted(),
        _ => ca.into_iter().zip(min.into_iter()).map(f).collect_trusted(),
    };

    out
}

fn clip_max_ca<T>(ca: &ChunkedArray<T>, max: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let f = |(src, max)| match (src, max) {
        (Some(_), Some(_)) => {
            if src > max {
                max
            } else {
                src
            }
        }
        (Some(_), _) => src,
        _ => None,
    };

    let out: ChunkedArray<T> = match max.len() {
        1 => ca.into_iter().map(|x| f((x, max.get(0)))).collect_trusted(),
        _ => ca.into_iter().zip(max.into_iter()).map(f).collect_trusted(),
    };

    out
}

fn clip_ca<T>(ca: &ChunkedArray<T>, min: &ChunkedArray<T>, max: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let f = |(src, (min, max))| match (src, (min, max)) {
        (Some(_), (Some(_), Some(_))) => {
            if min > max {
                None
            } else if src < min {
                min
            } else if src > max {
                max
            } else {
                src
            }
        }
        (Some(_), (Some(_), _)) => {
            if src < min {
                min
            } else {
                src
            }
        }
        (Some(_), (_, Some(_))) => {
            if src > max {
                max
            } else {
                src
            }
        }
        (Some(_), (_, _)) => src,
        _ => None,
    };

    let out: ChunkedArray<T> = match (min.len(), max.len()) {
        (1, 1) => ca
            .into_iter()
            .map(|x| f((x, (min.get(0), max.get(0)))))
            .collect_trusted(),
        (1, _) => ca
            .into_iter()
            .zip(max.into_iter())
            .map(|(x, max)| f((x, (min.get(0), max))))
            .collect_trusted(),
        (_, 1) => ca
            .into_iter()
            .zip(min.into_iter())
            .map(|(x, min)| f((x, (min, max.get(0)))))
            .collect_trusted(),
        _ => ca
            .into_iter()
            .zip(min.into_iter().zip(max.into_iter()))
            .map(f)
            .collect_trusted(),
    };
    out
}
