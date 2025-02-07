use super::*;
use crate::series::{index_of, index_of_null};

pub fn list_index_of_in(ca: &ListChunked, needles: &Series) -> PolarsResult<Series> {
    // Handle scalar case separately, since we can do some optimizations given
    // the extra knowledge we have.
    if needles.len() == 1 {
        let needle = needles.get(0).unwrap();
        return list_index_of_in_for_scalar(ca, needle);
    }

    polars_ensure!(
        ca.len() == needles.len(),
        ComputeError: "shapes don't match: expected {} elements in 'index_of_in' comparison, got {}",
        ca.len(),
        needles.len()
    );
    let mut builder = PrimitiveChunkedBuilder::<IdxType>::new(ca.name().clone(), ca.len());
    let needles = needles.rechunk();
    ca.amortized_iter()
        // TODO iter() assumes a single chunk. could continue to use this
        // and just rechunk(), or have needles also be a ChunkedArray, in
        // which case we'd need to have to use one of the
        // dispatch-on-dtype-and-cast-to-relevant-chunkedarray-type macros
        // to duplicate the implementation code per dtype.
        .zip(needles.iter())
        .for_each(|(opt_series, needle)| match (opt_series, needle) {
            (None, _) => builder.append_null(),
            (Some(subseries), needle) => {
                let needle = Scalar::new(needles.dtype().clone(), needle.into_static());
                builder.append_option(
                    index_of(subseries.as_ref(), needle)
                        .unwrap()
                        .map(|v| v.try_into().unwrap()),
                );
            },
        });

    Ok(builder.finish().into())
}

macro_rules! process_series_for_numeric_value {
    ($extractor:ident, $needle:ident) => {{
        use arrow::array::PrimitiveArray;

        use crate::series::index_of_value;

        let needle = $needle.extract::<$extractor>().unwrap();
        Box::new(move |subseries| {
            index_of_value::<_, PrimitiveArray<$extractor>>(subseries.$extractor().unwrap(), needle)
        })
    }};
}

fn list_index_of_in_for_scalar(ca: &ListChunked, needle: AnyValue<'_>) -> PolarsResult<Series> {
    let mut builder = PrimitiveChunkedBuilder::<IdxType>::new(ca.name().clone(), ca.len());
    let needle = needle.into_static();
    let inner_dtype = ca.dtype().inner_dtype().unwrap();
    let needle_dtype = needle.dtype();

    let process_series: Box<dyn Fn(&Series) -> Option<usize>> = match needle_dtype {
        DataType::Null => Box::new(|subseries| index_of_null(subseries)),
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => process_series_for_numeric_value!(u8, needle),
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => process_series_for_numeric_value!(u16, needle),
        DataType::UInt32 => process_series_for_numeric_value!(u32, needle),
        DataType::UInt64 => process_series_for_numeric_value!(u64, needle),
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => process_series_for_numeric_value!(i8, needle),
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => process_series_for_numeric_value!(i16, needle),
        DataType::Int32 => process_series_for_numeric_value!(i32, needle),
        DataType::Int64 => process_series_for_numeric_value!(i64, needle),
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => process_series_for_numeric_value!(i128, needle),
        DataType::Float32 => process_series_for_numeric_value!(f32, needle),
        DataType::Float64 => process_series_for_numeric_value!(f64, needle),
        // Just use the general purpose index_of() function:
        _ => Box::new(|subseries| {
            let needle = Scalar::new(inner_dtype.clone(), needle.clone());
            index_of(subseries, needle).unwrap()
        }),
    };

    ca.amortized_iter().for_each(|opt_series| {
        if let Some(subseries) = opt_series {
            builder
                .append_option(process_series(subseries.as_ref()).map(|v| v.try_into().unwrap()));
        } else {
            builder.append_null();
        }
    });
    Ok(builder.finish().into())
}
