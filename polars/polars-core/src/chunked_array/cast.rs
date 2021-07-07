//! Implementations of the ChunkCast Trait.
use crate::chunked_array::builder::CategoricalChunkedBuilder;
use crate::chunked_array::kernels::{cast_numeric_from_dtype, transmute_array_from_dtype};
use crate::prelude::*;
use arrow::array::{make_array, Array, ArrayDataBuilder};
use arrow::compute::cast;
use num::NumCast;

fn cast_ca<N, T>(ca: &ChunkedArray<T>) -> Result<ChunkedArray<N>>
where
    N: PolarsDataType,
    T: PolarsDataType,
{
    if N::get_dtype() == T::get_dtype() {
        return Ok(ChunkedArray::new_from_chunks(ca.name(), ca.chunks.clone()));
    };
    let chunks = ca
        .chunks
        .iter()
        .map(|arr| cast(arr, &N::get_dtype().to_arrow()))
        .collect::<arrow::error::Result<Vec<_>>>()?;

    Ok(ChunkedArray::new_from_chunks(ca.field.name(), chunks))
}

macro_rules! cast_from_dtype {
    ($self: expr, $kernel:expr, $dtype: expr) => {{
        let chunks = $self
            .downcast_iter()
            .into_iter()
            .map(|arr| $kernel(arr, $dtype))
            .collect();

        Ok(ChunkedArray::new_from_chunks($self.field.name(), chunks))
    }};
}

macro_rules! cast_with_dtype {
    ($self:expr, $data_type:expr) => {{
        use DataType::*;
        match $data_type {
            Boolean => ChunkCast::cast::<BooleanType>($self).map(|ca| ca.into_series()),
            Utf8 => ChunkCast::cast::<Utf8Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-u8")]
            UInt8 => ChunkCast::cast::<UInt8Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-u16")]
            UInt16 => ChunkCast::cast::<UInt16Type>($self).map(|ca| ca.into_series()),
            UInt32 => ChunkCast::cast::<UInt32Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-u64")]
            UInt64 => ChunkCast::cast::<UInt64Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-i8")]
            Int8 => ChunkCast::cast::<Int8Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-i16")]
            Int16 => ChunkCast::cast::<Int16Type>($self).map(|ca| ca.into_series()),
            Int32 => ChunkCast::cast::<Int32Type>($self).map(|ca| ca.into_series()),
            Int64 => ChunkCast::cast::<Int64Type>($self).map(|ca| ca.into_series()),
            Float32 => ChunkCast::cast::<Float32Type>($self).map(|ca| ca.into_series()),
            Float64 => ChunkCast::cast::<Float64Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-date32")]
            Date32 => ChunkCast::cast::<Date32Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-date64")]
            Date64 => ChunkCast::cast::<Date64Type>($self).map(|ca| ca.into_series()),
            #[cfg(feature = "dtype-time64-ns")]
            Time64(TimeUnit::Nanosecond) => {
                ChunkCast::cast::<Time64NanosecondType>($self).map(|ca| ca.into_series())
            }
            #[cfg(feature = "dtype-duration-ns")]
            Duration(TimeUnit::Nanosecond) => {
                ChunkCast::cast::<DurationNanosecondType>($self).map(|ca| ca.into_series())
            }
            #[cfg(feature = "dtype-duration-ms")]
            Duration(TimeUnit::Millisecond) => {
                ChunkCast::cast::<DurationMillisecondType>($self).map(|ca| ca.into_series())
            }
            List(_) => ChunkCast::cast::<ListType>($self).map(|ca| ca.into_series()),
            Categorical => ChunkCast::cast::<CategoricalType>($self).map(|ca| ca.into_series()),
            dt => Err(PolarsError::Other(
                format!(
                    "Casting to {:?} is not supported. \
                This error may occur because you did not activate a certain dtype feature",
                    dt
                )
                .into(),
            )),
        }
    }};
}

impl ChunkCast for CategoricalChunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        match N::get_dtype() {
            DataType::Utf8 => {
                let mapping = &**self.categorical_map.as_ref().expect("should be set");

                let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.len() * 5);

                let f = |idx: u32| mapping.get(idx);

                if self.null_count() == 0 {
                    self.into_no_null_iter()
                        .for_each(|idx| builder.append_value(f(idx)));
                } else {
                    self.into_iter().for_each(|opt_idx| {
                        builder.append_option(opt_idx.map(f));
                    });
                }

                let ca = builder.finish();
                let ca = unsafe { std::mem::transmute(ca) };
                Ok(ca)
            }
            DataType::UInt32 => {
                let mut ca: ChunkedArray<N> = unsafe { std::mem::transmute(self.clone()) };
                ca.field = Arc::new(Field::new(ca.name(), DataType::UInt32));
                Ok(ca)
            }
            DataType::Categorical => {
                let mut out = ChunkedArray::new_from_chunks(self.name(), self.chunks.clone());
                out.categorical_map = self.categorical_map.clone();
                Ok(out)
            }
            _ => cast_ca(self),
        }
    }
    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
        cast_with_dtype!(self, data_type)
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        use DataType::*;
        let ca = match (T::get_dtype(), N::get_dtype()) {
            (UInt32, Categorical) => {
                let mut ca: ChunkedArray<N> = unsafe { std::mem::transmute(self.clone()) };
                ca.field = Arc::new(Field::new(ca.name(), DataType::Categorical));
                return Ok(ca);
            }
            // the underlying datatype is i64 so we transmute array
            (Duration(_), Int64) => unsafe {
                cast_from_dtype!(self, transmute_array_from_dtype, Int64.to_arrow())
            },
            // paths not supported by arrow kernel
            // to float32
            (Duration(_), Float32) | (Date32, Float32) | (Date64, Float32)
            // to float64
            | (Duration(_), Float64) | (Date32, Float64) | (Date64, Float64)
            // underlying type: i64
            | (Duration(_), UInt64)
            => {
                cast_from_dtype!(self, cast_numeric_from_dtype, N::get_dtype().to_arrow())
            }
            // to date64
            (Float64, Date64) | (Float32, Date64) => {
                let out: Result<Int64Chunked> = cast_from_dtype!(self, cast_numeric_from_dtype, Int64.to_arrow());
                out?.cast::<N>()
            }
            // to date64
            (Float64, Date32) | (Float32, Date32) => {
                let out: Result<Int32Chunked> = cast_from_dtype!(self, cast_numeric_from_dtype, Int32.to_arrow());
                out?.cast::<N>()
            }
            _ => cast_ca(self),
        };
        ca.map(|mut ca| {
            ca.field = Arc::new(Field::new(ca.name(), N::get_dtype()));
            ca
        })
    }

    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
        cast_with_dtype!(self, data_type)
    }
}

impl ChunkCast for Utf8Chunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        match N::get_dtype() {
            DataType::Categorical => {
                let iter = self.into_iter();
                let mut builder = CategoricalChunkedBuilder::new(self.name(), self.len());
                builder.from_iter(iter);
                let ca = builder.finish();
                let ca = unsafe { std::mem::transmute(ca) };
                Ok(ca)
            }
            _ => cast_ca(self),
        }
    }
    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
        cast_with_dtype!(self, data_type)
    }
}

fn boolean_to_utf8(ca: &BooleanChunked) -> Utf8Chunked {
    ca.into_iter()
        .map(|opt_b| match opt_b {
            Some(true) => Some("true"),
            Some(false) => Some("false"),
            None => None,
        })
        .collect()
}

impl ChunkCast for BooleanChunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        if matches!(N::get_dtype(), DataType::Utf8) {
            let mut ca = boolean_to_utf8(self);
            Ok(ChunkedArray::new_from_chunks(
                self.name(),
                std::mem::take(&mut ca.chunks),
            ))
        } else {
            cast_ca(self)
        }
    }
    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
        if matches!(data_type, DataType::Utf8) {
            let mut ca = boolean_to_utf8(self);
            ca.rename(self.name());
            Ok(ca.into_series())
        } else {
            cast_with_dtype!(self, data_type)
        }
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner tyupe
impl ChunkCast for ListChunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        match N::get_dtype() {
            // Cast list inner type
            DataType::List(child_type) => {
                let chunks = self
                    .downcast_iter()
                    .map(|list| {
                        let ad = list.data().clone();
                        let child = ad.child_data()[0].clone();
                        let child = make_array(child);
                        let child = cast(&child, &child_type)?;

                        let new = ArrayDataBuilder::new(ArrowDataType::LargeList(Box::new(
                            ArrowField::new("", child.data_type().clone(), true),
                        )))
                        .len(list.len())
                        .buffers(ad.buffers().to_vec())
                        .add_child_data(child.data().clone())
                        .build();
                        Ok(make_array(new))
                    })
                    .collect::<Result<_>>()?;
                let mut ca = ListChunked::new_from_chunks(self.name(), chunks);
                Ok(ChunkedArray::new_from_chunks(
                    self.name(),
                    std::mem::take(&mut ca.chunks),
                ))
            }
            _ => Err(PolarsError::Other("Cannot cast list type".into())),
        }
    }
    fn cast_with_dtype(&self, data_type: &DataType) -> Result<Series> {
        match data_type {
            DataType::List(child_type) => {
                let chunks = self
                    .downcast_iter()
                    .map(|list| {
                        let ad = list.data().clone();
                        let child = ad.child_data()[0].clone();
                        let child = make_array(child);
                        let child = cast(&child, child_type)?;

                        let new = ArrayDataBuilder::new(ArrowDataType::LargeList(Box::new(
                            ArrowField::new("", child.data_type().clone(), true),
                        )))
                        .len(list.len())
                        .buffers(ad.buffers().to_vec())
                        .add_child_data(child.data().clone())
                        .build();
                        Ok(make_array(new))
                    })
                    .collect::<Result<_>>()?;
                let ca = ListChunked::new_from_chunks(self.name(), chunks);
                Ok(ca.into_series())
            }
            _ => Err(PolarsError::Other("Cannot cast list type".into())),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use arrow::array::PrimitiveBuilder;

    #[test]
    fn test_cast_list() -> Result<()> {
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", PrimitiveBuilder::new(10), 10);
        builder.append_slice(Some(&[1i32, 2, 3]));
        builder.append_slice(Some(&[1i32, 2, 3]));
        let ca = builder.finish();

        let new = ca.cast_with_dtype(&DataType::List(ArrowDataType::Float64))?;

        assert_eq!(new.dtype(), &DataType::List(ArrowDataType::Float64));
        Ok(())
    }

    #[test]
    fn test_cast_noop() {
        // check if we can cast categorical twice without panic
        let ca = Utf8Chunked::new_from_slice("foo", &["bar", "ham"]);
        let out = ca.cast::<CategoricalType>().unwrap();
        let out = out.cast::<CategoricalType>().unwrap();
        assert_eq!(out.dtype(), &DataType::Categorical)
    }
}
