mod _trait;
mod implementations;
mod ops;

use std::sync::Arc;

use polars_core::prelude::*;
use polars_core::utils::Wrap;

pub use self::_trait::*;

type SeriesOpsRef = Arc<dyn SeriesOps>;

pub trait IntoSeriesOps {
    fn to_ops(&self) -> SeriesOpsRef;
}
pub use ops::*;

impl IntoSeriesOps for Series {
    fn to_ops(&self) -> SeriesOpsRef {
        match self.dtype() {
            DataType::Int8 => self.i8().unwrap().to_ops(),
            DataType::Int16 => self.i16().unwrap().to_ops(),
            DataType::Int32 => self.i32().unwrap().to_ops(),
            DataType::Int64 => self.i64().unwrap().to_ops(),
            DataType::UInt8 => self.u8().unwrap().to_ops(),
            DataType::UInt16 => self.u16().unwrap().to_ops(),
            DataType::UInt32 => self.u32().unwrap().to_ops(),
            DataType::UInt64 => self.u64().unwrap().to_ops(),
            DataType::Float32 => self.f32().unwrap().to_ops(),
            DataType::Float64 => self.f64().unwrap().to_ops(),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => self.categorical().unwrap().to_ops(),
            DataType::Boolean => self.bool().unwrap().to_ops(),
            DataType::Utf8 => self.utf8().unwrap().to_ops(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().unwrap().to_ops(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().unwrap().to_ops(),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => self.duration().unwrap().to_ops(),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().unwrap().to_ops(),
            DataType::List(_) => self.list().unwrap().to_ops(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.struct_().unwrap().to_ops(),
            _ => unimplemented!(),
        }
    }
}

impl<T: PolarsIntegerType> IntoSeriesOps for &ChunkedArray<T>
where
    T::Native: NumericNative,
{
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapInt((*self).clone()))
    }
}

#[repr(transparent)]
pub(crate) struct WrapFloat<T>(pub T);

#[repr(transparent)]
pub(crate) struct WrapInt<T>(pub T);

impl IntoSeriesOps for Float32Chunked {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapFloat(self.clone()))
    }
}

impl IntoSeriesOps for Float64Chunked {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapFloat(self.clone()))
    }
}

macro_rules! into_ops_impl_wrapped {
    ($tp:ty) => {
        impl IntoSeriesOps for $tp {
            fn to_ops(&self) -> SeriesOpsRef {
                Arc::new(Wrap(self.clone()))
            }
        }
    };
}

into_ops_impl_wrapped!(Utf8Chunked);
into_ops_impl_wrapped!(BooleanChunked);
#[cfg(feature = "dtype-date")]
into_ops_impl_wrapped!(DateChunked);
#[cfg(feature = "dtype-time")]
into_ops_impl_wrapped!(TimeChunked);
#[cfg(feature = "dtype-duration")]
into_ops_impl_wrapped!(DurationChunked);
#[cfg(feature = "dtype-datetime")]
into_ops_impl_wrapped!(DatetimeChunked);
#[cfg(feature = "dtype-struct")]
into_ops_impl_wrapped!(StructChunked);
into_ops_impl_wrapped!(ListChunked);

#[cfg(feature = "dtype-categorical")]
into_ops_impl_wrapped!(CategoricalChunked);

#[cfg(feature = "object")]
impl<T: PolarsObject> IntoSeriesOps for ObjectChunked<T> {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(Wrap(self.clone()))
    }
}
