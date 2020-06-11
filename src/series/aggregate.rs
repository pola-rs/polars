use crate::series::{
    chunked_array::{aggregate::Agg, iterator::ChunkIterator},
    series::Series,
};
use num::{Num, NumCast, ToPrimitive, Zero};

macro_rules! apply_agg_fn {
    ($self:ident, $agg:ident) => {
        match $self {
            Series::Bool(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast bool to T")),
            Series::Int32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i32 to T")),
            Series::UInt32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast u32 to T")),
            Series::Int64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast i64 to T")),
            Series::Float32(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast f32 to T")),
            Series::Float64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast f64 to T")),
            Series::Utf8(_a) => unimplemented!(),
        }
    };
}

impl<T> Agg<T> for Series
where
    T: Num + NumCast + Zero + ToPrimitive,
{
    fn sum(&self) -> Option<T> {
        apply_agg_fn!(self, sum)
    }

    fn min(&self) -> Option<T> {
        apply_agg_fn!(self, min)
    }

    fn max(&self) -> Option<T> {
        apply_agg_fn!(self, max)
    }
}
