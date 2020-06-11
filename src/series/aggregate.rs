use crate::series::{
    chunked_array::{aggregate::Agg, iterator::ChunkIterator},
    series::Series,
};
use num::{Num, NumCast, ToPrimitive, Zero};

impl Series {
    pub fn sum<T>(&self) -> Option<T>
    where
        T: Num + NumCast + Zero + ToPrimitive,
    {
        match self {
            Series::Bool(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast bool to T")),
            Series::Int32(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast i32 to T")),
            Series::UInt32(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast u32 to T")),
            Series::Int64(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast i64 to T")),
            Series::Float32(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast f32 to T")),
            Series::Float64(a) => a
                .sum()
                .map(|v| T::from(v).expect("could not cast f64 to T")),
            Series::Utf8(_a) => unimplemented!(),
        }
    }
}
