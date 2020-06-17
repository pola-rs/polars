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
            Series::Date64(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Date64 to T")),
            Series::Time64Ns(a) => a
                .$agg()
                .map(|v| T::from(v).expect("could not cast Time64Ns to T")),
            Series::Utf8(_a) => unimplemented!(),
        }
    };
}

impl Series {
    fn sum<T>(&self) -> Option<T>
    where
        T: Num + NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, sum)
    }

    fn min<T>(&self) -> Option<T>
    where
        T: Num + NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, min)
    }

    fn max<T>(&self) -> Option<T>
    where
        T: Num + NumCast + Zero + ToPrimitive,
    {
        apply_agg_fn!(self, max)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_agg_bool() {
        let s = Series::init("", vec![true, false, true].as_slice());
        assert_eq!(s.max::<u8>(), Some(1));
        assert_eq!(s.min::<u8>(), Some(0));
    }
}
