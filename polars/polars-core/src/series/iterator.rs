use crate::prelude::*;
use crate::utils::Xob;
use std::iter::FromIterator;

macro_rules! from_iterator {
    ($native:ty, $variant:ident) => {
        impl FromIterator<Option<$native>> for Series {
            fn from_iter<I: IntoIterator<Item = Option<$native>>>(iter: I) -> Self {
                let ca: ChunkedArray<$variant> = iter.into_iter().collect();
                ca.into_series()
            }
        }

        impl FromIterator<$native> for Series {
            fn from_iter<I: IntoIterator<Item = $native>>(iter: I) -> Self {
                let ca: Xob<ChunkedArray<$variant>> = iter.into_iter().collect();
                ca.into_inner().into_series()
            }
        }

        impl<'a> FromIterator<&'a $native> for Series {
            fn from_iter<I: IntoIterator<Item = &'a $native>>(iter: I) -> Self {
                let ca: ChunkedArray<$variant> = iter.into_iter().map(|v| Some(*v)).collect();
                ca.into_series()
            }
        }
    };
}

from_iterator!(u8, UInt8Type);
from_iterator!(u16, UInt16Type);
from_iterator!(u32, UInt32Type);
from_iterator!(u64, UInt64Type);
from_iterator!(i8, Int8Type);
from_iterator!(i16, Int16Type);
from_iterator!(i32, Int32Type);
from_iterator!(i64, Int64Type);
from_iterator!(f32, Float32Type);
from_iterator!(f64, Float64Type);
from_iterator!(bool, BooleanType);

impl<'a> FromIterator<&'a str> for Series {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let ca: Utf8Chunked = iter.into_iter().collect();
        ca.into_series()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_iter() {
        let a = Series::new("age", [23, 71, 9].as_ref());
        let _b = a
            .i32()
            .unwrap()
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v * 2));
    }
}
