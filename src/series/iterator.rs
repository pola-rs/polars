use crate::prelude::*;
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::FromIterator;
use std::mem;

macro_rules! from_iterator {
    ($native:ty, $variant:ident) => {
        impl FromIterator<Option<$native>> for Series {
            fn from_iter<I: IntoIterator<Item = Option<$native>>>(iter: I) -> Self {
                let ca = iter.into_iter().collect();
                Series::$variant(ca)
            }
        }

        impl FromIterator<$native> for Series {
            fn from_iter<I: IntoIterator<Item = $native>>(iter: I) -> Self {
                let ca = iter.into_iter().map(|v| Some(v)).collect();
                Series::$variant(ca)
            }
        }

        impl<'a> FromIterator<&'a $native> for Series {
            fn from_iter<I: IntoIterator<Item = &'a $native>>(iter: I) -> Self {
                let ca = iter.into_iter().map(|v| Some(*v)).collect();
                Series::$variant(ca)
            }
        }
    };
}

from_iterator!(u32, UInt32);
from_iterator!(i32, Int32);
from_iterator!(i64, Int64);
from_iterator!(f32, Float32);
from_iterator!(f64, Float64);
from_iterator!(bool, Bool);

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_iter() {
        let a = Series::init("age", [23, 71, 9].as_ref());
        let b = a.i32().unwrap().iter().map(|opt_v| opt_v.map(|v| v * 2));
    }
}
