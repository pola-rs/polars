use crate::prelude::any_value::arr_to_any_value;
use crate::prelude::*;
use crate::utils::NoNull;

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
                let ca: NoNull<ChunkedArray<$variant>> = iter.into_iter().collect();
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

#[cfg(feature = "dtype-u8")]
from_iterator!(u8, UInt8Type);
#[cfg(feature = "dtype-u16")]
from_iterator!(u16, UInt16Type);
from_iterator!(u32, UInt32Type);
from_iterator!(u64, UInt64Type);
#[cfg(feature = "dtype-i8")]
from_iterator!(i8, Int8Type);
#[cfg(feature = "dtype-i16")]
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

impl FromIterator<String> for Series {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let ca: Utf8Chunked = iter.into_iter().collect();
        ca.into_series()
    }
}

#[cfg(any(feature = "rows", feature = "dtype-struct"))]
impl Series {
    pub fn iter(&self) -> SeriesIter<'_> {
        let dtype = self.dtype();
        let arr = match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {
                let ca = self.struct_().unwrap();
                &**ca.arrow_array()
            }
            _ => {
                assert_eq!(self.chunks().len(), 1, "impl error");
                &*self.chunks()[0]
            }
        };
        let len = arr.len();
        SeriesIter {
            arr,
            dtype,
            idx: 0,
            len,
        }
    }
}

pub struct SeriesIter<'a> {
    arr: &'a dyn Array,
    dtype: &'a DataType,
    idx: usize,
    len: usize,
}

impl<'a> Iterator for SeriesIter<'a> {
    type Item = AnyValue<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;

        if idx == self.len {
            None
        } else {
            unsafe { Some(arr_to_any_value(self.arr, idx, self.dtype)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl ExactSizeIterator for SeriesIter<'_> {}

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
