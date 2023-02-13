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

pub type SeriesPhysIter<'a> = Box<dyn ExactSizeIterator<Item = AnyValue<'a>> + 'a>;

#[cfg(any(feature = "rows", feature = "dtype-struct"))]
impl Series {
    /// iterate over [`Series`] as [`AnyValue`].
    ///
    /// # Panics
    /// This will panic if the array is not rechunked first.
    pub fn iter(&self) -> SeriesIter<'_> {
        let dtype = self.dtype();
        assert_eq!(self.chunks().len(), 1, "impl error");
        let arr = &*self.chunks()[0];
        let len = arr.len();
        SeriesIter {
            arr,
            dtype,
            idx: 0,
            len,
        }
    }

    pub fn phys_iter(&self) -> SeriesPhysIter<'_> {
        let dtype = self.dtype();
        let phys_dtype = dtype.to_physical();

        assert_eq!(dtype, &phys_dtype, "impl error");
        assert_eq!(self.chunks().len(), 1, "impl error");
        let arr = &*self.chunks()[0];

        if phys_dtype.is_numeric() {
            if arr.null_count() == 0 {
                with_match_physical_numeric_type!(phys_dtype, |$T| {
                        let arr = arr.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        let values = arr.values().as_slice();
                        Box::new(values.iter().map(|&value| AnyValue::from(value))) as Box<dyn ExactSizeIterator<Item=AnyValue<'_>> + '_>
                })
            } else {
                with_match_physical_numeric_type!(phys_dtype, |$T| {
                        let arr = arr.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                        Box::new(arr.iter().map(|value| {

                        match value {
                            Some(value) => AnyValue::from(*value),
                            None => AnyValue::Null
                        }

                    })) as Box<dyn ExactSizeIterator<Item=AnyValue<'_>> + '_>
                })
            }
        } else {
            match dtype {
                DataType::Utf8 => {
                    let arr = arr.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                    if arr.null_count() == 0 {
                        Box::new(arr.values_iter().map(AnyValue::Utf8))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    } else {
                        let zipvalid = arr.iter();
                        Box::new(zipvalid.unwrap_optional().map(|v| match v {
                            Some(value) => AnyValue::Utf8(value),
                            None => AnyValue::Null,
                        }))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    }
                }
                DataType::Boolean => {
                    let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                    if arr.null_count() == 0 {
                        Box::new(arr.values_iter().map(AnyValue::Boolean))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    } else {
                        let zipvalid = arr.iter();
                        Box::new(zipvalid.unwrap_optional().map(|v| match v {
                            Some(value) => AnyValue::Boolean(value),
                            None => AnyValue::Null,
                        }))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    }
                }
                _ => Box::new(self.iter()),
            }
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;

        if idx == self.len {
            None
        } else {
            self.idx += 1;
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
