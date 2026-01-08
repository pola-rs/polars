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

impl<'a> FromIterator<Option<&'a str>> for Series {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        let ca: StringChunked = iter.into_iter().collect();
        ca.into_series()
    }
}

impl<'a> FromIterator<&'a str> for Series {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let ca: StringChunked = iter.into_iter().collect();
        ca.into_series()
    }
}

impl FromIterator<Option<String>> for Series {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        let ca: StringChunked = iter.into_iter().collect();
        ca.into_series()
    }
}

impl FromIterator<String> for Series {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let ca: StringChunked = iter.into_iter().collect();
        ca.into_series()
    }
}

pub type SeriesPhysIter<'a> = Box<dyn ExactSizeIterator<Item = AnyValue<'a>> + 'a>;

impl Series {
    /// Iterate over [`Series`] as [`AnyValue`].
    ///
    /// # Panics
    /// This will panic if the array is not rechunked first.
    pub fn iter(&self) -> SeriesIter<'_> {
        let arrays = self.chunks();
        SeriesIter {
            idx_in_cur_arr: 0,
            cur_arr_idx: 0,
            cur_arr_len: arrays[0].len(),
            arrays,
            dtype: self.dtype(),
            total_elems_in_remaining_arrays: self.len(),
        }
    }

    pub fn phys_iter(&self) -> SeriesPhysIter<'_> {
        let dtype = self.dtype();
        let phys_dtype = dtype.to_physical();

        assert_eq!(dtype, &phys_dtype, "impl error");
        assert_eq!(self.chunks().len(), 1, "impl error");
        let arr = &*self.chunks()[0];

        if phys_dtype.is_primitive_numeric() {
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
                DataType::String => {
                    let arr = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                    if arr.null_count() == 0 {
                        Box::new(arr.values_iter().map(AnyValue::String))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    } else {
                        let zipvalid = arr.iter();
                        Box::new(zipvalid.unwrap_optional().map(|v| match v {
                            Some(value) => AnyValue::String(value),
                            None => AnyValue::Null,
                        }))
                            as Box<dyn ExactSizeIterator<Item = AnyValue<'_>> + '_>
                    }
                },
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
                },
                _ => Box::new(self.iter()),
            }
        }
    }
}

pub struct SeriesIter<'a> {
    arrays: &'a [Box<dyn Array>],
    dtype: &'a DataType,
    idx_in_cur_arr: usize,
    cur_arr_len: usize,
    cur_arr_idx: usize,
    total_elems_in_remaining_arrays: usize,
}

impl<'a> Iterator for SeriesIter<'a> {
    type Item = AnyValue<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.idx_in_cur_arr < self.cur_arr_len {
                let arr = unsafe { self.arrays.get_unchecked(self.cur_arr_idx) };
                let ret = unsafe { arr_to_any_value(&**arr, self.idx_in_cur_arr, self.dtype) };
                self.idx_in_cur_arr += 1;
                return Some(ret);
            }

            if self.cur_arr_idx + 1 < self.arrays.len() {
                self.total_elems_in_remaining_arrays -= self.cur_arr_len;
                self.cur_arr_idx += 1;
                self.idx_in_cur_arr = 0;
                let arr = unsafe { self.arrays.get_unchecked(self.cur_arr_idx) };
                self.cur_arr_len = arr.len();
            } else {
                return None;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.total_elems_in_remaining_arrays - self.idx_in_cur_arr;
        (len, Some(len))
    }
}

impl ExactSizeIterator for SeriesIter<'_> {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_iter() {
        let a = Series::new("age".into(), [23, 71, 9].as_ref());
        let _b = a
            .i32()
            .unwrap()
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v * 2));
    }

    #[test]
    fn test_iter_str() {
        let data = [Some("John"), Some("Doe"), None];
        let a: Series = data.into_iter().collect();
        let b = Series::new("".into(), data);
        assert_eq!(a, b);
    }

    #[test]
    fn test_iter_string() {
        let data = [Some("John".to_string()), Some("Doe".to_string()), None];
        let a: Series = data.clone().into_iter().collect();
        let b = Series::new("".into(), data);
        assert_eq!(a, b);
    }
}
