//! Implementations of the ChunkCast Trait.
use std::convert::TryFrom;

use arrow::compute::cast::CastOptions;
use polars_arrow::compute::cast;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::CategoricalChunkedBuilder;
use crate::prelude::*;

pub(crate) fn cast_chunks(
    chunks: &[ArrayRef],
    dtype: &DataType,
    checked: bool,
) -> PolarsResult<Vec<ArrayRef>> {
    let options = if checked {
        Default::default()
    } else {
        CastOptions {
            wrapped: true,
            partial: false,
        }
    };

    let arrow_dtype = dtype.to_arrow();
    let chunks = chunks
        .iter()
        .map(|arr| arrow::compute::cast::cast(arr.as_ref(), &arrow_dtype, options))
        .collect::<arrow::error::Result<Vec<_>>>()?;
    Ok(chunks)
}

fn cast_impl_inner(
    name: &str,
    chunks: &[ArrayRef],
    dtype: &DataType,
    checked: bool,
) -> PolarsResult<Series> {
    let chunks = cast_chunks(chunks, &dtype.to_physical(), checked)?;
    let out = Series::try_from((name, chunks))?;
    use DataType::*;
    let out = match dtype {
        Date => out.into_date(),
        Datetime(tu, tz) => out.into_datetime(*tu, tz.clone()),
        Duration(tu) => out.into_duration(*tu),
        #[cfg(feature = "dtype-time")]
        Time => out.into_time(),
        _ => out,
    };

    Ok(out)
}

fn cast_impl(name: &str, chunks: &[ArrayRef], dtype: &DataType) -> PolarsResult<Series> {
    cast_impl_inner(name, chunks, dtype, true)
}

#[cfg(feature = "dtype-struct")]
fn cast_single_to_struct(
    name: &str,
    chunks: &[ArrayRef],
    fields: &[Field],
) -> PolarsResult<Series> {
    let mut new_fields = Vec::with_capacity(fields.len());
    // cast to first field dtype
    let mut fields = fields.iter();
    let fld = fields.next().unwrap();
    let s = cast_impl_inner(&fld.name, chunks, &fld.dtype, true)?;
    let length = s.len();
    new_fields.push(s);

    for fld in fields {
        new_fields.push(Series::full_null(&fld.name, length, &fld.dtype));
    }

    Ok(StructChunked::new_unchecked(name, &new_fields).into_series())
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_impl(&self, data_type: &DataType, checked: bool) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                polars_ensure!(
                    self.dtype() == &DataType::UInt32,
                    ComputeError: "cannot cast numeric types to 'Categorical'"
                );
                // SAFETY
                // we are guarded by the type system
                let ca = unsafe { &*(self as *const ChunkedArray<T> as *const UInt32Chunked) };
                CategoricalChunked::from_global_indices(ca.clone()).map(|ca| ca.into_series())
            }
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl_inner(self.name(), &self.chunks, data_type, checked).map(|mut s| {
                // maintain sorted if data types remain signed
                // this may still fail with overflow?
                if ((self.dtype().is_signed() && data_type.is_signed())
                    || (self.dtype().is_unsigned() && data_type.is_unsigned()))
                    && (s.null_count() == self.null_count())
                {
                    let is_sorted = self.is_sorted_flag2();
                    s.set_sorted_flag(is_sorted)
                }
                s
            }),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast_impl(data_type, true)
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(Some(rev_map)) => {
                if self.dtype() == &DataType::UInt32 {
                    // safety:
                    // we are guarded by the type system.
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const UInt32Chunked) };
                    Ok(unsafe {
                        CategoricalChunked::from_cats_and_rev_map_unchecked(
                            ca.clone(),
                            rev_map.clone(),
                        )
                    }
                    .into_series())
                } else {
                    polars_bail!(ComputeError: "cannot cast numeric types to 'Categorical'");
                }
            }
            _ => self.cast_impl(data_type, false),
        }
    }
}

impl ChunkCast for Utf8Chunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let iter = self.into_iter();
                let mut builder = CategoricalChunkedBuilder::new(self.name(), self.len());
                builder.drain_iter(iter);
                let ca = builder.finish();
                Ok(ca.into_series())
            }
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

unsafe fn binary_to_utf8_unchecked(from: &BinaryArray<i64>) -> Utf8Array<i64> {
    let values = from.values().clone();
    let offsets = from.offsets().clone();
    Utf8Array::<i64>::try_new_unchecked(
        ArrowDataType::LargeUtf8,
        offsets,
        values,
        from.validity().cloned(),
    )
    .unwrap()
}

impl BinaryChunked {
    /// # Safety
    /// Utf8 is not validated
    pub unsafe fn to_utf8(&self) -> Utf8Chunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| Box::new(binary_to_utf8_unchecked(arr)) as ArrayRef)
            .collect();
        Utf8Chunked::from_chunks(self.name(), chunks)
    }
}

impl Utf8Chunked {
    pub fn as_binary(&self) -> BinaryChunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                Box::new(arrow::compute::cast::utf8_to_binary(
                    arr,
                    ArrowDataType::LargeBinary,
                )) as ArrayRef
            })
            .collect();
        unsafe { BinaryChunked::from_chunks(self.name(), chunks) }
    }
}

impl ChunkCast for BinaryChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::Utf8 => unsafe { Ok(self.to_utf8().into_series()) },
            _ => self.cast(data_type),
        }
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
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::Utf8 => {
                let mut ca = boolean_to_utf8(self);
                ca.rename(self.name());
                Ok(ca.into_series())
            }
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

fn cast_inner_list_type(list: &ListArray<i64>, child_type: &DataType) -> PolarsResult<ArrayRef> {
    let child = list.values();
    let offsets = list.offsets();
    let child = cast::cast(child.as_ref(), &child_type.to_arrow())?;

    let data_type = ListArray::<i64>::default_datatype(child_type.to_arrow());
    let list = ListArray::new(data_type, offsets.clone(), child, list.validity().cloned());
    Ok(Box::new(list) as ArrayRef)
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match data_type {
            List(child_type) => {
                let phys_child = child_type.to_physical();

                match (self.inner_dtype(), &**child_type) {
                    #[cfg(feature = "dtype-categorical")]
                    (Utf8, Categorical(_)) => {
                        let (arr, child_type) = cast_list(self, child_type)?;
                        Ok(unsafe {
                            Series::from_chunks_and_dtype_unchecked(
                                self.name(),
                                vec![arr],
                                &List(Box::new(child_type)),
                            )
                        })
                    }
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(None)) => {
                        polars_bail!(ComputeError: "cannot cast list inner type: '{:?}' to Categorical", dt)
                    }
                    _ if phys_child.is_primitive() => {
                        let mut ca = if child_type.to_physical() != self.inner_dtype().to_physical()
                        {
                            let chunks = self
                                .downcast_iter()
                                .map(|list| cast_inner_list_type(list, &phys_child))
                                .collect::<PolarsResult<_>>()?;
                            unsafe { ListChunked::from_chunks(self.name(), chunks) }
                        } else {
                            self.clone()
                        };
                        ca.set_inner_dtype(*child_type.clone());
                        Ok(ca.into_series())
                    }
                    _ => {
                        let arr = cast_list(self, child_type)?.0;
                        Series::try_from((self.name(), arr))
                    }
                }
            }
            _ => polars_bail!(ComputeError: "cannot cast list type"),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
fn cast_list(ca: &ListChunked, child_type: &DataType) -> PolarsResult<(ArrayRef, DataType)> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let s = Series::try_from(("", arr.values().clone())).unwrap();
    let new_inner = s.cast(child_type)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.array_ref(0).clone();

    let data_type = ListArray::<i64>::default_datatype(new_values.data_type().clone());
    let new_arr = ListArray::<i64>::new(
        data_type,
        arr.offsets().clone(),
        new_values,
        arr.validity().cloned(),
    );
    Ok((Box::new(new_arr), inner_dtype))
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> PolarsResult<()> {
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        let ca = builder.finish();

        let new = ca.cast(&DataType::List(DataType::Float64.into()))?;

        assert_eq!(new.dtype(), &DataType::List(DataType::Float64.into()));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_cast_noop() {
        // check if we can cast categorical twice without panic
        let ca = Utf8Chunked::new("foo", &["bar", "ham"]);
        let out = ca.cast(&DataType::Categorical(None)).unwrap();
        let out = out.cast(&DataType::Categorical(None)).unwrap();
        assert!(matches!(out.dtype(), &DataType::Categorical(_)))
    }
}
