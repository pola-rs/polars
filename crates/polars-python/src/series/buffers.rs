//! Construct and deconstruct Series based on the underlying buffers.
//!
//! This functionality is mainly intended for use with the Python dataframe
//! interchange protocol.
//!
//! As Polars has no Buffer concept in Python, each buffer is represented as
//! a Series of its physical type.
//!
//! Note that String Series have underlying `Utf8View` buffers, which
//! currently cannot be represented as Series. Since the interchange protocol
//! cannot handle these buffers anyway and expects bytes and offsets buffers,
//! operations on String Series will convert from/to such buffers. This
//! conversion requires data to be copied.

use polars::export::arrow;
use polars::export::arrow::array::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use polars::export::arrow::bitmap::Bitmap;
use polars::export::arrow::buffer::Buffer;
use polars::export::arrow::offset::OffsetsBuffer;
use polars::export::arrow::types::NativeType;
use polars::prelude::*;
use polars_core::{with_match_physical_numeric_polars_type, with_match_physical_numeric_type};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::{PySeries, ToSeries};
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::raise_err;

struct BufferInfo {
    pointer: usize,
    offset: usize,
    length: usize,
}
impl IntoPy<PyObject> for BufferInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (self.pointer, self.offset, self.length).to_object(py)
    }
}
impl<'a> FromPyObject<'a> for BufferInfo {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let (pointer, offset, length) = ob.extract()?;
        Ok(Self {
            pointer,
            offset,
            length,
        })
    }
}

#[pymethods]
impl PySeries {
    /// Return pointer, offset, and length information about the underlying buffer.
    fn _get_buffer_info(&self) -> PyResult<BufferInfo> {
        let s = self.series.to_physical_repr();
        let arrays = s.chunks();
        if arrays.len() != 1 {
            let msg = "cannot get buffer info for Series consisting of multiple chunks";
            raise_err!(msg, ComputeError);
        }
        match s.dtype() {
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                let (slice, offset, len) = arr.values().as_slice();
                Ok(BufferInfo {
                    pointer: slice.as_ptr() as usize,
                    offset,
                    length: len,
                })
            },
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                BufferInfo { pointer: get_pointer(ca), offset: 0, length: ca.len() }
            })),
            dt => {
                let msg = format!("`_get_buffer_info` not implemented for non-physical type {dt}; try to select a buffer first");
                Err(PyTypeError::new_err(msg))
            },
        }
    }

    /// Return the underlying values, validity, and offsets buffers as Series.
    fn _get_buffers(&self) -> PyResult<(Self, Option<Self>, Option<Self>)> {
        let s = &self.series;
        match s.dtype().to_physical() {
            dt if dt.is_numeric() => get_buffers_from_primitive(s),
            DataType::Boolean => get_buffers_from_primitive(s),
            DataType::String => get_buffers_from_string(s),
            dt => {
                let msg = format!("`_get_buffers` not implemented for `dtype` {dt}");
                Err(PyTypeError::new_err(msg))
            },
        }
    }
}

fn get_pointer<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
}

fn get_buffers_from_primitive(
    s: &Series,
) -> PyResult<(PySeries, Option<PySeries>, Option<PySeries>)> {
    let chunks = s
        .chunks()
        .iter()
        .map(|arr| arr.with_validity(None))
        .collect::<Vec<_>>();
    let values = Series::try_from((s.name(), chunks))
        .map_err(PyPolarsErr::from)?
        .into();

    let validity = get_bitmap(s);
    let offsets = None;
    Ok((values, validity, offsets))
}

/// The underlying buffers for `String` Series cannot be represented in this
/// format. Instead, the buffers are converted to a values and offsets buffer.
/// This copies data.
fn get_buffers_from_string(s: &Series) -> PyResult<(PySeries, Option<PySeries>, Option<PySeries>)> {
    // We cannot do this zero copy anyway, so rechunk first
    let s = s.rechunk();

    let ca = s.str().map_err(PyPolarsErr::from)?;
    let arr_binview = ca.downcast_iter().next().unwrap();

    // This is not zero-copy
    let arr_utf8 = arrow::compute::cast::utf8view_to_utf8(arr_binview);

    let values = get_string_bytes(&arr_utf8)?;
    let validity = get_bitmap(&s);
    let offsets = get_string_offsets(&arr_utf8)?;

    Ok((values, validity, Some(offsets)))
}

fn get_bitmap(s: &Series) -> Option<PySeries> {
    if s.null_count() > 0 {
        Some(s.is_not_null().into_series().into())
    } else {
        None
    }
}

fn get_string_bytes(arr: &Utf8Array<i64>) -> PyResult<PySeries> {
    let values_buffer = arr.values();
    let values_arr =
        PrimitiveArray::<u8>::try_new(ArrowDataType::UInt8, values_buffer.clone(), None)
            .map_err(PyPolarsErr::from)?;
    let values = Series::from_arrow("", values_arr.to_boxed())
        .map_err(PyPolarsErr::from)?
        .into();
    Ok(values)
}

fn get_string_offsets(arr: &Utf8Array<i64>) -> PyResult<PySeries> {
    let offsets_buffer = arr.offsets().buffer();
    let offsets_arr =
        PrimitiveArray::<i64>::try_new(ArrowDataType::Int64, offsets_buffer.clone(), None)
            .map_err(PyPolarsErr::from)?;
    let offsets = Series::from_arrow("", offsets_arr.to_boxed())
        .map_err(PyPolarsErr::from)?
        .into();
    Ok(offsets)
}

#[pymethods]
impl PySeries {
    /// Construct a PySeries from information about its underlying buffer.
    #[staticmethod]
    unsafe fn _from_buffer(
        py: Python,
        dtype: Wrap<DataType>,
        buffer_info: BufferInfo,
        owner: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let BufferInfo {
            pointer,
            offset,
            length,
        } = buffer_info;
        let owner = owner.to_object(py);

        let arr_boxed = match dtype {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_type!(dt, |$T|  unsafe {
                    from_buffer_impl::<$T>(pointer, offset, length, owner)
                })
            },
            DataType::Boolean => {
                unsafe { from_buffer_boolean_impl(pointer, offset, length, owner) }?
            },
            dt => {
                let msg = format!(
                    "`_from_buffer` requires a physical type as input for `dtype`, got {dt}"
                );
                return Err(PyTypeError::new_err(msg));
            },
        };

        let s = Series::from_arrow("", arr_boxed).unwrap().into();
        Ok(s)
    }
}

unsafe fn from_buffer_impl<T: NativeType>(
    pointer: usize,
    offset: usize,
    length: usize,
    owner: Py<PyAny>,
) -> Box<dyn Array> {
    let pointer = pointer as *const T;
    let pointer = unsafe { pointer.add(offset) };
    let slice = unsafe { std::slice::from_raw_parts(pointer, length) };
    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(slice, owner) };
    arr.to_boxed()
}
unsafe fn from_buffer_boolean_impl(
    pointer: usize,
    offset: usize,
    length: usize,
    owner: Py<PyAny>,
) -> PyResult<Box<dyn Array>> {
    let length_in_bytes = get_boolean_buffer_length_in_bytes(length, offset);

    let pointer = pointer as *const u8;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length_in_bytes) };
    let arr_result = unsafe { arrow::ffi::mmap::bitmap_and_owner(slice, offset, length, owner) };
    let arr = arr_result.map_err(PyPolarsErr::from)?;
    Ok(arr.to_boxed())
}
fn get_boolean_buffer_length_in_bytes(length: usize, offset: usize) -> usize {
    let n_bits = offset + length;
    let n_bytes = n_bits / 8;
    let rest = n_bits % 8;
    if rest == 0 {
        n_bytes
    } else {
        n_bytes + 1
    }
}

#[pymethods]
impl PySeries {
    /// Construct a PySeries from information about its underlying buffers.
    #[staticmethod]
    unsafe fn _from_buffers(
        dtype: Wrap<DataType>,
        data: Vec<PySeries>,
        validity: Option<PySeries>,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let mut data = data.to_series();

        match data.len() {
            0 => {
                let msg = "`data` input to `_from_buffers` must contain at least one buffer";
                return Err(PyTypeError::new_err(msg));
            },
            1 if validity.is_none() => {
                let values = data.pop().unwrap();
                let s = values.strict_cast(&dtype).map_err(PyPolarsErr::from)?;
                return Ok(s.into());
            },
            _ => (),
        }

        let validity = match validity {
            Some(s) => {
                let dtype = s.series.dtype();
                if !dtype.is_bool() {
                    let msg = format!(
                        "validity buffer must have data type Boolean, got {:?}",
                        dtype
                    );
                    return Err(PyTypeError::new_err(msg));
                }
                Some(series_to_bitmap(s.series).unwrap())
            },
            None => None,
        };

        let s = match dtype.to_physical() {
            dt if dt.is_numeric() => {
                let values = data.into_iter().next().unwrap();
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let values_buffer = series_to_buffer::<$T>(values);
                    from_buffers_num_impl::<<$T as PolarsNumericType>::Native>(values_buffer, validity)?
                })
            },
            DataType::Boolean => {
                let values = data.into_iter().next().unwrap();
                let values_buffer = series_to_bitmap(values)?;
                from_buffers_bool_impl(values_buffer, validity)?
            },
            DataType::String => {
                let mut data_iter = data.into_iter();
                let values = data_iter.next().unwrap();
                let offsets = match data_iter.next() {
                    Some(s) => {
                        let dtype = s.dtype();
                        if !matches!(dtype, DataType::Int64) {
                            return Err(PyTypeError::new_err(format!(
                                "offsets buffer must have data type Int64, got {:?}",
                                dtype
                            )));
                        }
                        series_to_offsets(s)
                    },
                    None => return Err(PyTypeError::new_err(
                        "`_from_buffers` cannot create a String column without an offsets buffer",
                    )),
                };
                let values = series_to_buffer::<UInt8Type>(values);
                from_buffers_string_impl(values, validity, offsets)?
            },
            dt => {
                let msg = format!("`_from_buffers` not implemented for `dtype` {dt}");
                return Err(PyTypeError::new_err(msg));
            },
        };

        let out = s.strict_cast(&dtype).map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}

fn series_to_buffer<T>(s: Series) -> Buffer<T::Native>
where
    T: PolarsNumericType,
{
    let ca: &ChunkedArray<T> = s.as_ref().as_ref();
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().clone()
}
fn series_to_bitmap(s: Series) -> PyResult<Bitmap> {
    let ca_result = s.bool();
    let ca = ca_result.map_err(PyPolarsErr::from)?;
    let arr = ca.downcast_iter().next().unwrap();
    let bitmap = arr.values().clone();
    Ok(bitmap)
}
fn series_to_offsets(s: Series) -> OffsetsBuffer<i64> {
    let buffer = series_to_buffer::<Int64Type>(s);
    unsafe { OffsetsBuffer::new_unchecked(buffer) }
}

fn from_buffers_num_impl<T: NativeType>(
    data: Buffer<T>,
    validity: Option<Bitmap>,
) -> PyResult<Series> {
    let arr = PrimitiveArray::new(T::PRIMITIVE.into(), data, validity);
    let s_result = Series::from_arrow("", arr.to_boxed());
    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
fn from_buffers_bool_impl(data: Bitmap, validity: Option<Bitmap>) -> PyResult<Series> {
    let arr = BooleanArray::new(ArrowDataType::Boolean, data, validity);
    let s_result = Series::from_arrow("", arr.to_boxed());
    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
/// Constructing a `String` Series requires specifying a values and offsets buffer,
/// which does not match the actual underlying buffers. The values and offsets
/// buffer are converted into the actual buffers, which copies data.
fn from_buffers_string_impl(
    data: Buffer<u8>,
    validity: Option<Bitmap>,
    offsets: OffsetsBuffer<i64>,
) -> PyResult<Series> {
    let arr = Utf8Array::new(ArrowDataType::LargeUtf8, offsets, data, validity);

    // This is not zero-copy
    let s_result = Series::from_arrow("", arr.to_boxed());

    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
