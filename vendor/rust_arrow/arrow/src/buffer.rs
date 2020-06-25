// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The main type in the module is `Buffer`, a contiguous immutable memory region of
//! fixed size aligned at a 64-byte boundary. `MutableBuffer` is like `Buffer`, but it can
//! be mutated and grown.
#[cfg(feature = "simd")]
use packed_simd::u8x64;

use std::cmp;
use std::convert::AsRef;
use std::fmt::{Debug, Formatter};
use std::io::{Error as IoError, ErrorKind, Result as IoResult, Write};
use std::mem;
use std::ops::{BitAnd, BitOr, Not};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;

use crate::array::{BufferBuilderTrait, UInt8BufferBuilder};
use crate::datatypes::ArrowNativeType;
use crate::error::{ArrowError, Result};
use crate::memory;
use crate::util::bit_util;

/// Buffer is a contiguous memory region of fixed size and is aligned at a 64-byte
/// boundary. Buffer is immutable.
#[derive(PartialEq, Debug)]
pub struct Buffer {
    /// Reference-counted pointer to the internal byte buffer.
    data: Arc<BufferData>,

    /// The offset into the buffer.
    offset: usize,
}

struct BufferData {
    /// The raw pointer into the buffer bytes
    ptr: *const u8,

    /// The length (num of bytes) of the buffer. The region `[0, len)` of the buffer
    /// is occupied with meaningful data, while the rest `[len, capacity)` is the
    /// unoccupied region.
    len: usize,

    /// Whether this piece of memory is owned by this object
    owned: bool,

    /// The capacity (num of bytes) of the buffer
    /// Invariant: len <= capacity
    capacity: usize,
}

impl PartialEq for BufferData {
    fn eq(&self, other: &BufferData) -> bool {
        if self.capacity != other.capacity {
            return false;
        }

        self.data() == other.data()
    }
}

/// Release the underlying memory when the current buffer goes out of scope
impl Drop for BufferData {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.owned {
            unsafe { memory::free_aligned(self.ptr as *mut u8, self.capacity) };
        }
    }
}

impl Debug for BufferData {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "BufferData {{ ptr: {:?}, len: {}, capacity: {}, data: ",
            self.ptr, self.len, self.capacity
        )?;

        f.debug_list().entries(self.data().iter()).finish()?;

        write!(f, " }}")
    }
}

impl BufferData {
    fn data(&self) -> &[u8] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Buffer {
    /// Creates a buffer from an existing memory region (must already be byte-aligned), this
    /// `Buffer` will free this piece of memory when dropped.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to raw parts
    /// * `len` - Length of raw parts in **bytes**
    /// * `capacity` - Total allocated memory for the pointer `ptr`, in **bytes**
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// bytes.
    pub unsafe fn from_raw_parts(ptr: *const u8, len: usize, capacity: usize) -> Self {
        Buffer::build_with_arguments(ptr, len, capacity, true)
    }

    /// Creates a buffer from an existing memory region (must already be byte-aligned), this
    /// `Buffer` **does not** free this piece of memory when dropped.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to raw parts
    /// * `len` - Length of raw parts in **bytes**
    /// * `capacity` - Total allocated memory for the pointer `ptr`, in **bytes**
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// bytes.
    pub unsafe fn from_unowned(ptr: *const u8, len: usize, capacity: usize) -> Self {
        Buffer::build_with_arguments(ptr, len, capacity, false)
    }

    /// Creates a buffer from an existing memory region (must already be byte-aligned).
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to raw parts
    /// * `len` - Length of raw parts in bytes
    /// * `capacity` - Total allocated memory for the pointer `ptr`, in **bytes**
    /// * `owned` - Whether the raw parts is owned by this `Buffer`. If true, this `Buffer` will
    /// free this memory when dropped, otherwise it will skip freeing the raw parts.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// bytes.
    unsafe fn build_with_arguments(
        ptr: *const u8,
        len: usize,
        capacity: usize,
        owned: bool,
    ) -> Self {
        assert!(
            memory::is_aligned(ptr, memory::ALIGNMENT),
            "memory not aligned"
        );
        let buf_data = BufferData {
            ptr,
            len,
            capacity,
            owned,
        };
        Buffer {
            data: Arc::new(buf_data),
            offset: 0,
        }
    }

    /// Returns the number of bytes in the buffer
    pub fn len(&self) -> usize {
        self.data.len - self.offset
    }

    /// Returns the capacity of this buffer
    pub fn capacity(&self) -> usize {
        self.data.capacity
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.len - self.offset == 0
    }

    /// Returns the byte slice stored in this buffer
    pub fn data(&self) -> &[u8] {
        &self.data.data()[self.offset..]
    }

    /// Returns a slice of this buffer, starting from `offset`.
    pub fn slice(&self, offset: usize) -> Self {
        assert!(
            offset <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        Self {
            data: self.data.clone(),
            offset: self.offset + offset,
        }
    }

    /// Returns a raw pointer for this buffer.
    ///
    /// Note that this should be used cautiously, and the returned pointer should not be
    /// stored anywhere, to avoid dangling pointers.
    pub fn raw_data(&self) -> *const u8 {
        unsafe { self.data.ptr.add(self.offset) }
    }

    /// View buffer as typed slice.
    ///
    /// # Safety
    ///
    /// `ArrowNativeType` is public so that it can be used as a trait bound for other public
    /// components, such as the `ToByteSlice` trait.  However, this means that it can be
    /// implemented by user defined types, which it is not intended for.
    ///
    /// Also `typed_data::<bool>` is unsafe as `0x00` and `0x01` are the only valid values for
    /// `bool` in Rust.  However, `bool` arrays in Arrow are bit-packed which breaks this condition.
    pub unsafe fn typed_data<T: ArrowNativeType + num::Num>(&self) -> &[T] {
        assert_eq!(self.len() % mem::size_of::<T>(), 0);
        assert!(memory::is_ptr_aligned::<T>(self.raw_data() as *const T));
        from_raw_parts(
            self.raw_data() as *const T,
            self.len() / mem::size_of::<T>(),
        )
    }

    /// Returns an empty buffer.
    pub fn empty() -> Self {
        unsafe { Self::from_raw_parts(::std::ptr::null(), 0, 0) }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        Buffer {
            data: self.data.clone(),
            offset: self.offset,
        }
    }
}

/// Creating a `Buffer` instance by copying the memory from a `AsRef<[u8]>` into a newly
/// allocated memory region.
impl<T: AsRef<[u8]>> From<T> for Buffer {
    fn from(p: T) -> Self {
        // allocate aligned memory buffer
        let slice = p.as_ref();
        let len = slice.len() * mem::size_of::<u8>();
        let capacity = bit_util::round_upto_multiple_of_64(len);
        let buffer = memory::allocate_aligned(capacity);
        unsafe {
            memory::memcpy(buffer, slice.as_ptr(), len);
            Buffer::from_raw_parts(buffer, len, capacity)
        }
    }
}

///  Helper function for SIMD `BitAnd` and `BitOr` implementations
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
fn bitwise_bin_op_simd_helper<F>(left: &Buffer, right: &Buffer, op: F) -> Buffer
where
    F: Fn(u8x64, u8x64) -> u8x64,
{
    let mut result = MutableBuffer::new(left.len()).with_bitset(left.len(), false);
    let lanes = u8x64::lanes();
    for i in (0..left.len()).step_by(lanes) {
        let left_data = unsafe { from_raw_parts(left.raw_data().add(i), lanes) };
        let right_data = unsafe { from_raw_parts(right.raw_data().add(i), lanes) };
        let result_slice: &mut [u8] = unsafe {
            from_raw_parts_mut((result.data_mut().as_mut_ptr() as *mut u8).add(i), lanes)
        };
        unsafe {
            bit_util::bitwise_bin_op_simd(&left_data, &right_data, result_slice, &op)
        };
    }
    result.freeze()
}

impl<'a, 'b> BitAnd<&'b Buffer> for &'a Buffer {
    type Output = Result<Buffer>;

    fn bitand(self, rhs: &'b Buffer) -> Result<Buffer> {
        if self.len() != rhs.len() {
            return Err(ArrowError::ComputeError(
                "Buffers must be the same size to apply Bitwise AND.".to_string(),
            ));
        }

        // SIMD implementation if available
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        {
            return Ok(bitwise_bin_op_simd_helper(&self, &rhs, |a, b| a & b));
        }

        // Default implementation
        #[allow(unreachable_code)]
        {
            let mut builder = UInt8BufferBuilder::new(self.len());
            for i in 0..self.len() {
                unsafe {
                    builder
                        .append(
                            self.data().get_unchecked(i) & rhs.data().get_unchecked(i),
                        )
                        .unwrap();
                }
            }
            Ok(builder.finish())
        }
    }
}

impl<'a, 'b> BitOr<&'b Buffer> for &'a Buffer {
    type Output = Result<Buffer>;

    fn bitor(self, rhs: &'b Buffer) -> Result<Buffer> {
        if self.len() != rhs.len() {
            return Err(ArrowError::ComputeError(
                "Buffers must be the same size to apply Bitwise OR.".to_string(),
            ));
        }

        // SIMD implementation if available
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        {
            return Ok(bitwise_bin_op_simd_helper(&self, &rhs, |a, b| a | b));
        }

        // Default implementation
        #[allow(unreachable_code)]
        {
            let mut builder = UInt8BufferBuilder::new(self.len());
            for i in 0..self.len() {
                unsafe {
                    builder
                        .append(
                            self.data().get_unchecked(i) | rhs.data().get_unchecked(i),
                        )
                        .unwrap();
                }
            }
            Ok(builder.finish())
        }
    }
}

impl Not for &Buffer {
    type Output = Buffer;

    fn not(self) -> Buffer {
        // SIMD implementation if available
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        {
            let mut result =
                MutableBuffer::new(self.len()).with_bitset(self.len(), false);
            let lanes = u8x64::lanes();
            for i in (0..self.len()).step_by(lanes) {
                unsafe {
                    let data = from_raw_parts(self.raw_data().add(i), lanes);
                    let data_simd = u8x64::from_slice_unaligned_unchecked(data);
                    let simd_result = !data_simd;
                    let result_slice: &mut [u8] = from_raw_parts_mut(
                        (result.data_mut().as_mut_ptr() as *mut u8).add(i),
                        lanes,
                    );
                    simd_result.write_to_slice_unaligned_unchecked(result_slice);
                }
            }
            return result.freeze();
        }

        // Default implementation
        #[allow(unreachable_code)]
        {
            let mut builder = UInt8BufferBuilder::new(self.len());
            for i in 0..self.len() {
                unsafe {
                    builder.append(!self.data().get_unchecked(i)).unwrap();
                }
            }
            builder.finish()
        }
    }
}

unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}

/// Similar to `Buffer`, but is growable and can be mutated. A mutable buffer can be
/// converted into a immutable buffer via the `freeze` method.
#[derive(Debug)]
pub struct MutableBuffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

impl MutableBuffer {
    /// Allocate a new mutable buffer with initial capacity to be `capacity`.
    pub fn new(capacity: usize) -> Self {
        let new_capacity = bit_util::round_upto_multiple_of_64(capacity);
        let ptr = memory::allocate_aligned(new_capacity);
        Self {
            data: ptr,
            len: 0,
            capacity: new_capacity,
        }
    }

    /// Set the bits in the range of `[0, end)` to 0 (if `val` is false), or 1 (if `val`
    /// is true). Also extend the length of this buffer to be `end`.
    ///
    /// This is useful when one wants to clear (or set) the bits and then manipulate
    /// the buffer directly (e.g., modifying the buffer by holding a mutable reference
    /// from `data_mut()`).
    pub fn with_bitset(mut self, end: usize, val: bool) -> Self {
        assert!(end <= self.capacity);
        let v = if val { 255 } else { 0 };
        unsafe {
            std::ptr::write_bytes(self.data, v, end);
            self.len = end;
        }
        self
    }

    /// Ensure that `count` bytes from `start` contain zero bits
    ///
    /// This is used to initialize the bits in a buffer, however, it has no impact on the
    /// `len` of the buffer and so can be used to initialize the memory region from
    /// `len` to `capacity`.
    pub fn set_null_bits(&mut self, start: usize, count: usize) {
        assert!(start + count <= self.capacity);
        unsafe {
            std::ptr::write_bytes(self.data.add(start), 0, count);
        }
    }

    /// Ensures that this buffer has at least `capacity` slots in this buffer. This will
    /// also ensure the new capacity will be a multiple of 64 bytes.
    ///
    /// Returns the new capacity for this buffer.
    pub fn reserve(&mut self, capacity: usize) -> Result<usize> {
        if capacity > self.capacity {
            let new_capacity = bit_util::round_upto_multiple_of_64(capacity);
            let new_capacity = cmp::max(new_capacity, self.capacity * 2);
            let new_data =
                unsafe { memory::reallocate(self.data, self.capacity, new_capacity) };
            self.data = new_data as *mut u8;
            self.capacity = new_capacity;
        }
        Ok(self.capacity)
    }

    /// Resizes the buffer so that the `len` will equal to the `new_len`.
    ///
    /// If `new_len` is greater than `len`, the buffer's length is simply adjusted to be
    /// the former, optionally extending the capacity. The data between `len` and
    /// `new_len` will be zeroed out.
    ///
    /// If `new_len` is less than `len`, the buffer will be truncated.
    pub fn resize(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.len {
            self.reserve(new_len)?;
        } else {
            let new_capacity = bit_util::round_upto_multiple_of_64(new_len);
            if new_capacity < self.capacity {
                let new_data =
                    unsafe { memory::reallocate(self.data, self.capacity, new_capacity) };
                self.data = new_data as *mut u8;
                self.capacity = new_capacity;
            }
        }
        self.len = new_len;
        Ok(())
    }

    /// Returns whether this buffer is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the length (the number of bytes written) in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total capacity in this buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all existing data from this buffer.
    pub fn clear(&mut self) {
        self.len = 0
    }

    /// Returns the data stored in this buffer as a slice.
    pub fn data(&self) -> &[u8] {
        if self.data.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw_data(), self.len()) }
        }
    }

    /// Returns the data stored in this buffer as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        if self.data.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.raw_data_mut(), self.len()) }
        }
    }

    /// Returns a raw pointer for this buffer.
    ///
    /// Note that this should be used cautiously, and the returned pointer should not be
    /// stored anywhere, to avoid dangling pointers.
    pub fn raw_data(&self) -> *const u8 {
        self.data
    }

    pub fn raw_data_mut(&mut self) -> *mut u8 {
        self.data
    }

    /// Freezes this buffer and return an immutable version of it.
    pub fn freeze(self) -> Buffer {
        let buffer_data = BufferData {
            ptr: self.data,
            len: self.len,
            capacity: self.capacity,
            owned: true,
        };
        std::mem::forget(self);
        Buffer {
            data: Arc::new(buffer_data),
            offset: 0,
        }
    }

    /// View buffer as typed slice.
    pub fn typed_data_mut<T: ArrowNativeType + num::Num>(&mut self) -> &mut [T] {
        assert_eq!(self.len() % mem::size_of::<T>(), 0);
        assert!(memory::is_ptr_aligned::<T>(self.raw_data() as *const T));
        unsafe {
            from_raw_parts_mut(
                self.raw_data() as *mut T,
                self.len() / mem::size_of::<T>(),
            )
        }
    }

    /// Writes a byte slice to the underlying buffer and updates the `len`, i.e. the
    /// number array elements in the buffer.  Also, converts the `io::Result`
    /// required by the `Write` trait to the Arrow `Result` type.
    pub fn write_bytes(&mut self, bytes: &[u8], len_added: usize) -> Result<()> {
        let write_result = self.write(bytes);
        // `io::Result` has many options one of which we use, so pattern matching is
        // overkill here
        if write_result.is_err() {
            Err(ArrowError::IoError(
                "Could not write to Buffer, not big enough".to_string(),
            ))
        } else {
            self.len += len_added;
            Ok(())
        }
    }
}

impl Drop for MutableBuffer {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe { memory::free_aligned(self.data, self.capacity) };
        }
    }
}

impl PartialEq for MutableBuffer {
    fn eq(&self, other: &MutableBuffer) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.capacity != other.capacity {
            return false;
        }
        unsafe { memory::memcmp(self.data, other.data, self.len) == 0 }
    }
}

impl Write for MutableBuffer {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let remaining_capacity = self.capacity - self.len;
        if buf.len() > remaining_capacity {
            return Err(IoError::new(ErrorKind::Other, "Buffer not big enough"));
        }
        unsafe {
            memory::memcpy(self.data.add(self.len), buf.as_ptr(), buf.len());
            self.len += buf.len();
            Ok(buf.len())
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

unsafe impl Sync for MutableBuffer {}
unsafe impl Send for MutableBuffer {}

#[cfg(test)]
mod tests {
    use crate::util::bit_util;
    use std::ptr::null_mut;
    use std::thread;

    use super::*;
    use crate::datatypes::ToByteSlice;

    #[test]
    fn test_buffer_data_equality() {
        let buf1 = Buffer::from(&[0, 1, 2, 3, 4]);
        let mut buf2 = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(buf1, buf2);

        // slice with same offset should still preserve equality
        let buf3 = buf1.slice(2);
        assert_ne!(buf1, buf3);
        let buf4 = buf2.slice(2);
        assert_eq!(buf3, buf4);

        // unequal because of different elements
        buf2 = Buffer::from(&[0, 0, 2, 3, 4]);
        assert_ne!(buf1, buf2);

        // unequal because of different length
        buf2 = Buffer::from(&[0, 1, 2, 3]);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_from_raw_parts() {
        let buf = unsafe { Buffer::from_raw_parts(null_mut(), 0, 0) };
        assert_eq!(0, buf.len());
        assert_eq!(0, buf.data().len());
        assert_eq!(0, buf.capacity());
        assert!(buf.raw_data().is_null());

        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(5, buf.len());
        assert!(!buf.raw_data().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.data());
    }

    #[test]
    fn test_from_vec() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(5, buf.len());
        assert!(!buf.raw_data().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.data());
    }

    #[test]
    fn test_copy() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        let buf2 = buf.clone();
        assert_eq!(5, buf2.len());
        assert_eq!(64, buf2.capacity());
        assert!(!buf2.raw_data().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf2.data());
    }

    #[test]
    fn test_slice() {
        let buf = Buffer::from(&[2, 4, 6, 8, 10]);
        let buf2 = buf.slice(2);

        assert_eq!([6, 8, 10], buf2.data());
        assert_eq!(3, buf2.len());
        assert_eq!(unsafe { buf.raw_data().offset(2) }, buf2.raw_data());

        let buf3 = buf2.slice(1);
        assert_eq!([8, 10], buf3.data());
        assert_eq!(2, buf3.len());
        assert_eq!(unsafe { buf.raw_data().offset(3) }, buf3.raw_data());

        let buf4 = buf.slice(5);
        let empty_slice: [u8; 0] = [];
        assert_eq!(empty_slice, buf4.data());
        assert_eq!(0, buf4.len());
        assert!(buf4.is_empty());
        assert_eq!(buf2.slice(2).data(), &[10]);
    }

    #[test]
    #[should_panic(
        expected = "the offset of the new Buffer cannot exceed the existing length"
    )]
    fn test_slice_offset_out_of_bound() {
        let buf = Buffer::from(&[2, 4, 6, 8, 10]);
        buf.slice(6);
    }

    #[test]
    fn test_with_bitset() {
        let mut_buf = MutableBuffer::new(64).with_bitset(64, false);
        let buf = mut_buf.freeze();
        assert_eq!(0, bit_util::count_set_bits(buf.data()));

        let mut_buf = MutableBuffer::new(64).with_bitset(64, true);
        let buf = mut_buf.freeze();
        assert_eq!(512, bit_util::count_set_bits(buf.data()));
    }

    #[test]
    fn test_set_null_bits() {
        let mut mut_buf = MutableBuffer::new(64).with_bitset(64, true);
        mut_buf.set_null_bits(0, 64);
        let buf = mut_buf.freeze();
        assert_eq!(0, bit_util::count_set_bits(buf.data()));

        let mut mut_buf = MutableBuffer::new(64).with_bitset(64, true);
        mut_buf.set_null_bits(32, 32);
        let buf = mut_buf.freeze();
        assert_eq!(256, bit_util::count_set_bits(buf.data()));
    }

    #[test]
    fn test_bitwise_and() {
        let buf1 = Buffer::from([0b01101010]);
        let buf2 = Buffer::from([0b01001110]);
        assert_eq!(Buffer::from([0b01001010]), (&buf1 & &buf2).unwrap());
    }

    #[test]
    fn test_bitwise_or() {
        let buf1 = Buffer::from([0b01101010]);
        let buf2 = Buffer::from([0b01001110]);
        assert_eq!(Buffer::from([0b01101110]), (&buf1 | &buf2).unwrap());
    }

    #[test]
    fn test_bitwise_not() {
        let buf = Buffer::from([0b01101010]);
        assert_eq!(Buffer::from([0b10010101]), !&buf);
    }

    #[test]
    #[should_panic(expected = "Buffers must be the same size to apply Bitwise OR.")]
    fn test_buffer_bitand_different_sizes() {
        let buf1 = Buffer::from([1_u8, 1_u8]);
        let buf2 = Buffer::from([0b01001110]);
        let _buf3 = (&buf1 | &buf2).unwrap();
    }

    #[test]
    fn test_mutable_new() {
        let buf = MutableBuffer::new(63);
        assert_eq!(64, buf.capacity());
        assert_eq!(0, buf.len());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_mutable_write() {
        let mut buf = MutableBuffer::new(100);
        buf.write("hello".as_bytes()).expect("Ok");
        assert_eq!(5, buf.len());
        assert_eq!("hello".as_bytes(), buf.data());

        buf.write(" world".as_bytes()).expect("Ok");
        assert_eq!(11, buf.len());
        assert_eq!("hello world".as_bytes(), buf.data());

        buf.clear();
        assert_eq!(0, buf.len());
        buf.write("hello some".as_bytes()).expect("Ok");
        assert_eq!(11, buf.len());
        assert_eq!("hello some".as_bytes(), buf.data());
    }

    #[test]
    #[should_panic(expected = "Buffer not big enough")]
    fn test_mutable_write_overflow() {
        let mut buf = MutableBuffer::new(1);
        assert_eq!(64, buf.capacity());
        for _ in 0..10 {
            buf.write(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        }
    }

    #[test]
    fn test_mutable_reserve() {
        let mut buf = MutableBuffer::new(1);
        assert_eq!(64, buf.capacity());

        // Reserving a smaller capacity should have no effect.
        let mut new_cap = buf.reserve(10).expect("reserve should be OK");
        assert_eq!(64, new_cap);
        assert_eq!(64, buf.capacity());

        new_cap = buf.reserve(100).expect("reserve should be OK");
        assert_eq!(128, new_cap);
        assert_eq!(128, buf.capacity());
    }

    #[test]
    fn test_mutable_resize() {
        let mut buf = MutableBuffer::new(1);
        assert_eq!(64, buf.capacity());
        assert_eq!(0, buf.len());

        buf.resize(20).expect("resize should be OK");
        assert_eq!(64, buf.capacity());
        assert_eq!(20, buf.len());

        buf.resize(10).expect("resize should be OK");
        assert_eq!(64, buf.capacity());
        assert_eq!(10, buf.len());

        buf.resize(100).expect("resize should be OK");
        assert_eq!(128, buf.capacity());
        assert_eq!(100, buf.len());

        buf.resize(30).expect("resize should be OK");
        assert_eq!(64, buf.capacity());
        assert_eq!(30, buf.len());

        buf.resize(0).expect("resize should be OK");
        assert_eq!(0, buf.capacity());
        assert_eq!(0, buf.len());
    }

    #[test]
    fn test_mutable_freeze() {
        let mut buf = MutableBuffer::new(1);
        buf.write("aaaa bbbb cccc dddd".as_bytes())
            .expect("write should be OK");
        assert_eq!(19, buf.len());
        assert_eq!(64, buf.capacity());
        assert_eq!("aaaa bbbb cccc dddd".as_bytes(), buf.data());

        let immutable_buf = buf.freeze();
        assert_eq!(19, immutable_buf.len());
        assert_eq!(64, immutable_buf.capacity());
        assert_eq!("aaaa bbbb cccc dddd".as_bytes(), immutable_buf.data());
    }

    #[test]
    fn test_mutable_equal() -> Result<()> {
        let mut buf = MutableBuffer::new(1);
        let mut buf2 = MutableBuffer::new(1);

        buf.write(&[0xaa])?;
        buf2.write(&[0xaa, 0xbb])?;
        assert!(buf != buf2);

        buf.write(&[0xbb])?;
        assert_eq!(buf, buf2);

        buf2.reserve(65)?;
        assert!(buf != buf2);

        Ok(())
    }

    #[test]
    fn test_access_concurrently() {
        let buffer = Buffer::from(vec![1, 2, 3, 4, 5]);
        let buffer2 = buffer.clone();
        assert_eq!([1, 2, 3, 4, 5], buffer.data());

        let buffer_copy = thread::spawn(move || {
            // access buffer in another thread.
            buffer.clone()
        })
        .join();

        assert!(buffer_copy.is_ok());
        assert_eq!(buffer2, buffer_copy.ok().unwrap());
    }

    macro_rules! check_as_typed_data {
        ($input: expr, $native_t: ty) => {{
            let buffer = Buffer::from($input.to_byte_slice());
            let slice: &[$native_t] = unsafe { buffer.typed_data::<$native_t>() };
            assert_eq!($input, slice);
        }};
    }

    #[test]
    fn test_as_typed_data() {
        check_as_typed_data!(&[1i8, 3i8, 6i8], i8);
        check_as_typed_data!(&[1u8, 3u8, 6u8], u8);
        check_as_typed_data!(&[1i16, 3i16, 6i16], i16);
        check_as_typed_data!(&[1i32, 3i32, 6i32], i32);
        check_as_typed_data!(&[1i64, 3i64, 6i64], i64);
        check_as_typed_data!(&[1u16, 3u16, 6u16], u16);
        check_as_typed_data!(&[1u32, 3u32, 6u32], u32);
        check_as_typed_data!(&[1u64, 3u64, 6u64], u64);
        check_as_typed_data!(&[1f32, 3f32, 6f32], f32);
        check_as_typed_data!(&[1f64, 3f64, 6f64], f64);
    }
}
