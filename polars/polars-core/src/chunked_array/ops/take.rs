//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use crate::chunked_array::builder::{
    get_list_builder, PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
};
use crate::chunked_array::kernels::take::{
    take_no_null_primitive, take_no_null_primitive_iter, take_utf8,
};
use crate::prelude::*;
use crate::utils::NoNull;
use arrow::array::{
    Array, ArrayRef, BooleanArray, LargeListArray, LargeStringArray, PrimitiveArray,
};
use arrow::compute::kernels::take::take;
use polars_arrow::prelude::*;
use std::convert::TryFrom;
use std::ops::Deref;
use unsafe_unwrap::UnsafeUnwrap;

macro_rules! impl_take {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);

        let taker = $self.take_rand();
        for idx in $indices {
            match taker.get(idx) {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }};
}

macro_rules! impl_take_opt {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);
        let taker = $self.take_rand();

        for opt_idx in $indices {
            match opt_idx {
                Some(idx) => match taker.get(idx) {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            };
        }
        builder.finish()
    }};
}

macro_rules! impl_take_opt_unchecked {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);
        let taker = $self.take_rand();

        for opt_idx in $indices {
            match opt_idx {
                Some(idx) => {
                    let v = taker.get_unchecked(idx);
                    builder.append_value(v);
                }
                None => builder.append_null(),
            };
        }
        builder.finish()
    }};
}

macro_rules! impl_take_unchecked {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);

        let taker = $self.take_rand();
        for idx in $indices {
            let v = taker.get_unchecked(idx);
            builder.append_value(v);
        }
        builder.finish()
    }};
}

impl<T> ChunkTake for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        impl_take!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            if self.null_count() == 0 {
                let arr = self.downcast_chunks()[0];
                let arr = take_no_null_primitive_iter(arr, indices);
                return Self::new_from_chunks(self.name(), vec![arr]);
            }
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        impl_take_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        impl_take_opt!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        impl_take_opt_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_from_single_chunked(&self, idx: &UInt32Chunked) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if self.chunks.len() == 1 && idx.chunks.len() == 1 {
            let idx_arr = idx.downcast_chunks()[0];

            let new_arr = if self.null_count() == 0 {
                let arr = self.downcast_chunks()[0];
                unsafe { take_no_null_primitive(arr, idx_arr) as ArrayRef }
            } else {
                let arr = &self.chunks[0];
                take(&**arr, idx_arr, None).unwrap()
            };
            Ok(Self::new_from_chunks(self.name(), vec![new_arr]))
        } else {
            Err(PolarsError::NoSlice)
        }
    }
}

impl ChunkTake for BooleanChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        impl_take!(self, indices, capacity, BooleanChunkedBuilder)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        impl_take_unchecked!(self, indices, capacity, BooleanChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        impl_take_opt!(self, indices, capacity, BooleanChunkedBuilder)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        impl_take_opt_unchecked!(self, indices, capacity, BooleanChunkedBuilder)
    }

    fn take_from_single_chunked(&self, idx: &UInt32Chunked) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if self.chunks.len() == 1 && idx.chunks.len() == 1 {
            let idx_arr = idx.downcast_chunks()[0];
            let arr = &self.chunks[0];

            let new_arr = take(&**arr, idx_arr, None).unwrap();
            Ok(Self::new_from_chunks(self.name(), vec![new_arr]))
        } else {
            Err(PolarsError::NoSlice)
        }
    }
}

impl ChunkTake for CategoricalChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        let ca: CategoricalChunked = self.deref().take(indices, capacity).into();
        ca.set_state(self)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        let ca: CategoricalChunked = self.deref().take_unchecked(indices, capacity).into();
        ca.set_state(self)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        let ca: CategoricalChunked = self.deref().take_opt(indices, capacity).into();
        ca.set_state(self)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        let ca: CategoricalChunked = self.deref().take_opt_unchecked(indices, capacity).into();
        ca.set_state(self)
    }

    fn take_from_single_chunked(&self, idx: &UInt32Chunked) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let ca: CategoricalChunked = self.deref().take_from_single_chunked(idx)?.into();
        Ok(ca.set_state(self))
    }
}

impl ChunkTake for Utf8Chunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }

        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let fact = capacity as f32 / self.len() as f32 * 1.2;
        let values_cap = (capacity as f32 * fact) as usize;
        let mut builder = Utf8ChunkedBuilder::new(self.name(), capacity, values_cap);

        let taker = self.take_rand();
        for idx in indices {
            match taker.get(idx) {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }
    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let fact = capacity as f32 / self.len() as f32 * 1.2;
        let values_cap = (capacity as f32 * fact) as usize;

        let mut builder = Utf8ChunkedBuilder::new(self.name(), capacity, values_cap);

        let taker = self.take_rand();
        for idx in indices {
            let v = taker.get_unchecked(idx);
            builder.append_value(v);
        }
        builder.finish()
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        if self.is_empty() {
            return self.clone();
        }

        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let fact = capacity as f32 / self.len() as f32 * 1.2;
        let values_cap = (capacity as f32 * fact) as usize;

        let mut builder = Utf8ChunkedBuilder::new(self.name(), capacity, values_cap);
        let taker = self.take_rand();

        for opt_idx in indices {
            match opt_idx {
                Some(idx) => match taker.get(idx) {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            };
        }
        builder.finish()
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }

        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let fact = capacity as f32 / self.len() as f32 * 1.2;
        let values_cap = (capacity as f32 * fact) as usize;

        let mut builder = Utf8ChunkedBuilder::new(self.name(), capacity, values_cap);
        let taker = self.take_rand();

        for opt_idx in indices {
            match opt_idx {
                Some(idx) => {
                    let v = taker.get_unchecked(idx);
                    builder.append_value(v);
                }
                None => builder.append_null(),
            };
        }
        builder.finish()
    }

    fn take_from_single_chunked(&self, idx: &UInt32Chunked) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if self.chunks.len() == 1 && idx.chunks.len() == 1 {
            let idx_arr = idx.downcast_chunks()[0];
            let arr = self.downcast_chunks()[0];
            // TODO: mark this function as unsafe
            let new_arr = unsafe { take_utf8(arr, idx_arr) as ArrayRef };
            Ok(Self::new_from_chunks(self.name(), vec![new_arr]))
        } else {
            Err(PolarsError::NoSlice)
        }
    }
}

impl ChunkTake for ListChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let value_cap =
            (self.get_values_size() as f32 / self.len() as f32 * capacity as f32 * 1.4) as usize;

        match self.dtype() {
            DataType::List(dt) => {
                let mut builder = get_list_builder(&dt.into(), value_cap, capacity, self.name());
                let taker = self.take_rand();

                for idx in indices {
                    builder.append_opt_series(taker.get(idx).as_ref());
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }
    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.chunks.len() == 1 {
            return self.take_from_single_chunked_iter(indices).unwrap();
        }
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let value_cap =
            (self.get_values_size() as f32 / self.len() as f32 * capacity as f32 * 1.4) as usize;
        match self.dtype() {
            DataType::List(dt) => {
                let mut builder = get_list_builder(&dt.into(), value_cap, capacity, self.name());
                let taker = self.take_rand();
                for idx in indices {
                    let v = taker.get_unchecked(idx);
                    builder.append_opt_series(Some(&v));
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let value_cap =
            (self.get_values_size() as f32 / self.len() as f32 * capacity as f32 * 1.4) as usize;

        match self.dtype() {
            DataType::List(dt) => {
                let mut builder = get_list_builder(&dt.into(), value_cap, capacity, self.name());

                let taker = self.take_rand();

                for opt_idx in indices {
                    match opt_idx {
                        Some(idx) => {
                            let opt_s = taker.get(idx);
                            builder.append_opt_series(opt_s.as_ref())
                        }
                        None => builder.append_opt_series(None),
                    };
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        let value_cap =
            (self.get_values_size() as f32 / self.len() as f32 * capacity as f32 * 1.4) as usize;

        match self.dtype() {
            DataType::List(dt) => {
                let mut builder = get_list_builder(&dt.into(), value_cap, capacity, self.name());
                let taker = self.take_rand();

                for opt_idx in indices {
                    match opt_idx {
                        Some(idx) => {
                            let s = taker.get_unchecked(idx);
                            builder.append_opt_series(Some(&s))
                        }
                        None => builder.append_opt_series(None),
                    };
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    fn take_from_single_chunked(&self, idx: &UInt32Chunked) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if self.chunks.len() == 1 && idx.chunks.len() == 1 {
            let idx_arr = idx.downcast_chunks()[0];
            let arr = &self.chunks[0];

            let new_arr = take(&**arr, idx_arr, None).unwrap();
            Ok(Self::new_from_chunks(self.name(), vec![new_arr]))
        } else {
            Err(PolarsError::NoSlice)
        }
    }
}

#[cfg(feature = "object")]
impl<T> ChunkTake for ObjectChunked<T> {
    fn take_from_single_chunked(&self, _idx: &UInt32Chunked) -> Result<Self> {
        todo!()
    }

    fn take(&self, _indices: impl Iterator<Item = usize>, _capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        todo!()
    }

    unsafe fn take_unchecked(
        &self,
        _indices: impl Iterator<Item = usize>,
        _capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        todo!()
    }

    fn take_opt(
        &self,
        _indices: impl Iterator<Item = Option<usize>>,
        _capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        todo!()
    }

    unsafe fn take_opt_unchecked(
        &self,
        _indices: impl Iterator<Item = Option<usize>>,
        _capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        todo!()
    }
}

pub trait AsTakeIndex {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a>;

    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        unimplemented!()
    }

    fn take_index_len(&self) -> usize;
}

impl AsTakeIndex for &UInt32Chunked {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        match self.null_count() {
            0 => Box::new(self.into_no_null_iter().map(|val| val as usize)),
            _ => Box::new(
                self.into_iter()
                    .filter_map(|opt_val| opt_val.map(|val| val as usize)),
            ),
        }
    }
    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val as usize)),
        )
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl<T> AsTakeIndex for T
where
    T: AsRef<[usize]>,
{
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.as_ref().iter().copied())
    }
    fn take_index_len(&self) -> usize {
        self.as_ref().len()
    }
}

impl AsTakeIndex for [u32] {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.iter().map(|&v| v as usize))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

/// Create a type that implements a faster `TakeRandom`.
pub trait IntoTakeRandom<'a> {
    type Item;
    type TakeRandom;
    /// Create a type that implements `TakeRandom`.
    fn take_rand(&self) -> Self::TakeRandom;
}

/// Choose the Struct for multiple chunks or the struct for a single chunk.
macro_rules! many_or_single {
    ($self:ident, $StructSingle:ident, $StructMany:ident) => {{
        let chunks = $self.downcast_chunks();
        if chunks.len() == 1 {
            Box::new($StructSingle { arr: chunks[0] })
        } else {
            Box::new($StructMany { ca: $self, chunks })
        }
    }};
}

impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        match self.cont_slice() {
            Ok(slice) => Box::new(NumTakeRandomCont { slice }),
            _ => {
                let chunks = self.downcast_chunks();
                if chunks.len() == 1 {
                    Box::new(NumTakeRandomSingleChunk { arr: chunks[0] })
                } else {
                    Box::new(NumTakeRandomChunked { ca: self, chunks })
                }
            }
        }
    }
}

impl<'a> IntoTakeRandom<'a> for &'a Utf8Chunked {
    type Item = &'a str;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        many_or_single!(self, Utf8TakeRandomSingleChunk, Utf8TakeRandom)
    }
}

impl<'a> IntoTakeRandom<'a> for &'a BooleanChunked {
    type Item = bool;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        many_or_single!(self, BoolTakeRandomSingleChunk, BoolTakeRandom)
    }
}

impl<'a> IntoTakeRandom<'a> for &'a ListChunked {
    type Item = Series;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        let chunks = self.downcast_chunks();
        if chunks.len() == 1 {
            Box::new(ListTakeRandomSingleChunk {
                arr: chunks[0],
                name: self.name(),
            })
        } else {
            Box::new(ListTakeRandom { ca: self, chunks })
        }
    }
}

pub struct NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
}

macro_rules! take_random_get {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) = $self.ca.index_to_chunked_index($index);
        let arr = $self.chunks.get(chunk_idx);
        match arr {
            Some(arr) => {
                if arr.is_null(arr_idx) {
                    None
                } else {
                    // SAFETY:
                    // bounds checked above
                    unsafe { Some(arr.value_unchecked(arr_idx)) }
                }
            }
            None => None,
        }
    }};
}

macro_rules! take_random_get_unchecked {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) = $self.ca.index_to_chunked_index($index);
        $self
            .chunks
            .get_unchecked(chunk_idx)
            .value_unchecked(arr_idx)
    }};
}

macro_rules! take_random_get_single {
    ($self:ident, $index:ident) => {{
        if $self.arr.is_null($index) {
            None
        } else {
            // Safety:
            // bound checked above
            unsafe { Some($self.arr.value_unchecked($index)) }
        }
    }};
}

impl<'a, T> TakeRandom for NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct NumTakeRandomCont<'a, T> {
    slice: &'a [T],
}

impl<'a, T> TakeRandom for NumTakeRandomCont<'a, T>
where
    T: Copy,
{
    type Item = T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        self.slice.get(index).copied()
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        *self.slice.get_unchecked(index)
    }
}

pub struct NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    arr: &'a PrimitiveArray<T>,
}

impl<'a, T> TakeRandom for NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value_unchecked(index)
    }
}

pub struct Utf8TakeRandom<'a> {
    ca: &'a Utf8Chunked,
    chunks: Vec<&'a LargeStringArray>,
}

impl<'a> TakeRandom for Utf8TakeRandom<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let (chunk_idx, arr_idx) = self.ca.index_to_chunked_index(index);
        self.chunks
            .get_unchecked(chunk_idx)
            .value_unchecked(arr_idx)
    }
}

pub struct Utf8TakeRandomSingleChunk<'a> {
    arr: &'a LargeStringArray,
}

impl<'a> TakeRandom for Utf8TakeRandomSingleChunk<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value_unchecked(index)
    }
}

pub struct BoolTakeRandom<'a> {
    ca: &'a BooleanChunked,
    chunks: Vec<&'a BooleanArray>,
}

impl<'a> TakeRandom for BoolTakeRandom<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct BoolTakeRandomSingleChunk<'a> {
    arr: &'a BooleanArray,
}

impl<'a> TakeRandom for BoolTakeRandomSingleChunk<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}
pub struct ListTakeRandom<'a> {
    ca: &'a ListChunked,
    chunks: Vec<&'a LargeListArray>,
}

impl<'a> TakeRandom for ListTakeRandom<'a> {
    type Item = Series;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get!(self, index);
        v.map(|v| {
            let s = Series::try_from((self.ca.name(), v));
            s.unwrap()
        })
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let v = take_random_get_unchecked!(self, index);
        let s = Series::try_from((self.ca.name(), v));
        s.unwrap()
    }
}

pub struct ListTakeRandomSingleChunk<'a> {
    arr: &'a LargeListArray,
    name: &'a str,
}

impl<'a> TakeRandom for ListTakeRandomSingleChunk<'a> {
    type Item = Series;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get_single!(self, index);
        v.map(|v| {
            let s = Series::try_from((self.name, v));
            s.unwrap()
        })
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let s = Series::try_from((self.name, self.arr.value_unchecked(index)));
        s.unsafe_unwrap()
    }
}

impl<T> ChunkTakeEvery<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take_every(&self, n: usize) -> ChunkedArray<T> {
        if self.null_count() == 0 {
            let a: NoNull<_> = self.into_no_null_iter().step_by(n).collect();
            a.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<BooleanType> for BooleanChunked {
    fn take_every(&self, n: usize) -> BooleanChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<Utf8Type> for Utf8Chunked {
    fn take_every(&self, n: usize) -> Utf8Chunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<ListType> for ListChunked {
    fn take_every(&self, n: usize) -> ListChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<CategoricalType> for CategoricalChunked {
    fn take_every(&self, n: usize) -> CategoricalChunked {
        let mut ca = if self.null_count() == 0 {
            let ca: NoNull<UInt32Chunked> = self.into_no_null_iter().step_by(n).collect();
            ca.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.categorical_map = self.categorical_map.clone();
        ca.cast().unwrap()
    }
}
#[cfg(feature = "object")]
impl<T> ChunkTakeEvery<ObjectType<T>> for ObjectChunked<T> {
    fn take_every(&self, _n: usize) -> ObjectChunked<T> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_take_random() {
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        assert_eq!(ca.get(0), Some(1));
        assert_eq!(ca.get(1), Some(2));
        assert_eq!(ca.get(2), Some(3));

        let ca = Utf8Chunked::new_from_slice("a", &["a", "b", "c"]);
        assert_eq!(ca.get(0), Some("a"));
        assert_eq!(ca.get(1), Some("b"));
        assert_eq!(ca.get(2), Some("c"));
    }
}
