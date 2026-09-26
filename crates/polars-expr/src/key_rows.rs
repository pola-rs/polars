#![allow(unsafe_op_in_unsafe_fn)]
//! Multi-column keys stored as rows of `u64` words, with a stride fixed by the key
//! schema. Fixed-width values are stored by value and strings as their views; only
//! strings too long to inline compare their bytes.
use std::cmp::Reverse;
use std::hash::BuildHasher;
use std::sync::Arc;

use hashbrown::hash_table::{Entry as TEntry, HashTable};
use polars_arrow::array::{
    Array, BinaryViewArrayGeneric, BooleanArray, PrimitiveArray, UInt64Array, View,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and_many;
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_compute::gather::bitmap::take_bitmap_unchecked;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::IdxSize;
use polars_utils::hashing::folded_multiply;
use polars_utils::total_ord::{canonical_f32, canonical_f64};

use crate::hash_keys::{for_each_hash_prehashed, for_each_hash_subset_prehashed};

const BASE_KEY_BUFFER_CAPACITY: usize = 1024;
const MAX_KEY_BUFFER_CAPACITY: usize = 1 << 30;
const HASH_MULTIPLE: u64 = 0x5851f42d4c957f2d;
const NULL_WORD: u64 = 0x9e3779b97f4a7c15;
const MAX_KEY_COLUMNS: usize = 64;
const MAX_VIEW_COLUMNS: usize = 2;
pub(crate) const VERIFY_BATCH_SIZE: usize = 256;

#[inline(always)]
fn fold(h: u64, w: u64) -> u64 {
    folded_multiply(h ^ w, HASH_MULTIPLE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColKind {
    Bool,
    Fixed,
    View,
}

#[derive(Clone, Debug)]
struct ColLayout {
    kind: ColKind,
    physical: DataType,
    /// Byte offset within the row.
    offset: usize,
}

/// The layout of a key column of this dtype, with its width in the row.
fn key_col(dtype: &DataType) -> Option<(ColLayout, usize)> {
    let physical = dtype.to_physical();
    let (kind, width) = match &physical {
        DataType::Boolean => (ColKind::Bool, 1),
        DataType::String | DataType::Binary => (ColKind::View, 16),
        DataType::Int8 | DataType::UInt8 => (ColKind::Fixed, 1),
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => (ColKind::Fixed, 2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => (ColKind::Fixed, 4),
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => (ColKind::Fixed, 8),
        DataType::Int128 | DataType::UInt128 => (ColKind::Fixed, 16),
        _ => return None,
    };
    Some((
        ColLayout {
            kind,
            physical,
            offset: 0,
        },
        width,
    ))
}

#[derive(Debug)]
pub struct KeyRowLayout {
    cols: Vec<ColLayout>,
    /// Byte offset of the null bits, one per column.
    null_offset: usize,
    null_width: usize,
    stride: usize,
    /// The first word of each view column, in column order.
    view_words: Vec<usize>,
    /// Every word except the second half of each view.
    plain_words: Vec<usize>,
}

impl KeyRowLayout {
    /// The layout of keys of these dtypes, or `None` when they are not stored as key
    /// rows.
    pub fn new<'a>(dtypes: impl IntoIterator<Item = &'a DataType>) -> Option<Self> {
        let (mut cols, widths): (Vec<ColLayout>, Vec<usize>) = dtypes
            .into_iter()
            .map(key_col)
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .unzip();
        let num_views = cols.iter().filter(|c| c.kind == ColKind::View).count();
        if !(2..=MAX_KEY_COLUMNS).contains(&cols.len()) || num_views > MAX_VIEW_COLUMNS {
            return None;
        }

        let null_width = cols.len().div_ceil(8).next_power_of_two();
        let mut fields: Vec<(usize, Option<usize>)> = widths
            .into_iter()
            .enumerate()
            .map(|(i, width)| (width, Some(i)))
            .chain([(null_width, None)])
            .collect();
        fields.sort_by_key(|(width, _)| Reverse(*width));
        let mut offset = 0;
        let mut null_offset = 0;
        for (width, col) in fields {
            match col {
                Some(i) => cols[i].offset = offset,
                None => null_offset = offset,
            }
            offset += width;
        }

        let stride = offset.div_ceil(8);
        let view_words: Vec<usize> = cols
            .iter()
            .filter(|c| c.kind == ColKind::View)
            .map(|c| c.offset / 8)
            .collect();
        let plain_words = (0..stride)
            .filter(|w| !view_words.iter().any(|v| v + 1 == *w))
            .collect();
        Some(Self {
            cols,
            null_offset,
            null_width,
            stride,
            view_words,
            plain_words,
        })
    }

    fn has_views(&self) -> bool {
        !self.view_words.is_empty()
    }

    /// # Safety
    /// `row` must point to a row of this layout.
    #[inline(always)]
    unsafe fn read_nulls(&self, row: *const u8) -> u64 {
        let p = row.add(self.null_offset);
        match self.null_width {
            1 => *p as u64,
            2 => p.cast::<u16>().read_unaligned() as u64,
            4 => p.cast::<u32>().read_unaligned() as u64,
            _ => p.cast::<u64>().read_unaligned(),
        }
    }

    /// # Safety
    /// `row` must point to a row of this layout.
    #[inline(always)]
    unsafe fn write_nulls(&self, row: *mut u8, nulls: u64) {
        let p = row.add(self.null_offset);
        match self.null_width {
            1 => *p = nulls as u8,
            2 => p.cast::<u16>().write_unaligned(nulls as u16),
            4 => p.cast::<u32>().write_unaligned(nulls as u32),
            _ => p.cast::<u64>().write_unaligned(nulls),
        }
    }

    /// Compares two rows of this layout. `long_eq` compares the bytes of two views
    /// that are too long to be inlined.
    ///
    /// # Safety
    /// Both rows must have `stride` words.
    #[inline(always)]
    unsafe fn rows_eq(
        &self,
        a: &[u64],
        b: &[u64],
        mut long_eq: impl FnMut(View, View) -> bool,
    ) -> bool {
        if self.view_words.is_empty() {
            return a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0;
        }
        for w in &self.plain_words {
            if a.get_unchecked(*w) != b.get_unchecked(*w) {
                return false;
            }
        }
        for w in &self.view_words {
            let view_a = view_at(a, *w);
            if view_a.length <= View::MAX_INLINE_SIZE {
                if a.get_unchecked(*w + 1) != b.get_unchecked(*w + 1) {
                    return false;
                }
            } else if !long_eq(view_a, view_at(b, *w)) {
                return false;
            }
        }
        true
    }

    /// Compares key `i` of `cols` to a stored row, whose long views are resolved by
    /// `stored_long`.
    ///
    /// # Safety
    /// `i` must be in-bounds and `row` must have `stride` words.
    #[inline(always)]
    unsafe fn eq_columns<'a>(
        &self,
        cols: &[KeyColumn],
        i: usize,
        row: &[u64],
        stored_long: impl Fn(View) -> &'a [u8],
    ) -> bool {
        let row = row.as_ptr() as *const u8;
        let mut nulls = 0;
        let mut eq = true;
        for (c, col) in cols.iter().enumerate() {
            let valid = col.validity.as_ref().is_none_or(|v| v.get_bit_unchecked(i));
            nulls |= (!valid as u64) << c;
            let p = row.add(col.offset);
            let col_eq = match &col.values {
                ColValues::Bool(b) => b.get_bit_unchecked(i) as u8 == *p,
                ColValues::W1(v) => *v.get_unchecked(i) == *p,
                ColValues::W2(v) => *v.get_unchecked(i) == p.cast::<u16>().read_unaligned(),
                ColValues::W4(v) => *v.get_unchecked(i) == p.cast::<u32>().read_unaligned(),
                ColValues::W8(v) => *v.get_unchecked(i) == p.cast::<u64>().read_unaligned(),
                ColValues::W16(v) => *v.get_unchecked(i) == p.cast::<u128>().read_unaligned(),
                ColValues::View(views, buffers) => {
                    !valid || {
                        let a = *views.get_unchecked(i);
                        let b = p.cast::<u128>().read_unaligned();
                        view_eq(a, b, buffers, &stored_long)
                    }
                },
            };
            eq &= col_eq | !valid;
        }
        eq && nulls == self.read_nulls(row)
    }

    /// Writes key `i` of `cols` into a zeroed row, storing the bytes of its long
    /// views with `store_long`, which returns the buffer index and offset of the copy.
    ///
    /// # Safety
    /// `i` must be in-bounds and `row` must have `stride` words.
    #[inline(always)]
    unsafe fn write_columns(
        &self,
        cols: &[KeyColumn],
        i: usize,
        row: &mut [u64],
        mut store_long: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        let row = row.as_mut_ptr() as *mut u8;
        let mut nulls = 0;
        for (c, col) in cols.iter().enumerate() {
            if !col.validity.as_ref().is_none_or(|v| v.get_bit_unchecked(i)) {
                nulls |= 1 << c;
                continue;
            }
            let p = row.add(col.offset);
            match &col.values {
                ColValues::Bool(b) => *p = b.get_bit_unchecked(i) as u8,
                ColValues::W1(v) => *p = *v.get_unchecked(i),
                ColValues::W2(v) => p.cast::<u16>().write_unaligned(*v.get_unchecked(i)),
                ColValues::W4(v) => p.cast::<u32>().write_unaligned(*v.get_unchecked(i)),
                ColValues::W8(v) => p.cast::<u64>().write_unaligned(*v.get_unchecked(i)),
                ColValues::W16(v) => p.cast::<u128>().write_unaligned(*v.get_unchecked(i)),
                ColValues::View(views, buffers) => {
                    let mut view = *views.get_unchecked(i);
                    if view.length > View::MAX_INLINE_SIZE {
                        let (buffer_idx, offset) =
                            store_long(view.get_external_slice_unchecked(buffers));
                        view.buffer_idx = buffer_idx;
                        view.offset = offset;
                    }
                    p.cast::<u128>().write_unaligned(view.as_u128());
                },
            }
        }
        self.write_nulls(row, nulls);
    }

    /// Clears `ok[r]` when key `idxs[r]` of `cols` differs from the stored row at
    /// `rows[r]`, whose long views are resolved by `stored_long`.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    unsafe fn verify_columns<'a>(
        &self,
        cols: &[KeyColumn],
        idxs: &[IdxSize],
        rows: &[*const u64],
        ok: &mut [bool],
        stored_long: impl Fn(View) -> &'a [u8],
    ) {
        if cols.iter().all(|c| c.validity.is_none()) {
            for (ok, row) in ok.iter_mut().zip(rows) {
                *ok &= self.read_nulls(row.cast()) == 0;
            }
        } else {
            for ((ok, i), row) in ok.iter_mut().zip(idxs).zip(rows) {
                let mut nulls = 0;
                for (c, col) in cols.iter().enumerate() {
                    if let Some(v) = &col.validity {
                        nulls |= (!v.get_bit_unchecked(*i as usize) as u64) << c;
                    }
                }
                *ok &= self.read_nulls(row.cast()) == nulls;
            }
        }

        for col in cols {
            let off = col.offset;
            let validity = col.validity.as_ref();
            match &col.values {
                ColValues::Bool(b) => verify_col(idxs, rows, ok, validity, |i, p| {
                    b.get_bit_unchecked(i) as u8 == *p.add(off)
                }),
                ColValues::W1(v) => verify_col(idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == *p.add(off)
                }),
                ColValues::W2(v) => verify_col(idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u16>().read_unaligned()
                }),
                ColValues::W4(v) => verify_col(idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u32>().read_unaligned()
                }),
                ColValues::W8(v) => verify_col(idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u64>().read_unaligned()
                }),
                ColValues::W16(v) => verify_col(idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u128>().read_unaligned()
                }),
                ColValues::View(views, buffers) => verify_col(idxs, rows, ok, validity, |i, p| {
                    let b = p.add(off).cast::<u128>().read_unaligned();
                    view_eq(*views.get_unchecked(i), b, buffers, &stored_long)
                }),
            }
        }
    }

    /// Writes the keys `idxs[r]` of `cols` into the zeroed rows at `rows[r]`, storing
    /// the bytes of long views with `store_long`, which returns the buffer index and
    /// offset of the copy.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    unsafe fn write_columns_batch(
        &self,
        cols: &[KeyColumn],
        idxs: &[IdxSize],
        rows: &[*mut u64],
        mut store_long: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        for col in cols {
            let off = col.offset;
            let validity = col.validity.as_ref();
            match &col.values {
                ColValues::Bool(b) => write_col(idxs, rows, validity, |i, p| {
                    *p.add(off) = b.get_bit_unchecked(i) as u8
                }),
                ColValues::W1(v) => write_col(idxs, rows, validity, |i, p| {
                    *p.add(off) = *v.get_unchecked(i)
                }),
                ColValues::W2(v) => write_col(idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u16>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W4(v) => write_col(idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u32>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W8(v) => write_col(idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u64>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W16(v) => write_col(idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u128>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::View(views, buffers) => write_col(idxs, rows, validity, |i, p| {
                    let mut view = *views.get_unchecked(i);
                    if view.length > View::MAX_INLINE_SIZE {
                        let (buffer_idx, offset) =
                            store_long(view.get_external_slice_unchecked(buffers));
                        view.buffer_idx = buffer_idx;
                        view.offset = offset;
                    }
                    p.add(off).cast::<u128>().write_unaligned(view.as_u128());
                }),
            }
        }

        if cols.iter().any(|c| c.validity.is_some()) {
            for (i, row) in idxs.iter().zip(rows) {
                let mut nulls = 0;
                for (c, col) in cols.iter().enumerate() {
                    if let Some(v) = &col.validity {
                        nulls |= (!v.get_bit_unchecked(*i as usize) as u64) << c;
                    }
                }
                self.write_nulls(row.cast(), nulls);
            }
        }
    }

    /// Replaces each long view in `row` by one pointing at a copy of its bytes made
    /// by `store`, which returns the buffer index and offset of the copy.
    ///
    /// # Safety
    /// The row must have `stride` words and `long_bytes` must return the bytes of
    /// its long views.
    #[inline(always)]
    unsafe fn repoint_long_views<'a>(
        &self,
        row: &mut [u64],
        long_bytes: impl Fn(View) -> &'a [u8],
        mut store: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        for w in &self.view_words {
            let view = view_at(row, *w);
            if view.length > View::MAX_INLINE_SIZE {
                let (buffer_idx, offset) = store(long_bytes(view));
                set_view(
                    row,
                    *w,
                    View {
                        buffer_idx,
                        offset,
                        ..view
                    },
                );
            }
        }
    }

    /// Reads the key columns back out of `num_rows` rows, each starting `skip`
    /// words into an entry of `entry_words` words. All views point into `buffers`.
    fn decode(
        &self,
        schema: &Schema,
        entries: &[u64],
        entry_words: usize,
        skip: usize,
        num_rows: usize,
        buffers: &Buffer<Buffer<u8>>,
    ) -> DataFrame {
        assert!(skip + self.stride <= entry_words && num_rows * entry_words <= entries.len());
        let row_bytes = entry_words * 8;
        let base = unsafe { (entries.as_ptr() as *const u8).add(skip * 8) };
        let nulls: Vec<u64> = (0..num_rows)
            .map(|i| unsafe { self.read_nulls(base.add(i * row_bytes)) })
            .collect();
        let any_nulls = nulls.iter().fold(0, |acc, m| acc | m);
        let cols = schema
            .iter()
            .zip(&self.cols)
            .enumerate()
            .map(|(c, ((name, dtype), col))| unsafe {
                let validity = (any_nulls >> c & 1 != 0)
                    .then(|| Bitmap::from_trusted_len_iter(nulls.iter().map(|m| m >> c & 1 == 0)));
                let offset = col.offset;
                let array: Box<dyn Array> = match col.kind {
                    ColKind::Bool => Box::new(BooleanArray::new(
                        ArrowDataType::Boolean,
                        Bitmap::from_trusted_len_iter(
                            (0..num_rows).map(|i| *base.add(i * row_bytes + offset) != 0),
                        ),
                        validity,
                    )),
                    ColKind::View => {
                        let views: Buffer<View> = (0..num_rows)
                            .map(|i| view_at(&entries[i * entry_words + skip..], offset / 8))
                            .collect();
                        let arrow_dtype = col.physical.to_arrow(CompatLevel::newest());
                        if col.physical == DataType::String {
                            Box::new(BinaryViewArrayGeneric::<str>::new_unchecked_unknown_md(
                                arrow_dtype,
                                views,
                                buffers.clone(),
                                validity,
                                None,
                            ))
                        } else {
                            Box::new(BinaryViewArrayGeneric::<[u8]>::new_unchecked_unknown_md(
                                arrow_dtype,
                                views,
                                buffers.clone(),
                                validity,
                                None,
                            ))
                        }
                    },
                    ColKind::Fixed => match &col.physical {
                        #[cfg(feature = "dtype-f16")]
                        DataType::Float16 => read_fixed::<polars_utils::float16::pf16>(
                            base, row_bytes, offset, num_rows, validity,
                        ),
                        DataType::Float32 => {
                            read_fixed::<f32>(base, row_bytes, offset, num_rows, validity)
                        },
                        DataType::Float64 => {
                            read_fixed::<f64>(base, row_bytes, offset, num_rows, validity)
                        },
                        dt => with_match_physical_integer_polars_type!(dt, |$T| {
                            read_fixed::<<$T as PolarsNumericType>::Native>(
                                base, row_bytes, offset, num_rows, validity,
                            )
                        }),
                    },
                };
                let s = Series::from_chunks_and_dtype_unchecked(
                    name.clone(),
                    vec![array],
                    &col.physical,
                );
                s.from_physical_unchecked(dtype).unwrap().into_column()
            })
            .collect();
        unsafe { DataFrame::new_unchecked(num_rows, cols) }
    }
}

/// Whether view `a` into `buffers` has the same bytes as the stored view `b`, whose
/// long bytes are resolved by `stored_long`.
///
/// # Safety
/// Both views must be valid.
#[inline(always)]
unsafe fn view_eq<'a>(
    a: View,
    b: u128,
    buffers: &[Buffer<u8>],
    stored_long: impl Fn(View) -> &'a [u8],
) -> bool {
    let a_bits = a.as_u128();
    if a.length <= View::MAX_INLINE_SIZE {
        a_bits == b
    } else {
        a_bits as u64 == b as u64
            && bytes_eq(
                a.get_external_slice_unchecked(buffers),
                stored_long(std::mem::transmute::<u128, View>(b)),
            )
    }
}

#[inline(always)]
fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() {
        return false;
    }
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    unsafe {
        if (8..=16).contains(&n) {
            let x = pa.cast::<u64>().read_unaligned() ^ pb.cast::<u64>().read_unaligned();
            let y = pa.add(n - 8).cast::<u64>().read_unaligned()
                ^ pb.add(n - 8).cast::<u64>().read_unaligned();
            (x | y) == 0
        } else if (16..=32).contains(&n) {
            let x = pa.cast::<u128>().read_unaligned() ^ pb.cast::<u128>().read_unaligned();
            let y = pa.add(n - 16).cast::<u128>().read_unaligned()
                ^ pb.add(n - 16).cast::<u128>().read_unaligned();
            (x | y) == 0
        } else {
            a == b
        }
    }
}

#[inline(always)]
unsafe fn view_at(row: &[u64], w: usize) -> View {
    let lo = *row.get_unchecked(w);
    let hi = *row.get_unchecked(w + 1);
    std::mem::transmute::<u128, View>(lo as u128 | ((hi as u128) << 64))
}

#[inline(always)]
unsafe fn set_view(row: &mut [u64], w: usize, view: View) {
    let bits = view.as_u128();
    *row.get_unchecked_mut(w) = bits as u64;
    *row.get_unchecked_mut(w + 1) = (bits >> 64) as u64;
}

/// # Safety
/// The view must be a long view whose bytes are at its offset in `bytes`.
#[inline(always)]
unsafe fn long_slice(bytes: &[u8], view: View) -> &[u8] {
    bytes.get_unchecked(view.offset as usize..view.offset as usize + view.length as usize)
}

/// # Safety
/// `num_rows` rows of `row_bytes` bytes start at `base`, each with a `T` at `offset`.
unsafe fn read_fixed<T: NativeType>(
    base: *const u8,
    row_bytes: usize,
    offset: usize,
    num_rows: usize,
    validity: Option<Bitmap>,
) -> Box<dyn Array> {
    let values: Vec<T> = (0..num_rows)
        .map(|i| {
            base.add(i * row_bytes + offset)
                .cast::<T>()
                .read_unaligned()
        })
        .collect();
    Box::new(PrimitiveArray::from_vec(values).with_validity(validity))
}

/// Stores `bytes` in the last of `buffers`, starting a new one when it is full, and
/// returns where they were stored.
fn push_long_bytes(buffers: &mut Vec<Vec<u8>>, bytes: &[u8]) -> (u32, u32) {
    if buffers
        .last()
        .is_none_or(|buf| buf.len() + bytes.len() > buf.capacity())
    {
        let next_cap = buffers.last().map_or(BASE_KEY_BUFFER_CAPACITY, |buf| {
            (2 * buf.capacity()).min(MAX_KEY_BUFFER_CAPACITY)
        });
        buffers.push(Vec::with_capacity(next_cap.max(bytes.len())));
    }
    let buffer_idx = (buffers.len() - 1) as u32;
    let buffer = buffers.last_mut().unwrap();
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);
    (buffer_idx, offset)
}

fn append_long_bytes(buffer: &mut Vec<u8>, bytes: &[u8]) -> u32 {
    let offset = buffer.len() as u32;
    buffer.extend_from_slice(bytes);
    offset
}

/// The values of a key column, with floats canonicalized and every other type
/// reinterpreted as unsigned integers of the same width.
#[derive(Clone, Debug)]
enum ColValues {
    Bool(Bitmap),
    W1(Buffer<u8>),
    W2(Buffer<u16>),
    W4(Buffer<u32>),
    W8(Buffer<u64>),
    W16(Buffer<u128>),
    View(Buffer<View>, Buffer<Buffer<u8>>),
}

fn fixed_values<T: NativeType>(values: Buffer<T>) -> ColValues {
    match size_of::<T>() {
        1 => ColValues::W1(values.try_transmute().unwrap()),
        2 => ColValues::W2(values.try_transmute().unwrap()),
        4 => ColValues::W4(values.try_transmute().unwrap()),
        8 => ColValues::W8(values.try_transmute().unwrap()),
        16 => ColValues::W16(values.try_transmute().unwrap()),
        _ => unreachable!(),
    }
}

/// # Safety
/// The indices must be in-bounds.
unsafe fn gather_buffer<T: Copy>(values: &Buffer<T>, idxs: &[IdxSize]) -> Buffer<T> {
    idxs.iter()
        .map(|i| *values.get_unchecked(*i as usize))
        .collect()
}

/// Clears `ok[r]` when `eq(idxs[r], rows[r])` fails for a valid key.
///
/// # Safety
/// The indices must be in-bounds for `validity`.
#[inline(always)]
unsafe fn verify_col(
    idxs: &[IdxSize],
    rows: &[*const u64],
    ok: &mut [bool],
    validity: Option<&Bitmap>,
    eq: impl Fn(usize, *const u8) -> bool,
) {
    match validity {
        None => {
            for ((ok, i), row) in ok.iter_mut().zip(idxs).zip(rows) {
                *ok &= eq(*i as usize, row.cast());
            }
        },
        Some(v) => {
            for ((ok, i), row) in ok.iter_mut().zip(idxs).zip(rows) {
                if v.get_bit_unchecked(*i as usize) {
                    *ok &= eq(*i as usize, row.cast());
                }
            }
        },
    }
}

/// Calls `write(idxs[r], rows[r])` for each valid key.
///
/// # Safety
/// The indices must be in-bounds for `validity`.
#[inline(always)]
unsafe fn write_col(
    idxs: &[IdxSize],
    rows: &[*mut u64],
    validity: Option<&Bitmap>,
    mut write: impl FnMut(usize, *mut u8),
) {
    match validity {
        None => {
            for (i, row) in idxs.iter().zip(rows) {
                write(*i as usize, row.cast());
            }
        },
        Some(v) => {
            for (i, row) in idxs.iter().zip(rows) {
                if v.get_bit_unchecked(*i as usize) {
                    write(*i as usize, row.cast());
                }
            }
        },
    }
}

#[inline(always)]
fn fold_each(hashes: &mut [u64], validity: Option<&Bitmap>, mut f: impl FnMut(u64, usize) -> u64) {
    match validity {
        None => {
            for (i, h) in hashes.iter_mut().enumerate() {
                *h = f(*h, i);
            }
        },
        Some(validity) => {
            for (i, (h, valid)) in hashes.iter_mut().zip(validity.iter()).enumerate() {
                *h = if valid { f(*h, i) } else { fold(*h, NULL_WORD) };
            }
        },
    }
}

#[derive(Clone, Debug)]
struct KeyColumn {
    values: ColValues,
    /// Only kept when the column has nulls.
    validity: Option<Bitmap>,
    /// Byte offset within the row.
    offset: usize,
}

impl KeyColumn {
    fn new(column: &Column, col: &ColLayout) -> Self {
        let s = column.as_materialized_series().to_physical_repr().rechunk();
        assert_eq!(s.dtype(), &col.physical);
        let validity = s.chunks()[0]
            .validity()
            .filter(|v| v.unset_bits() > 0)
            .cloned();
        let values = match s.dtype() {
            DataType::Boolean => {
                ColValues::Bool(s.bool().unwrap().downcast_as_array().values().clone())
            },
            DataType::String => {
                let arr = s.str().unwrap().downcast_as_array();
                ColValues::View(arr.views().clone(), arr.data_buffers().clone())
            },
            DataType::Binary => {
                let arr = s.binary().unwrap().downcast_as_array();
                ColValues::View(arr.views().clone(), arr.data_buffers().clone())
            },
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                let arr = s.f16().unwrap().downcast_as_array();
                ColValues::W2(
                    arr.values()
                        .iter()
                        .map(|x| polars_utils::total_ord::canonical_f16(*x).to_bits())
                        .collect(),
                )
            },
            DataType::Float32 => {
                let arr = s.f32().unwrap().downcast_as_array();
                ColValues::W4(
                    arr.values()
                        .iter()
                        .map(|x| canonical_f32(*x).to_bits())
                        .collect(),
                )
            },
            DataType::Float64 => {
                let arr = s.f64().unwrap().downcast_as_array();
                ColValues::W8(
                    arr.values()
                        .iter()
                        .map(|x| canonical_f64(*x).to_bits())
                        .collect(),
                )
            },
            dt => with_match_physical_integer_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_phys_any().downcast_ref().unwrap();
                fixed_values(ca.downcast_as_array().values().clone())
            }),
        };
        Self {
            values,
            validity,
            offset: col.offset,
        }
    }

    /// Folds this column into the hash of each row.
    fn hash_into(&self, hashes: &mut [u64], random_state: &PlRandomState) {
        let validity = self.validity.as_ref();
        unsafe {
            match &self.values {
                ColValues::Bool(b) => fold_each(hashes, validity, |h, i| {
                    fold(h, b.get_bit_unchecked(i) as u64)
                }),
                ColValues::W1(v) => {
                    fold_each(hashes, validity, |h, i| fold(h, *v.get_unchecked(i) as u64))
                },
                ColValues::W2(v) => {
                    fold_each(hashes, validity, |h, i| fold(h, *v.get_unchecked(i) as u64))
                },
                ColValues::W4(v) => {
                    fold_each(hashes, validity, |h, i| fold(h, *v.get_unchecked(i) as u64))
                },
                ColValues::W8(v) => {
                    fold_each(hashes, validity, |h, i| fold(h, *v.get_unchecked(i)))
                },
                ColValues::W16(v) => fold_each(hashes, validity, |h, i| {
                    let x = *v.get_unchecked(i);
                    fold(fold(h, x as u64), (x >> 64) as u64)
                }),
                ColValues::View(views, buffers) => fold_each(hashes, validity, |h, i| {
                    let view = *views.get_unchecked(i);
                    let bits = view.as_u128();
                    let hi = if view.length <= View::MAX_INLINE_SIZE {
                        (bits >> 64) as u64
                    } else {
                        random_state.hash_one(view.get_external_slice_unchecked(buffers))
                    };
                    fold(fold(h, bits as u64), hi)
                }),
            }
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    unsafe fn gather_unchecked(&self, idxs: &[IdxSize]) -> Self {
        let values = match &self.values {
            ColValues::Bool(b) => ColValues::Bool(take_bitmap_unchecked(b, idxs)),
            ColValues::W1(v) => ColValues::W1(gather_buffer(v, idxs)),
            ColValues::W2(v) => ColValues::W2(gather_buffer(v, idxs)),
            ColValues::W4(v) => ColValues::W4(gather_buffer(v, idxs)),
            ColValues::W8(v) => ColValues::W8(gather_buffer(v, idxs)),
            ColValues::W16(v) => ColValues::W16(gather_buffer(v, idxs)),
            ColValues::View(v, buffers) => ColValues::View(gather_buffer(v, idxs), buffers.clone()),
        };
        Self {
            values,
            validity: self
                .validity
                .as_ref()
                .map(|v| take_bitmap_unchecked(v, idxs)),
            offset: self.offset,
        }
    }
}

#[derive(Clone, Debug)]
enum KeyData {
    Columns(Vec<KeyColumn>),
    /// Rows whose long views all point into `buffers`.
    Rows {
        rows: Buffer<u64>,
        buffers: Arc<[Vec<u8>]>,
    },
}

/// Keys of several columns, with a hash per key. Keys come either as columns or as
/// rows of the layout.
#[derive(Clone, Debug)]
pub struct KeyRowKeys {
    layout: Arc<KeyRowLayout>,
    pub hashes: UInt64Array,
    /// Keys with a null, when nulls are not keys.
    pub validity: Option<Bitmap>,
    data: KeyData,
}

impl KeyRowKeys {
    pub fn from_columns(
        columns: &[Column],
        layout: Arc<KeyRowLayout>,
        random_state: &PlRandomState,
        null_is_valid: bool,
    ) -> Self {
        assert_eq!(columns.len(), layout.cols.len());
        let len = columns[0].len();
        assert!(columns.iter().all(|c| c.len() == len));
        let cols: Vec<KeyColumn> = columns
            .iter()
            .zip(&layout.cols)
            .map(|(column, col)| KeyColumn::new(column, col))
            .collect();
        let mut hashes = vec![random_state.hash_one(HASH_MULTIPLE); len];
        for col in &cols {
            col.hash_into(&mut hashes, random_state);
        }
        let validity = if null_is_valid {
            None
        } else {
            combine_validities_and_many(
                &cols.iter().map(|c| c.validity.as_ref()).collect::<Vec<_>>(),
            )
        };
        Self {
            layout,
            hashes: PrimitiveArray::from_vec(hashes),
            validity,
            data: KeyData::Columns(cols),
        }
    }

    fn from_rows(
        layout: Arc<KeyRowLayout>,
        hashes: Vec<u64>,
        rows: Vec<u64>,
        buffers: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            layout,
            hashes: PrimitiveArray::from_vec(hashes),
            validity: None,
            data: KeyData::Rows {
                rows: Buffer::from(rows),
                buffers: buffers.into(),
            },
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn for_each_hash<F: FnMut(IdxSize, Option<u64>)>(&self, f: F) {
        for_each_hash_prehashed(self.hashes.values().as_slice(), self.validity.as_ref(), f);
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn for_each_hash_subset<F: FnMut(IdxSize, Option<u64>)>(
        &self,
        subset: &[IdxSize],
        f: F,
    ) {
        for_each_hash_subset_prehashed(
            self.hashes.values().as_slice(),
            self.validity.as_ref(),
            subset,
            f,
        );
    }

    /// Whether key `i` equals a stored row, whose long views are resolved by
    /// `stored_long`.
    ///
    /// # Safety
    /// `i` must be in-bounds and `row` must be a row of this layout.
    #[inline(always)]
    unsafe fn eq_stored<'a>(
        &self,
        i: usize,
        row: &[u64],
        stored_long: impl Fn(View) -> &'a [u8],
    ) -> bool {
        match &self.data {
            KeyData::Columns(cols) => self.layout.eq_columns(cols, i, row, stored_long),
            KeyData::Rows { rows, buffers } => {
                let stride = self.layout.stride;
                self.layout.rows_eq(
                    rows.get_unchecked(i * stride..(i + 1) * stride),
                    row,
                    |a, b| bytes_eq(a.get_external_slice_unchecked(buffers), stored_long(b)),
                )
            },
        }
    }

    /// Clears `ok[r]` when key `idxs[r]` differs from the stored row at `rows[r]`,
    /// whose long views are resolved by `stored_long`.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    unsafe fn verify<'a>(
        &self,
        idxs: &[IdxSize],
        rows: &[*const u64],
        ok: &mut [bool],
        stored_long: impl Fn(View) -> &'a [u8],
    ) {
        match &self.data {
            KeyData::Columns(cols) => self
                .layout
                .verify_columns(cols, idxs, rows, ok, stored_long),
            KeyData::Rows { .. } => {
                let stride = self.layout.stride;
                for ((ok, i), row) in ok.iter_mut().zip(idxs).zip(rows) {
                    *ok &= self.eq_stored(
                        *i as usize,
                        std::slice::from_raw_parts(*row, stride),
                        &stored_long,
                    );
                }
            },
        }
    }

    /// Writes the keys `idxs[r]` into the zeroed rows at `rows[r]`, storing the bytes
    /// of long views with `store_long`, which returns the buffer index and offset of
    /// the copy.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    unsafe fn write_rows(
        &self,
        idxs: &[IdxSize],
        rows: &[*mut u64],
        mut store_long: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        match &self.data {
            KeyData::Columns(cols) => self
                .layout
                .write_columns_batch(cols, idxs, rows, store_long),
            KeyData::Rows { .. } => {
                let stride = self.layout.stride;
                for (i, row) in idxs.iter().zip(rows) {
                    self.write_row(
                        *i as usize,
                        std::slice::from_raw_parts_mut(*row, stride),
                        &mut store_long,
                    );
                }
            },
        }
    }

    /// Writes key `i` into a zeroed row, storing the bytes of its long views with
    /// `store_long`, which returns the buffer index and offset of the copy.
    ///
    /// # Safety
    /// `i` must be in-bounds and `row` must have `stride` words.
    #[inline(always)]
    unsafe fn write_row(
        &self,
        i: usize,
        row: &mut [u64],
        store_long: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        match &self.data {
            KeyData::Columns(cols) => self.layout.write_columns(cols, i, row, store_long),
            KeyData::Rows { rows, buffers } => {
                let stride = self.layout.stride;
                row.copy_from_slice(rows.get_unchecked(i * stride..(i + 1) * stride));
                self.layout.repoint_long_views(
                    row,
                    |view| view.get_external_slice_unchecked(buffers),
                    store_long,
                );
            },
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather_unchecked(&self, idxs: &[IdxSize]) -> Self {
        let idx_arr = polars_arrow::ffi::mmap::slice(idxs);
        let data = match &self.data {
            KeyData::Columns(cols) => {
                KeyData::Columns(cols.iter().map(|c| c.gather_unchecked(idxs)).collect())
            },
            KeyData::Rows { rows, buffers } => {
                let stride = self.layout.stride;
                let mut out = Vec::with_capacity(idxs.len() * stride);
                for i in idxs {
                    let i = *i as usize;
                    out.extend_from_slice(rows.get_unchecked(i * stride..(i + 1) * stride));
                }
                KeyData::Rows {
                    rows: Buffer::from(out),
                    buffers: buffers.clone(),
                }
            },
        };
        Self {
            layout: self.layout.clone(),
            hashes: polars_compute::gather::primitive::take_primitive_unchecked(
                &self.hashes,
                &idx_arr,
            ),
            validity: self
                .validity
                .as_ref()
                .map(|v| take_bitmap_unchecked(v, idxs)),
            data,
        }
    }
}

/// Candidates found by hash, to be verified together.
struct VerifyBatch {
    pos: Vec<usize>,
    idxs: Vec<IdxSize>,
    entries: Vec<IdxSize>,
    rows: Vec<*const u64>,
    ok: Vec<bool>,
    new_pos: Vec<usize>,
    new_idxs: Vec<IdxSize>,
    new_entries: Vec<IdxSize>,
    new_rows: Vec<*mut u64>,
}

impl Default for VerifyBatch {
    fn default() -> Self {
        let n = VERIFY_BATCH_SIZE;
        Self {
            pos: Vec::with_capacity(n),
            idxs: Vec::with_capacity(n),
            entries: Vec::with_capacity(n),
            rows: Vec::with_capacity(n),
            ok: Vec::with_capacity(n),
            new_pos: Vec::with_capacity(n),
            new_idxs: Vec::with_capacity(n),
            new_entries: Vec::with_capacity(n),
            new_rows: Vec::with_capacity(n),
        }
    }
}

impl VerifyBatch {
    fn clear(&mut self) {
        self.pos.clear();
        self.idxs.clear();
        self.entries.clear();
        self.new_pos.clear();
        self.new_idxs.clear();
        self.new_entries.clear();
    }

    #[inline(always)]
    fn push_new(&mut self, pos: usize, i: IdxSize, entry: IdxSize) {
        self.new_pos.push(pos);
        self.new_idxs.push(i);
        self.new_entries.push(entry);
    }

    /// Writes the rows of the new entries, which are zeroed.
    ///
    /// # Safety
    /// The new entries must be in-bounds for `keys` and `entries`.
    unsafe fn write_new(
        &mut self,
        keys: &KeyRowKeys,
        entries: &mut [u64],
        entry_words: usize,
        buffers: &mut Vec<Vec<u8>>,
    ) {
        let base = entries.as_mut_ptr();
        self.new_rows.clear();
        self.new_rows.extend(
            self.new_entries
                .iter()
                .map(|j| base.add(*j as usize * entry_words + 1)),
        );
        keys.write_rows(&self.new_idxs, &self.new_rows, |bytes| {
            push_long_bytes(buffers, bytes)
        });
    }

    #[inline(always)]
    fn push(&mut self, pos: usize, i: IdxSize, entry: IdxSize) {
        self.pos.push(pos);
        self.idxs.push(i);
        self.entries.push(entry);
    }

    /// # Safety
    /// The candidates must be in-bounds for `keys` and `entries`.
    unsafe fn verify(
        &mut self,
        keys: &KeyRowKeys,
        entries: &[u64],
        entry_words: usize,
        buffers: &[Vec<u8>],
    ) {
        self.rows.clear();
        self.rows.extend(
            self.entries
                .iter()
                .map(|j| entries.as_ptr().add(*j as usize * entry_words + 1)),
        );
        self.ok.clear();
        self.ok.resize(self.idxs.len(), true);
        keys.verify(&self.idxs, &self.rows, &mut self.ok, |view| {
            view.get_external_slice_unchecked(buffers)
        });
    }

    /// The position and key index of each candidate that is a different key.
    fn mismatches(&self) -> impl Iterator<Item = (usize, IdxSize)> + '_ {
        self.ok
            .iter()
            .enumerate()
            .filter(|(_, ok)| !**ok)
            .map(|(c, _)| (self.pos[c], self.idxs[c]))
    }
}

/// An IndexMap from key rows to values. It owns copies of the bytes of long views.
pub struct KeyRowIndexMap<V> {
    table: HashTable<IdxSize>,
    /// Per key its hash followed by its row.
    entries: Vec<u64>,
    values: Vec<V>,
    buffers: Vec<Vec<u8>>,
    layout: Option<Arc<KeyRowLayout>>,
    /// A random odd number that hashes are multiplied by before they are probed.
    seed: u64,
}

impl<V> Default for KeyRowIndexMap<V> {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            entries: Vec::new(),
            values: Vec::new(),
            buffers: Vec::new(),
            layout: None,
            seed: rand::random::<u64>() | 1,
        }
    }
}

impl<V> KeyRowIndexMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry_words(&self) -> usize {
        self.layout.as_ref().map_or(0, |l| l.stride + 1)
    }

    pub fn reserve(&mut self, additional: usize) {
        let (entries, entry_words, seed) = (&self.entries, self.entry_words(), self.seed);
        self.table.reserve(additional, |i| unsafe {
            entries
                .get_unchecked(*i as usize * entry_words)
                .wrapping_mul(seed)
        });
        self.entries.reserve(additional * entry_words);
        self.values.reserve(additional);
    }

    pub(crate) fn len(&self) -> IdxSize {
        self.values.len() as IdxSize
    }

    /// Gets the index by insertion order of key `i` of `keys`.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of the keys in this map.
    #[inline(always)]
    pub unsafe fn get_index_of(&self, keys: &KeyRowKeys, i: usize) -> Option<IdxSize> {
        let layout = self.layout.as_deref()?;
        let hash = keys.hashes.value_unchecked(i);
        let entry_words = layout.stride + 1;
        self.table
            .find(hash.wrapping_mul(self.seed), |j| {
                let entry = self
                    .entries
                    .get_unchecked(*j as usize * entry_words..(*j as usize + 1) * entry_words);
                *entry.get_unchecked(0) == hash
                    && keys.eq_stored(i, entry.get_unchecked(1..), |view| {
                        view.get_external_slice_unchecked(&self.buffers)
                    })
            })
            .copied()
    }

    /// Returns the index of key `i` of `keys`, inserting it with `value()` if it is
    /// new, and whether it was inserted.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of the keys in this map.
    #[inline(always)]
    pub unsafe fn get_or_insert_with(
        &mut self,
        keys: &KeyRowKeys,
        i: usize,
        value: impl FnOnce() -> V,
    ) -> (IdxSize, bool) {
        let entry_words = self
            .layout
            .get_or_insert_with(|| keys.layout.clone())
            .stride
            + 1;
        let hash = keys.hashes.value_unchecked(i);
        let (entries, buffers, seed) = (&self.entries, &self.buffers, self.seed);
        let entry = self.table.entry(
            hash.wrapping_mul(seed),
            |j| {
                let entry = entries
                    .get_unchecked(*j as usize * entry_words..(*j as usize + 1) * entry_words);
                *entry.get_unchecked(0) == hash
                    && keys.eq_stored(i, entry.get_unchecked(1..), |view| {
                        view.get_external_slice_unchecked(buffers)
                    })
            },
            |j| {
                entries
                    .get_unchecked(*j as usize * entry_words)
                    .wrapping_mul(seed)
            },
        );
        match entry {
            TEntry::Occupied(o) => (*o.get(), false),
            TEntry::Vacant(v) => {
                let idx = self.values.len() as IdxSize;
                v.insert(idx);
                Self::push_entry(
                    &mut self.entries,
                    &mut self.buffers,
                    entry_words,
                    keys,
                    i,
                    hash,
                );
                self.values.push(value());
                (idx, true)
            },
        }
    }

    /// # Safety
    /// `i` must be in-bounds.
    #[inline(always)]
    unsafe fn push_entry(
        entries: &mut Vec<u64>,
        buffers: &mut Vec<Vec<u8>>,
        entry_words: usize,
        keys: &KeyRowKeys,
        i: usize,
        hash: u64,
    ) {
        entries.push(hash);
        let start = entries.len();
        entries.resize(start + entry_words - 1, 0);
        keys.write_row(i, &mut entries[start..], |bytes| {
            push_long_bytes(buffers, bytes)
        });
    }

    /// # Safety
    /// The map must have a layout.
    #[inline(always)]
    unsafe fn find_hash(&self, hash: u64, entry_words: usize) -> Option<IdxSize> {
        self.table
            .find(hash.wrapping_mul(self.seed), |j| {
                *self.entries.get_unchecked(*j as usize * entry_words) == hash
            })
            .copied()
    }

    /// Pushes the index of each key `idxs[r]` of `keys` to `out`, or `IdxSize::MAX`
    /// when the key is absent or null.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of the keys in
    /// this map.
    pub unsafe fn get_indices_of(
        &self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        out: &mut Vec<IdxSize>,
    ) {
        let Some(layout) = self.layout.as_deref() else {
            out.extend(std::iter::repeat_n(IdxSize::MAX, idxs.len()));
            return;
        };
        let entry_words = layout.stride + 1;
        let mut batch = VerifyBatch::default();
        for chunk in idxs.chunks(VERIFY_BATCH_SIZE) {
            let start = out.len();
            batch.clear();
            for (r, i) in chunk.iter().enumerate() {
                let is_valid = keys
                    .validity
                    .as_ref()
                    .is_none_or(|v| v.get_bit_unchecked(*i as usize));
                let hash = keys.hashes.value_unchecked(*i as usize);
                match self.find_hash(hash, entry_words).filter(|_| is_valid) {
                    Some(j) => {
                        out.push(j);
                        batch.push(r, *i, j);
                    },
                    None => out.push(IdxSize::MAX),
                }
            }
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            for (r, i) in batch.mismatches() {
                *out.get_unchecked_mut(start + r) =
                    self.get_index_of(keys, i as usize).unwrap_or(IdxSize::MAX);
            }
        }
    }

    /// Pushes the index of each key `idxs[r]` of `keys` to `out`, inserting missing
    /// keys with `value(r)`. New keys get indices in the order they first occur.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of the keys in
    /// this map.
    pub unsafe fn get_or_insert_batch(
        &mut self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        mut value: impl FnMut(usize) -> V,
        out: &mut Vec<IdxSize>,
    ) {
        let entry_words = self
            .layout
            .get_or_insert_with(|| keys.layout.clone())
            .stride
            + 1;
        let seed = self.seed;
        let mut batch = VerifyBatch::default();
        for (c, chunk) in idxs.chunks(VERIFY_BATCH_SIZE).enumerate() {
            let (start, chunk_start) = (out.len(), c * VERIFY_BATCH_SIZE);
            let first_new = self.len();
            let mut next = first_new;
            let num_buffers = self.buffers.len();
            let last_buffer_len = self.buffers.last().map_or(0, Vec::len);
            batch.clear();
            for (r, i) in chunk.iter().enumerate() {
                let hash = keys.hashes.value_unchecked(*i as usize);
                let entries = &self.entries;
                let entry = self.table.entry(
                    hash.wrapping_mul(seed),
                    |j| *entries.get_unchecked(*j as usize * entry_words) == hash,
                    |j| {
                        entries
                            .get_unchecked(*j as usize * entry_words)
                            .wrapping_mul(seed)
                    },
                );
                match entry {
                    TEntry::Occupied(o) => {
                        let j = *o.get();
                        out.push(j);
                        batch.push(r, *i, j);
                    },
                    TEntry::Vacant(v) => {
                        v.insert(next);
                        self.entries.push(hash);
                        self.entries.resize(self.entries.len() + entry_words - 1, 0);
                        out.push(next);
                        batch.push_new(r, *i, next);
                        next += 1;
                    },
                }
            }
            batch.write_new(keys, &mut self.entries, entry_words, &mut self.buffers);
            batch.verify(keys, &self.entries, entry_words, &self.buffers);
            if !batch.ok.contains(&false) {
                self.values
                    .extend(batch.new_pos.iter().map(|r| value(chunk_start + r)));
                continue;
            }

            for j in first_new..next {
                let hash = *self.entries.get_unchecked(j as usize * entry_words);
                self.table
                    .find_entry(hash.wrapping_mul(seed), |k| *k == j)
                    .unwrap()
                    .remove();
            }
            self.entries.truncate(first_new as usize * entry_words);
            self.buffers.truncate(num_buffers);
            if let Some(buffer) = self.buffers.last_mut() {
                buffer.truncate(last_buffer_len);
            }
            out.truncate(start);
            for (r, i) in chunk.iter().enumerate() {
                out.push(
                    self.get_or_insert_with(keys, *i as usize, || value(chunk_start + r))
                        .0,
                );
            }
        }
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub unsafe fn value_unchecked(&self, idx: IdxSize) -> &V {
        self.values.get_unchecked(idx as usize)
    }

    /// # Safety
    /// `idx` must be less than `len()`.
    #[inline(always)]
    pub unsafe fn value_unchecked_mut(&mut self, idx: IdxSize) -> &mut V {
        self.values.get_unchecked_mut(idx as usize)
    }

    pub fn get_value(&self, idx: IdxSize) -> Option<&V> {
        self.values.get(idx as usize)
    }

    /// Returns the keys as columns of `schema`, in insertion order.
    pub fn keys_frame(&self, schema: &Schema) -> DataFrame {
        let Some(layout) = &self.layout else {
            return DataFrame::empty_with_schema(schema);
        };
        let buffers = self
            .buffers
            .iter()
            .map(|b| Buffer::from(b.clone()))
            .collect();
        layout.decode(
            schema,
            &self.entries,
            layout.stride + 1,
            1,
            self.values.len(),
            &buffers,
        )
    }
}

/// The keys of a hot grouper as rows, each owning the bytes of its long views.
pub struct HotKeyRows {
    layout: Arc<KeyRowLayout>,
    hashes: Vec<u64>,
    rows: Vec<u64>,
    long: Vec<Vec<u8>>,
}

impl HotKeyRows {
    pub fn new(layout: Arc<KeyRowLayout>) -> Self {
        Self {
            layout,
            hashes: Vec::new(),
            rows: Vec::new(),
            long: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// # Safety
    /// `k` must be in-bounds.
    #[inline(always)]
    pub unsafe fn hash(&self, k: IdxSize) -> u64 {
        *self.hashes.get_unchecked(k as usize)
    }

    /// Whether hot key `k` is key `i` of `keys`.
    ///
    /// # Safety
    /// `k` and `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub unsafe fn eq_key(&self, k: IdxSize, keys: &KeyRowKeys, i: usize) -> bool {
        let (k, stride) = (k as usize, self.layout.stride);
        let row = self.rows.get_unchecked(k * stride..(k + 1) * stride);
        keys.eq_stored(i, row, |view| view.get_external_slice_unchecked(&self.long))
    }

    /// Clears `ok[r]` when key `idxs[r]` of `keys` is not hot key `ks[r]`.
    ///
    /// # Safety
    /// The indices must be in-bounds, and `keys` must have the layout of these keys.
    pub unsafe fn verify(
        &self,
        keys: &KeyRowKeys,
        idxs: &[IdxSize],
        ks: &[IdxSize],
        rows: &mut Vec<*const u64>,
        ok: &mut [bool],
    ) {
        let stride = self.layout.stride;
        rows.clear();
        rows.extend(
            ks.iter()
                .map(|k| self.rows.as_ptr().add(*k as usize * stride)),
        );
        keys.verify(idxs, rows, ok, |view| {
            view.get_external_slice_unchecked(&self.long)
        });
    }

    /// Adds key `i` of `keys`, returning its index.
    ///
    /// # Safety
    /// `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub unsafe fn push(&mut self, keys: &KeyRowKeys, i: usize) -> IdxSize {
        let k = self.hashes.len();
        self.hashes.push(0);
        self.rows.resize(self.rows.len() + self.layout.stride, 0);
        if self.layout.has_views() {
            self.long.push(Vec::new());
        }
        self.write(k, keys, i);
        k as IdxSize
    }

    /// Replaces hot key `k` by key `i` of `keys`.
    ///
    /// # Safety
    /// `k` and `i` must be in-bounds, and `keys` must have the layout of these keys.
    #[inline(always)]
    pub unsafe fn replace(&mut self, k: IdxSize, keys: &KeyRowKeys, i: usize) {
        let (k, stride) = (k as usize, self.layout.stride);
        self.rows
            .get_unchecked_mut(k * stride..(k + 1) * stride)
            .fill(0);
        if self.layout.has_views() {
            self.long.get_unchecked_mut(k).clear();
        }
        self.write(k, keys, i);
    }

    unsafe fn write(&mut self, k: usize, keys: &KeyRowKeys, i: usize) {
        let stride = self.layout.stride;
        *self.hashes.get_unchecked_mut(k) = keys.hashes.value_unchecked(i);
        let row = self.rows.get_unchecked_mut(k * stride..(k + 1) * stride);
        if self.layout.has_views() {
            let long = self.long.get_unchecked_mut(k);
            keys.write_row(i, row, |bytes| (k as u32, append_long_bytes(long, bytes)));
        } else {
            keys.write_row(i, row, |_| unreachable!());
        }
    }

    /// Adds hot key `k` to `collector`.
    ///
    /// # Safety
    /// `k` must be in-bounds.
    pub unsafe fn collect(&self, k: IdxSize, collector: &mut KeyRowCollector) {
        let (k, stride) = (k as usize, self.layout.stride);
        let row = self.rows.get_unchecked(k * stride..(k + 1) * stride);
        let long = self.long.get(k).map_or(&[][..], |l| l.as_slice());
        collector.push(&self.layout, *self.hashes.get_unchecked(k), row, long);
    }

    /// Returns all hot keys, in key order.
    pub fn keys(&self) -> KeyRowKeys {
        let mut collector = KeyRowCollector::default();
        for k in 0..self.len() {
            unsafe { self.collect(k as IdxSize, &mut collector) };
        }
        collector.take(self.layout.clone())
    }
}

/// Collects key rows, re-pointing their long views into shared buffers.
#[derive(Default)]
pub struct KeyRowCollector {
    hashes: Vec<u64>,
    rows: Vec<u64>,
    buffers: Vec<Vec<u8>>,
}

impl KeyRowCollector {
    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// # Safety
    /// `row` must be a row of `layout` whose long views are at their offset in `long`.
    unsafe fn push(&mut self, layout: &KeyRowLayout, hash: u64, row: &[u64], long: &[u8]) {
        self.hashes.push(hash);
        let start = self.rows.len();
        self.rows.extend_from_slice(row);
        let buffers = &mut self.buffers;
        layout.repoint_long_views(
            &mut self.rows[start..],
            |view| long_slice(long, view),
            |b| push_long_bytes(buffers, b),
        );
    }

    pub fn take(&mut self, layout: Arc<KeyRowLayout>) -> KeyRowKeys {
        KeyRowKeys::from_rows(
            layout,
            std::mem::take(&mut self.hashes),
            std::mem::take(&mut self.rows),
            std::mem::take(&mut self.buffers),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(df: &DataFrame, null_is_valid: bool, random_state: &PlRandomState) -> KeyRowKeys {
        let layout = KeyRowLayout::new(df.columns().iter().map(|c| c.dtype()));
        KeyRowKeys::from_columns(
            df.columns(),
            Arc::new(layout.unwrap()),
            random_state,
            null_is_valid,
        )
    }

    #[test]
    fn layout_places_wide_columns_first() {
        let layout =
            KeyRowLayout::new(&[DataType::Int32, DataType::String, DataType::Int64]).unwrap();
        let offsets: Vec<usize> = layout.cols.iter().map(|c| c.offset).collect();
        assert_eq!(offsets, [24, 0, 16]);
        assert_eq!(layout.null_offset, 28);
        assert_eq!(layout.stride, 4);
        assert_eq!(layout.view_words, [0]);
        assert_eq!(layout.plain_words, [0, 2, 3]);

        let two_i64 = KeyRowLayout::new(&[DataType::Int64, DataType::Int64]).unwrap();
        assert_eq!(two_i64.stride, 3);
        assert!(KeyRowLayout::new(&[DataType::Int64, DataType::Null]).is_none());
        assert!(
            KeyRowLayout::new(&[DataType::String, DataType::Binary, DataType::String]).is_none()
        );
    }

    #[test]
    fn equal_content_has_equal_rows_and_hashes() {
        let long = "a string that does not fit in a view";
        let a = df!(
            "s" => [Some("x"), Some(long), None, Some(long)],
            "f" => [Some(0.0), Some(f64::NAN), Some(1.0), None],
        )
        .unwrap();
        // Same content, but the long strings live in other buffers at other offsets.
        let b = df!(
            "s" => [Some(long), Some("x"), Some("padding to move the next string"), Some(long)],
            "f" => [None, Some(-0.0), Some(2.0), Some(-f64::NAN)],
        )
        .unwrap();
        let random_state = PlRandomState::default();
        let (ka, kb) = (keys(&a, true, &random_state), keys(&b, true, &random_state));
        let mut map = KeyRowIndexMap::<()>::new();
        unsafe {
            let groups: Vec<IdxSize> = (0..4)
                .map(|i| map.get_or_insert_with(&ka, i, || ()).0)
                .collect();
            assert_eq!(groups, [0, 1, 2, 3]);
            let found: Vec<Option<IdxSize>> = (0..4).map(|i| map.get_index_of(&kb, i)).collect();
            assert_eq!(found, [Some(3), Some(0), None, Some(1)]);
            assert_eq!(ka.hashes.value(3), kb.hashes.value(0));
            assert_eq!(ka.hashes.value(1), kb.hashes.value(3));
        }
        let out = map.keys_frame(a.schema());
        assert!(out.equals_missing(&a));

        // The same keys as rows, as a hot grouper hands them over.
        let mut hot = HotKeyRows::new(ka.layout.clone());
        unsafe {
            for i in 0..4 {
                hot.push(&ka, i);
            }
            assert!((0..4).all(|i| hot.eq_key(i as IdxSize, &ka, i)));
            assert!(!hot.eq_key(0, &ka, 1));
        }
        let rows = hot.keys();
        let mut map = KeyRowIndexMap::<()>::new();
        unsafe {
            for i in 0..4 {
                map.get_or_insert_with(&rows, i, || ());
            }
            assert_eq!(map.get_or_insert_with(&rows, 3, || ()), (3, false));
            let found: Vec<Option<IdxSize>> = (0..4).map(|i| map.get_index_of(&kb, i)).collect();
            assert_eq!(found, [Some(3), Some(0), None, Some(1)]);
        }
        assert!(map.keys_frame(a.schema()).equals_missing(&a));
    }

    #[test]
    fn colliding_hashes_are_told_apart() {
        let df = df!(
            "a" => [Some(1i64), Some(2), None, Some(1), Some(2), None],
            "s" => ["x", "y", "x", "x", "y", "x"],
        )
        .unwrap();
        let mut ks = keys(&df, true, &PlRandomState::default());
        ks.hashes = PrimitiveArray::from_vec(vec![42; df.height()]);
        let idxs: Vec<IdxSize> = (0..6).collect();
        let mut map = KeyRowIndexMap::<()>::new();
        let (mut groups, mut found) = (Vec::new(), Vec::new());
        unsafe {
            map.get_or_insert_batch(&ks, &idxs, |_| (), &mut groups);
            map.get_indices_of(&ks, &[4, 5, 3], &mut found);
        }
        assert_eq!(groups, [0, 1, 2, 0, 1, 2]);
        assert_eq!(found, [1, 2, 0]);

        let mut hot = HotKeyRows::new(ks.layout.clone());
        unsafe {
            for i in 0..3 {
                hot.push(&ks, i);
            }
        }
        let rows = hot.keys();
        let mut map = KeyRowIndexMap::<()>::new();
        let (mut groups, mut found) = (Vec::new(), Vec::new());
        unsafe {
            map.get_or_insert_batch(&rows, &[2, 1, 0, 1], |_| (), &mut groups);
            map.get_indices_of(&ks, &idxs, &mut found);
        }
        assert_eq!(groups, [0, 1, 2, 1]);
        assert_eq!(found, [2, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn colliding_new_keys_get_indices_in_order() {
        let df = df!("a" => [1i64, 2, 3, 2], "b" => [1i64, 2, 3, 2]).unwrap();
        let mut ks = keys(&df, true, &PlRandomState::default());
        ks.hashes = PrimitiveArray::from_vec(vec![42, 42, 7, 42]);
        let mut map = KeyRowIndexMap::<IdxSize>::new();
        let mut groups = Vec::new();
        unsafe {
            map.get_or_insert_batch(&ks, &[0], |r| r as IdxSize, &mut groups);
            map.get_or_insert_batch(&ks, &[1, 2, 3], |r| r as IdxSize, &mut groups);
        }
        assert_eq!(groups, [0, 1, 2, 1]);
        let values: Vec<IdxSize> = (0..3).map(|g| *map.get_value(g).unwrap()).collect();
        assert_eq!(values, [0, 0, 1]);
        assert!(map.keys_frame(df.schema()).equals(&df.slice(0, 3)));
    }

    #[test]
    #[should_panic]
    fn columns_of_unequal_length_are_rejected() {
        let columns = [
            Column::new("a".into(), [1i64, 2]),
            Column::new("b".into(), [1i64]),
        ];
        let layout = KeyRowLayout::new(columns.iter().map(|c| c.dtype())).unwrap();
        KeyRowKeys::from_columns(&columns, Arc::new(layout), &PlRandomState::default(), true);
    }

    #[test]
    fn bytes_eq_compares_every_byte() {
        let a: Vec<u8> = (0..41).collect();
        for n in 0..=40 {
            let x = &a[..n];
            assert!(bytes_eq(x, &a[..n]));
            assert!(n == 0 || !bytes_eq(x, &a[1..=n]));
            for i in 0..n {
                let mut y = x.to_vec();
                y[i] ^= 1;
                assert!(!bytes_eq(x, &y));
            }
        }
    }

    #[test]
    fn nulls_are_keys_only_when_valid() {
        let df = df!("a" => [Some(1i64), None], "b" => [Some(true), Some(false)]).unwrap();
        assert!(
            keys(&df, true, &PlRandomState::default())
                .validity
                .is_none()
        );
        let invalid = keys(&df, false, &PlRandomState::default())
            .validity
            .unwrap();
        assert_eq!(invalid.iter().collect::<Vec<_>>(), [true, false]);
    }
}
