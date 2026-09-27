use std::cmp::Reverse;

use polars_arrow::array::{Array, BinaryViewArrayGeneric, BooleanArray, PrimitiveArray, View};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::IdxSize;

use super::keys::{ColValues, KeyColumn};

const MAX_KEY_COLUMNS: usize = 64;
const MAX_VIEW_COLUMNS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColKind {
    Bool,
    Fixed,
    View,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ColLayout {
    kind: ColKind,
    pub(super) physical: DataType,
    /// Byte offset within the row.
    pub(super) offset: usize,
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

/// The row format of keys of one schema. It depends only on the key dtypes: the null
/// bits are always present, also when nulls are not keys, so keys that disagree on
/// whether nulls match (the two sides of an anti join) share one layout.
///
/// Rows are written into zeroed words and null values are left zero, so equal keys
/// have equal words, including padding. The one exception is the second word of a
/// long view, which holds where its bytes are stored: equality skips that word and
/// compares the bytes instead.
#[derive(Debug, PartialEq)]
pub(crate) struct KeyRowLayout {
    pub(super) cols: Vec<ColLayout>,
    /// Byte offset of the null bits, one per column.
    pub(super) null_offset: usize,
    null_width: usize,
    pub(super) stride_words: usize,
    /// The first word of each view column, in column order.
    pub(super) view_words: Vec<usize>,
    /// Every word except the second half of each view.
    pub(super) plain_words: Vec<usize>,
}

impl KeyRowLayout {
    /// The layout of keys of these dtypes, or `None` when they are not stored as key
    /// rows.
    pub(crate) fn new<'a>(dtypes: impl IntoIterator<Item = &'a DataType>) -> Option<Self> {
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

        let stride_words = offset.div_ceil(8);
        let view_words: Vec<usize> = cols
            .iter()
            .filter(|c| c.kind == ColKind::View)
            .map(|c| c.offset / 8)
            .collect();
        let plain_words = (0..stride_words)
            .filter(|w| !view_words.iter().any(|v| v + 1 == *w))
            .collect();
        Some(Self {
            cols,
            null_offset,
            null_width,
            stride_words,
            view_words,
            plain_words,
        })
    }

    pub(super) fn has_views(&self) -> bool {
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
    /// Both rows must have `stride_words` words.
    #[inline(always)]
    pub(super) unsafe fn rows_eq(
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
    /// `i` must be in-bounds and `row` must have `stride_words` words.
    #[inline(always)]
    pub(super) unsafe fn eq_columns<'a>(
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
    /// `i` must be in-bounds and `row` must have `stride_words` words.
    #[inline(always)]
    pub(super) unsafe fn write_columns(
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

    /// Clears `ok[r]` when key `key_idxs[r]` of `cols` differs from the stored row at
    /// `rows[r]`, whose long views are resolved by `stored_long`.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    pub(super) unsafe fn verify_columns<'a>(
        &self,
        cols: &[KeyColumn],
        key_idxs: &[IdxSize],
        rows: &[*const u64],
        ok: &mut [bool],
        stored_long: impl Fn(View) -> &'a [u8],
    ) {
        if cols.iter().all(|c| c.validity.is_none()) {
            for (ok, row) in ok.iter_mut().zip(rows) {
                *ok &= self.read_nulls(row.cast()) == 0;
            }
        } else {
            for ((ok, i), row) in ok.iter_mut().zip(key_idxs).zip(rows) {
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
                ColValues::Bool(b) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    b.get_bit_unchecked(i) as u8 == *p.add(off)
                }),
                ColValues::W1(v) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == *p.add(off)
                }),
                ColValues::W2(v) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u16>().read_unaligned()
                }),
                ColValues::W4(v) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u32>().read_unaligned()
                }),
                ColValues::W8(v) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u64>().read_unaligned()
                }),
                ColValues::W16(v) => verify_col(key_idxs, rows, ok, validity, |i, p| {
                    *v.get_unchecked(i) == p.add(off).cast::<u128>().read_unaligned()
                }),
                ColValues::View(views, buffers) => {
                    verify_col(key_idxs, rows, ok, validity, |i, p| {
                        let b = p.add(off).cast::<u128>().read_unaligned();
                        view_eq(*views.get_unchecked(i), b, buffers, &stored_long)
                    })
                },
            }
        }
    }

    /// Writes the keys `key_idxs[r]` of `cols` into the zeroed rows at `rows[r]`, storing
    /// the bytes of long views with `store_long`, which returns the buffer index and
    /// offset of the copy.
    ///
    /// # Safety
    /// The indices must be in-bounds and the pointers must point to rows of this
    /// layout.
    pub(super) unsafe fn write_columns_batch(
        &self,
        cols: &[KeyColumn],
        key_idxs: &[IdxSize],
        rows: &[*mut u64],
        mut store_long: impl FnMut(&[u8]) -> (u32, u32),
    ) {
        for col in cols {
            let off = col.offset;
            let validity = col.validity.as_ref();
            match &col.values {
                ColValues::Bool(b) => write_col(key_idxs, rows, validity, |i, p| {
                    *p.add(off) = b.get_bit_unchecked(i) as u8
                }),
                ColValues::W1(v) => write_col(key_idxs, rows, validity, |i, p| {
                    *p.add(off) = *v.get_unchecked(i)
                }),
                ColValues::W2(v) => write_col(key_idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u16>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W4(v) => write_col(key_idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u32>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W8(v) => write_col(key_idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u64>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::W16(v) => write_col(key_idxs, rows, validity, |i, p| {
                    p.add(off)
                        .cast::<u128>()
                        .write_unaligned(*v.get_unchecked(i))
                }),
                ColValues::View(views, buffers) => write_col(key_idxs, rows, validity, |i, p| {
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
            for (i, row) in key_idxs.iter().zip(rows) {
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
    /// The row must have `stride_words` words and `long_bytes` must return the bytes of
    /// its long views.
    #[inline(always)]
    pub(super) unsafe fn repoint_long_views<'a>(
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
    pub(super) fn decode(
        &self,
        schema: &Schema,
        entries: &[u64],
        entry_words: usize,
        skip: usize,
        num_rows: usize,
        buffers: &Buffer<Buffer<u8>>,
    ) -> DataFrame {
        assert!(skip + self.stride_words <= entry_words && num_rows * entry_words <= entries.len());
        if num_rows == 0 {
            return DataFrame::empty_with_schema(schema);
        }
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
                stored_long(View::from(b)),
            )
    }
}

#[inline(always)]
pub(super) fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
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
    View::from(lo as u128 | ((hi as u128) << 64))
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
pub(super) unsafe fn long_slice(bytes: &[u8], view: View) -> &[u8] {
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

/// Clears `ok[r]` when `eq(key_idxs[r], rows[r])` fails for a valid key.
///
/// # Safety
/// The indices must be in-bounds for `validity`.
#[inline(always)]
unsafe fn verify_col(
    key_idxs: &[IdxSize],
    rows: &[*const u64],
    ok: &mut [bool],
    validity: Option<&Bitmap>,
    eq: impl Fn(usize, *const u8) -> bool,
) {
    match validity {
        None => {
            for ((ok, i), row) in ok.iter_mut().zip(key_idxs).zip(rows) {
                *ok &= eq(*i as usize, row.cast());
            }
        },
        Some(v) => {
            for ((ok, i), row) in ok.iter_mut().zip(key_idxs).zip(rows) {
                if v.get_bit_unchecked(*i as usize) {
                    *ok &= eq(*i as usize, row.cast());
                }
            }
        },
    }
}

/// Calls `write(key_idxs[r], rows[r])` for each valid key.
///
/// # Safety
/// The indices must be in-bounds for `validity`.
#[inline(always)]
unsafe fn write_col(
    key_idxs: &[IdxSize],
    rows: &[*mut u64],
    validity: Option<&Bitmap>,
    mut write: impl FnMut(usize, *mut u8),
) {
    match validity {
        None => {
            for (i, row) in key_idxs.iter().zip(rows) {
                write(*i as usize, row.cast());
            }
        },
        Some(v) => {
            for (i, row) in key_idxs.iter().zip(rows) {
                if v.get_bit_unchecked(*i as usize) {
                    write(*i as usize, row.cast());
                }
            }
        },
    }
}
