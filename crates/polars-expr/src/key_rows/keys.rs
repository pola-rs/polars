use std::hash::BuildHasher;
use std::sync::Arc;

use polars_arrow::array::{PrimitiveArray, UInt64Array, View};
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

use super::layout::{ColLayout, KeyRowLayout, bytes_eq};
use crate::hash_keys::{for_each_hash_prehashed, for_each_hash_subset_prehashed};

const HASH_MULTIPLE: u64 = 0x5851f42d4c957f2d;
const NULL_WORD: u64 = 0x9e3779b97f4a7c15;

#[inline(always)]
fn fold(h: u64, w: u64) -> u64 {
    folded_multiply(h ^ w, HASH_MULTIPLE)
}

/// The values of a key column, with floats canonicalized and every other type
/// reinterpreted as unsigned integers of the same width.
#[derive(Clone, Debug)]
pub(super) enum ColValues {
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
pub(super) struct KeyColumn {
    pub(super) values: ColValues,
    /// Only kept when the column has nulls.
    pub(super) validity: Option<Bitmap>,
    /// Byte offset within the row.
    pub(super) offset: usize,
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
    pub(super) layout: Arc<KeyRowLayout>,
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

    pub(super) fn from_rows(
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
    pub(super) unsafe fn eq_stored<'a>(
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
    pub(super) unsafe fn verify<'a>(
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
    pub(super) unsafe fn write_rows(
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
    pub(super) unsafe fn write_row(
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
