use polars_utils::IdxSize;

use super::LargeUnionArray;
use crate::array::Array;
use crate::array::builder::{ArrayBuilder, ShareStrategy, StaticArrayBuilder, make_builder};
use crate::bitmap::OptBitmapBuilder;
use crate::datatypes::ArrowDataType;

/// Positions in one union child, collected to be appended to a builder of that child's dtype.
///
/// Children shorter than `IdxSize::MAX` are gathered by index; longer ones replay runs.
pub enum ChildPositions {
    /// `None` is stored as `IdxSize::MAX`, which is out of bounds for a narrow child.
    Idxs { idxs: Vec<IdxSize>, has_none: bool },
    /// `(start, len)` runs; a `start` of `usize::MAX` is a run of nulls.
    Runs(Vec<(usize, usize)>),
}

impl ChildPositions {
    pub fn new(child_len: usize) -> Self {
        if child_len < IdxSize::MAX as usize {
            Self::Idxs {
                idxs: Vec::new(),
                has_none: false,
            }
        } else {
            Self::Runs(Vec::new())
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Idxs { idxs, .. } => idxs.is_empty(),
            Self::Runs(v) => v.is_empty(),
        }
    }

    /// Queues `pos`, or a null for `None`.
    #[inline]
    pub fn push(&mut self, pos: Option<usize>) {
        match self {
            Self::Idxs { idxs, has_none } => match pos {
                Some(pos) => idxs.push(pos as IdxSize),
                None => {
                    idxs.push(IdxSize::MAX);
                    *has_none = true;
                },
            },
            Self::Runs(v) => {
                let start = pos.unwrap_or(usize::MAX);
                match v.last_mut() {
                    Some((s, len)) if *s == usize::MAX && start == usize::MAX => *len += 1,
                    Some((s, len)) if *s != usize::MAX && *s + *len == start => *len += 1,
                    _ => v.push((start, 1)),
                }
            },
        }
    }

    /// Appends the queued positions of `child` to `builder` and clears the queue.
    ///
    /// # Safety
    /// Every queued position must be in bounds of `child`, and `child` must have the length this
    /// buffer was created with.
    pub unsafe fn flush(
        &mut self,
        builder: &mut dyn ArrayBuilder,
        child: &dyn Array,
        share: ShareStrategy,
    ) {
        match self {
            Self::Idxs { idxs, has_none } => {
                if *has_none {
                    builder.opt_gather_extend(child, idxs, share);
                } else {
                    unsafe { builder.gather_extend(child, idxs, share) };
                }
                idxs.clear();
                *has_none = false;
            },
            Self::Runs(v) => {
                for (start, len) in v.drain(..) {
                    if start == usize::MAX {
                        builder.extend_nulls(len);
                    } else {
                        builder.subslice_extend(child, start, len, share);
                    }
                }
            },
        }
    }
}

pub struct LargeUnionArrayBuilder {
    dtype: ArrowDataType,
    types: Vec<i8>,
    offsets: Vec<i64>,
    children: Vec<Box<dyn ArrayBuilder>>,
    validity: OptBitmapBuilder,
}

impl LargeUnionArrayBuilder {
    pub fn new(dtype: ArrowDataType) -> Self {
        let children = LargeUnionArray::get_fields(&dtype)
            .iter()
            .map(|f| make_builder(f.dtype()))
            .collect();
        Self {
            dtype,
            types: Vec::new(),
            offsets: Vec::new(),
            children,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Appends the given rows of `other`; `None` appends a null. Works for children of any
    /// length.
    ///
    /// # Safety
    /// Every `Some(row)` must be in bounds of `other`.
    pub unsafe fn extend_rows<I: Iterator<Item = Option<usize>>>(
        &mut self,
        other: &LargeUnionArray,
        rows: I,
        share: ShareStrategy,
    ) {
        let (lower, _) = rows.size_hint();
        StaticArrayBuilder::reserve(self, lower);

        let mut pending: Vec<ChildPositions> = other
            .fields()
            .iter()
            .map(|c| ChildPositions::new(c.len()))
            .collect();
        let mut next: Vec<i64> = self.children.iter().map(|b| b.len() as i64).collect();
        let validity = other.validity();

        for row in rows {
            let row = row.filter(|&i| validity.is_none_or(|v| unsafe { v.get_bit_unchecked(i) }));
            // Null rows get a null placeholder in child 0.
            let (t, pos) = match row {
                Some(i) => {
                    let (t, pos) = unsafe { other.index_unchecked(i) };
                    (t, Some(pos))
                },
                None => (0, None),
            };
            pending[t].push(pos);
            self.validity.extend_constant(1, pos.is_some());
            self.types.push(t as i8);
            self.offsets.push(next[t]);
            next[t] += 1;
        }

        for ((p, b), c) in pending
            .iter_mut()
            .zip(&mut self.children)
            .zip(other.fields())
        {
            unsafe { p.flush(&mut **b, &**c, share) };
        }
        debug_assert!(
            self.children
                .iter()
                .zip(&next)
                .all(|(b, n)| b.len() as i64 == *n)
        );
    }
}

impl StaticArrayBuilder for LargeUnionArrayBuilder {
    type Array = LargeUnionArray;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.types.reserve(additional);
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(self) -> LargeUnionArray {
        let fields = self.children.into_iter().map(|b| b.freeze()).collect();
        unsafe {
            LargeUnionArray::new_unchecked(
                self.dtype,
                self.types.into(),
                self.offsets.into(),
                fields,
                self.validity.into_opt_validity(),
            )
        }
    }

    fn freeze_reset(&mut self) -> LargeUnionArray {
        let fields = self.children.iter_mut().map(|b| b.freeze_reset()).collect();
        unsafe {
            LargeUnionArray::new_unchecked(
                self.dtype.clone(),
                core::mem::take(&mut self.types).into(),
                core::mem::take(&mut self.offsets).into(),
                fields,
                core::mem::take(&mut self.validity).into_opt_validity(),
            )
        }
    }

    fn len(&self) -> usize {
        self.types.len()
    }

    fn extend_nulls(&mut self, length: usize) {
        let start = self.children[0].len() as i64;
        self.types.extend(std::iter::repeat_n(0, length));
        self.offsets.extend(start..start + length as i64);
        self.children[0].extend_nulls(length);
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &LargeUnionArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let length = length.min(other.len().saturating_sub(start));
        let types = &other.types()[start..start + length];
        let offsets = &other.offsets()[start..start + length];

        // Copy each child's referenced range as one subslice if no position in it is skipped.
        let n = other.fields().len();
        let mut first = vec![i64::MAX; n];
        let mut last = vec![-1i64; n];
        let mut count = vec![0i64; n];
        for (&t, &o) in types.iter().zip(offsets) {
            let t = t as usize;
            first[t] = first[t].min(o);
            last[t] = o;
            count[t] += 1;
        }
        let contiguous = (0..n).all(|t| count[t] == 0 || last[t] - first[t] + 1 == count[t]);
        if !contiguous {
            unsafe { self.extend_rows(other, (start..start + length).map(Some), share) };
            return;
        }

        let base: Vec<i64> = (0..n)
            .map(|t| self.children[t].len() as i64 - first[t].min(last[t]).max(0))
            .collect();
        self.types.extend_from_slice(types);
        self.offsets.extend(
            types
                .iter()
                .zip(offsets)
                .map(|(&t, &o)| o + base[t as usize]),
        );
        for t in 0..n {
            if count[t] > 0 {
                self.children[t].subslice_extend(
                    &*other.fields()[t],
                    first[t] as usize,
                    count[t] as usize,
                    share,
                );
            }
        }
        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &LargeUnionArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let length = length.min(other.len().saturating_sub(start));
        let rows = (start..start + length).flat_map(|i| std::iter::repeat_n(Some(i), repeats));
        unsafe { self.extend_rows(other, rows, share) };
    }

    unsafe fn gather_extend(
        &mut self,
        other: &LargeUnionArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let rows = idxs.iter().map(|i| Some(*i as usize));
        unsafe { self.extend_rows(other, rows, share) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &LargeUnionArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let len = other.len();
        let rows = idxs.iter().map(|i| Some(*i as usize).filter(|i| *i < len));
        unsafe { self.extend_rows(other, rows, share) };
    }
}
