mod frame;

use std::borrow::Cow;
use std::fmt::Write;

use arrow::array::StructArray;
use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use polars_error::{polars_ensure, PolarsResult};
use polars_utils::aliases::PlHashMap;
use polars_utils::itertools::Itertools;

use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::ops::row_encode::{_get_rows_encoded_arr, _get_rows_encoded_ca};
use crate::chunked_array::ChunkedArray;
use crate::prelude::*;
use crate::series::Series;
use crate::utils::Container;

pub type StructChunked = ChunkedArray<StructType>;

fn constructor<'a, I: ExactSizeIterator<Item = &'a Series> + Clone>(
    name: PlSmallStr,
    length: usize,
    fields: I,
) -> StructChunked {
    if fields.len() == 0 {
        let dtype = DataType::Struct(Vec::new());
        let arrow_dtype = dtype.to_physical().to_arrow(CompatLevel::newest());
        let chunks = vec![StructArray::new(arrow_dtype, length, Vec::new(), None).boxed()];

        // SAFETY: We construct each chunk above to have the `Struct` data type.
        return unsafe { StructChunked::from_chunks_and_dtype(name, chunks, dtype) };
    }

    // Different chunk lengths: rechunk and recurse.
    if !fields.clone().map(|s| s.n_chunks()).all_equal() {
        let fields = fields.map(|s| s.rechunk()).collect::<Vec<_>>();
        return constructor(name, length, fields.iter());
    }

    let n_chunks = fields.clone().next().unwrap().n_chunks();
    let dtype = DataType::Struct(fields.clone().map(|s| s.field().into_owned()).collect());
    let arrow_dtype = dtype.to_physical().to_arrow(CompatLevel::newest());

    let chunks = (0..n_chunks)
        .map(|c_i| {
            let fields = fields
                .clone()
                .map(|field| field.chunks()[c_i].clone())
                .collect::<Vec<_>>();
            let chunk_length = fields[0].len();

            if fields[1..].iter().any(|arr| chunk_length != arr.len()) {
                return None;
            }

            Some(StructArray::new(arrow_dtype.clone(), chunk_length, fields, None).boxed())
        })
        .collect::<Option<Vec<_>>>();

    match chunks {
        Some(chunks) => {
            // SAFETY: invariants checked above.
            unsafe { StructChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) }
        },
        // Different chunks: rechunk and recurse.
        None => {
            let fields = fields.map(|s| s.rechunk()).collect::<Vec<_>>();
            constructor(name, length, fields.iter())
        },
    }
}

impl StructChunked {
    pub fn from_columns(name: PlSmallStr, length: usize, fields: &[Column]) -> PolarsResult<Self> {
        Self::from_series(
            name,
            length,
            fields.iter().map(|c| c.as_materialized_series()),
        )
    }

    pub fn from_series<'a, I: ExactSizeIterator<Item = &'a Series> + Clone>(
        name: PlSmallStr,
        length: usize,
        fields: I,
    ) -> PolarsResult<Self> {
        let mut names = PlHashSet::with_capacity(fields.len());

        let mut needs_to_broadcast = false;
        for s in fields.clone() {
            let s_len = s.len();

            if s_len != length && s_len != 1 {
                polars_bail!(
                    ShapeMismatch: "expected struct fields to have given length. given = {length}, field length = {s_len}."
                );
            }

            needs_to_broadcast |= length != 1 && s_len == 1;

            polars_ensure!(
                names.insert(s.name()),
                Duplicate: "multiple fields with name '{}' found", s.name()
            );

            match s.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_, _) => {
                    polars_bail!(InvalidOperation: "nested objects are not allowed")
                },
                _ => {},
            }
        }

        if !needs_to_broadcast {
            return Ok(constructor(name, length, fields));
        }

        if length == 0 {
            // @NOTE: There are columns that are being broadcasted so we need to clear those.
            let new_fields = fields.map(|s| s.clear()).collect::<Vec<_>>();

            return Ok(constructor(name, length, new_fields.iter()));
        }

        let new_fields = fields
            .map(|s| {
                if s.len() == length {
                    s.clone()
                } else {
                    s.new_from_index(0, length)
                }
            })
            .collect::<Vec<_>>();
        Ok(constructor(name, length, new_fields.iter()))
    }

    /// Convert a struct to the underlying physical datatype.
    pub fn to_physical_repr(&self) -> Cow<StructChunked> {
        let mut physicals = Vec::new();

        let field_series = self.fields_as_series();
        for (i, s) in field_series.iter().enumerate() {
            if let Cow::Owned(physical) = s.to_physical_repr() {
                physicals.reserve(field_series.len());
                physicals.extend(field_series[..i].iter().cloned());
                physicals.push(physical);
                break;
            }
        }

        if physicals.is_empty() {
            return Cow::Borrowed(self);
        }

        physicals.extend(
            field_series[physicals.len()..]
                .iter()
                .map(|s| s.to_physical_repr().into_owned()),
        );

        let mut ca = constructor(self.name().clone(), self.length, physicals.iter());
        if self.null_count() > 0 {
            ca.zip_outer_validity(self);
        }

        Cow::Owned(ca)
    }

    /// Convert a non-logical [`StructChunked`] back into a logical [`StructChunked`] without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(
        &self,
        to_fields: &[Field],
    ) -> PolarsResult<StructChunked> {
        if cfg!(debug_assertions) {
            for f in self.struct_fields() {
                assert!(!f.dtype().is_logical());
            }
        }

        let length = self.len();
        let fields = self
            .fields_as_series()
            .iter()
            .zip(to_fields)
            .map(|(f, to)| unsafe { f.from_physical_unchecked(to.dtype()) })
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut out = StructChunked::from_series(self.name().clone(), length, fields.iter())?;
        out.zip_outer_validity(self);
        Ok(out)
    }

    pub fn struct_fields(&self) -> &[Field] {
        let DataType::Struct(fields) = self.dtype() else {
            unreachable!()
        };
        fields
    }

    pub fn fields_as_series(&self) -> Vec<Series> {
        self.struct_fields()
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let field_chunks = self
                    .downcast_iter()
                    .map(|chunk| chunk.values()[i].clone())
                    .collect::<Vec<_>>();

                // SAFETY: correct type.
                unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        field.name.clone(),
                        field_chunks,
                        &field.dtype,
                    )
                }
            })
            .collect()
    }

    unsafe fn cast_impl(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
        unchecked: bool,
    ) -> PolarsResult<Series> {
        match dtype {
            DataType::Struct(dtype_fields) => {
                let fields = self.fields_as_series();
                let map = PlHashMap::from_iter(fields.iter().map(|s| (s.name(), s)));
                let struct_len = self.len();
                let new_fields = dtype_fields
                    .iter()
                    .map(|new_field| match map.get(new_field.name()) {
                        Some(s) => {
                            if unchecked {
                                s.cast_unchecked(&new_field.dtype)
                            } else {
                                s.cast_with_options(&new_field.dtype, cast_options)
                            }
                        },
                        None => Ok(Series::full_null(
                            new_field.name().clone(),
                            struct_len,
                            &new_field.dtype,
                        )),
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let mut out =
                    Self::from_series(self.name().clone(), struct_len, new_fields.iter())?;
                if self.null_count > 0 {
                    out.zip_outer_validity(self);
                }
                Ok(out.into_series())
            },
            DataType::String => {
                let ca = self.clone();
                ca.rechunk();

                let fields = ca.fields_as_series();
                let mut iters = fields.iter().map(|s| s.iter()).collect::<Vec<_>>();
                let cap = ca.len();

                let mut builder = MutablePlString::with_capacity(cap);
                let mut scratch = String::new();

                for _ in 0..ca.len() {
                    let mut row_has_nulls = false;

                    write!(scratch, "{{").unwrap();
                    for iter in &mut iters {
                        let av = unsafe { iter.next().unwrap_unchecked() };
                        row_has_nulls |= matches!(&av, AnyValue::Null);
                        write!(scratch, "{},", av).unwrap();
                    }

                    // replace latest comma with '|'
                    unsafe {
                        *scratch.as_bytes_mut().last_mut().unwrap_unchecked() = b'}';
                    }

                    // TODO: this seem strange to me. We should use outer mutability to determine this.
                    // Also we should move this whole cast into arrow logic.
                    if row_has_nulls {
                        builder.push_null()
                    } else {
                        builder.push_value(scratch.as_str());
                    }
                    scratch.clear();
                }
                let array = builder.freeze().boxed();
                Series::try_from((ca.name().clone(), array))
            },
            _ => {
                let fields = self
                    .fields_as_series()
                    .iter()
                    .map(|s| {
                        if unchecked {
                            s.cast_unchecked(dtype)
                        } else {
                            s.cast_with_options(dtype, cast_options)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                let mut out = Self::from_series(self.name().clone(), self.len(), fields.iter())?;
                if self.null_count > 0 {
                    out.zip_outer_validity(self);
                }
                Ok(out.into_series())
            },
        }
    }

    pub(crate) unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        if dtype == self.dtype() {
            return Ok(self.clone().into_series());
        }
        self.cast_impl(dtype, CastOptions::Overflowing, true)
    }

    // in case of a struct, a cast will coerce the inner types
    pub fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        unsafe { self.cast_impl(dtype, cast_options, false) }
    }

    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::NonStrict)
    }

    pub fn _apply_fields<F>(&self, mut func: F) -> PolarsResult<Self>
    where
        F: FnMut(&Series) -> Series,
    {
        self.try_apply_fields(|s| Ok(func(s)))
    }

    pub fn try_apply_fields<F>(&self, func: F) -> PolarsResult<Self>
    where
        F: FnMut(&Series) -> PolarsResult<Series>,
    {
        let fields = self
            .fields_as_series()
            .iter()
            .map(func)
            .collect::<PolarsResult<Vec<_>>>()?;
        Self::from_series(self.name().clone(), self.len(), fields.iter()).map(|mut ca| {
            if self.null_count > 0 {
                // SAFETY: we don't change types/ lengths.
                unsafe {
                    for (new, this) in ca.downcast_iter_mut().zip(self.downcast_iter()) {
                        new.set_validity(this.validity().cloned())
                    }
                }
            }
            ca
        })
    }

    pub fn get_row_encoded_array(&self, options: SortOptions) -> PolarsResult<BinaryArray<i64>> {
        let c = self.clone().into_column();
        _get_rows_encoded_arr(&[c], &[options.descending], &[options.nulls_last])
    }

    pub fn get_row_encoded(&self, options: SortOptions) -> PolarsResult<BinaryOffsetChunked> {
        let c = self.clone().into_column();
        _get_rows_encoded_ca(
            self.name().clone(),
            &[c],
            &[options.descending],
            &[options.nulls_last],
        )
    }

    /// Set the outer nulls into the inner arrays, and clear the outer validity.
    pub(crate) fn propagate_nulls(&mut self) {
        if self.null_count > 0 {
            // SAFETY:
            // We keep length and dtypes the same.
            unsafe {
                for arr in self.downcast_iter_mut() {
                    *arr = arr.propagate_nulls()
                }
            }
        }
    }

    /// Combine the validities of two structs.
    pub fn zip_outer_validity(&mut self, other: &StructChunked) {
        // This might go wrong for broadcasting behavior. If this is not checked, it leads to a
        // segfault because we infinitely recurse.
        assert_eq!(self.len(), other.len());

        if other.null_count() == 0 {
            return;
        }

        if self.chunks.len() != other.chunks.len()
            || self
                .chunks
                .iter()
                .zip(other.chunks())
                .any(|(a, b)| a.len() != b.len())
        {
            self.rechunk_mut();
            let other = other.rechunk();
            return self.zip_outer_validity(&other);
        }

        // SAFETY:
        // We keep length and dtypes the same.
        unsafe {
            for (a, b) in self.downcast_iter_mut().zip(other.downcast_iter()) {
                let new = combine_validities_and(a.validity(), b.validity());
                a.set_validity(new)
            }
        }

        self.compute_len();
        self.propagate_nulls();
    }

    pub fn unnest(self) -> DataFrame {
        // @scalar-opt
        let columns = self
            .fields_as_series()
            .into_iter()
            .map(Column::from)
            .collect::<Vec<_>>();

        // SAFETY: invariants for struct are the same
        unsafe { DataFrame::new_no_checks(self.len(), columns) }
    }

    /// Get access to one of this [`StructChunked`]'s fields
    pub fn field_by_name(&self, name: &str) -> PolarsResult<Series> {
        self.fields_as_series()
            .into_iter()
            .find(|s| s.name().as_str() == name)
            .ok_or_else(|| polars_err!(StructFieldNotFound: "{}", name))
    }
    pub(crate) fn set_outer_validity(&mut self, validity: Option<Bitmap>) {
        assert_eq!(self.chunks().len(), 1);
        unsafe {
            let arr = self.chunks_mut().iter_mut().next().unwrap();
            *arr = arr.with_validity(validity);
        }
        self.compute_len();
        self.propagate_nulls();
    }

    pub fn with_outer_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_outer_validity(validity);
        self
    }
}
