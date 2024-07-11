mod frame;

use arrow::array::StructArray;
use arrow::legacy::utils::CustomIterTools;
use polars_error::{polars_ensure, PolarsResult};
use polars_utils::aliases::PlHashMap;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::ChunkedArray;
use crate::prelude::*;
use crate::series::Series;
use crate::utils::{Container, index_to_chunked_index};
use std::fmt::Write;
use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use crate::prelude::sort::arg_sort_multiple::{_get_rows_encoded_arr, _get_rows_encoded_ca};

pub type StructChunked2 = ChunkedArray<StructType>;

fn constructor(name: &str, fields: &[Series]) -> PolarsResult<StructChunked2> {
    // Different chunk lengths: rechunk and recurse.
    if !fields.iter().map(|s| s.n_chunks()).all_equal() {
        let fields = fields.iter().map(|s| s.rechunk()).collect::<Vec<_>>();
        return constructor(name, &fields)
    }

    let n_chunks = fields[0].n_chunks();
    let dtype = DataType::Struct(fields.iter().map(|s| s.field().into_owned()).collect());
    let arrow_dtype = dtype.to_physical().to_arrow(CompatLevel::newest());

    let chunks = (0..n_chunks).map(|c_i| {
        let fields = fields.iter().map(|field| {
            field.chunks()[c_i].clone()
        }).collect::<Vec<_>>();

        if !fields.iter().map(|arr| arr.len()).all_equal() {
            return Err(())
        }

        Ok(StructArray::new(arrow_dtype.clone(), fields, None).boxed())

    }).collect::<Result<Vec<_>, ()>>();

    match chunks {
        Ok(chunks) => {
            // SAFETY: invariants checked above.
            unsafe {
                Ok(StructChunked2::from_chunks_and_dtype_unchecked(name, chunks, dtype))
            }
        },
        // Different chunk lengths: rechunk and recurse.
        Err(_) => {
            let fields = fields.iter().map(|s| s.rechunk()).collect::<Vec<_>>();
            constructor(name, &fields)
        }
    }
}

impl StructChunked2 {
    pub fn from_series(name: &str, fields: &[Series]) -> PolarsResult<Self> {
        let mut names = PlHashSet::with_capacity(fields.len());
        let first_len = fields.first().map(|s| s.len()).unwrap_or(0);
        let mut max_len = first_len;

        let mut all_equal_len = true;
        let mut is_empty = false;
        for s in fields {
            let s_len = s.len();
            max_len = std::cmp::max(max_len, s_len);

            if s_len != first_len {
                all_equal_len = false;
            }
            if s_len == 0 {
                is_empty = true;
            }
            polars_ensure!(
                names.insert(s.name()),
                Duplicate: "multiple fields with name '{}' found", s.name()
            );
            match s.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_, _) => polars_bail!(InvalidOperation: "nested objects are not allowed"),
                _ => {}
            }
        }

        if !all_equal_len {
            let mut new_fields = Vec::with_capacity(fields.len());
            for s in fields {
                let s_len = s.len();
                if is_empty {
                    new_fields.push(s.clear())
                } else if s_len == max_len {
                    new_fields.push(s.clone())
                } else if s_len == 1 {
                    new_fields.push(s.new_from_index(0, max_len))
                } else {
                    polars_bail!(
                        ShapeMismatch: "expected all fields to have equal length"
                    );
                }
            }
            constructor(name, &new_fields)
        } else if fields.is_empty() {
            let fields = &[Series::new_null("", 0)];
            constructor(name, fields)
        } else {
            constructor(name, fields)
        }
    }

    pub fn struct_fields(&self) -> &[Field] {
        let DataType::Struct(fields) = self.dtype() else {unreachable!()};
        fields
    }

    pub fn fields_as_series(&self) -> Vec<Series> {
        dbg!(self.struct_fields());
        self.struct_fields().iter().enumerate().map(|(i, field)| {
            let field_chunks = self.downcast_iter().map(|chunk| {
                chunk.values()[i].clone()
            }).collect::<Vec<_>>();

            // SAFETY: correct type.
            unsafe { Series::from_chunks_and_dtype_unchecked(&field.name, field_chunks, &field.dtype) }
        }).collect()
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
                   .map(|new_field| match map.get(new_field.name().as_str()) {
                       Some(s) => {
                           if unchecked {
                               s.cast_unchecked(&new_field.dtype)
                           } else {
                               s.cast_with_options(&new_field.dtype, cast_options)
                           }
                       },
                       None => Ok(Series::full_null(
                           new_field.name(),
                           struct_len,
                           &new_field.dtype,
                       )),
                   })
                   .collect::<PolarsResult<Vec<_>>>()?;

               Self::from_series(self.name(), &new_fields).map(|ca| ca.into_series())
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
               Series::try_from((ca.name(), array))
           }
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
               Self::from_series(self.name(), &fields).map(|ca| ca.into_series())
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

    /// Gets AnyValue from LogicalType
    pub(crate) fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        unsafe { Ok(self.get_any_value_unchecked(i)) }
    }

    pub(crate) unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        let (chunk_idx, idx) = index_to_chunked_index(self.chunks.iter().map(|c| c.len()), i);
        if let DataType::Struct(flds) = self.dtype() {
            // SAFETY: we already have a single chunk and we are
            // guarded by the type system.
            unsafe {
                let arr = &**self.chunks.get_unchecked(chunk_idx);
                let arr = &*(arr as *const dyn Array as *const StructArray);
                AnyValue::Struct(idx, arr, flds)
            }
        } else {
            unreachable!()
        }
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
        let fields = self.fields_as_series().iter().map(func).collect::<PolarsResult<Vec<_>>>()?;
        Self::from_series(self.name(), &fields).map(|mut ca| {
            if self.null_count > 0 {
                // SAFETY: we don't change types/ lenghts.
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
        let s = self.clone().into_series();
        _get_rows_encoded_arr(&[s], &[options.descending], &[options.nulls_last])
    }

    pub fn get_row_encoded(&self, options: SortOptions) -> PolarsResult<BinaryOffsetChunked> {
        let s = self.clone().into_series();
        _get_rows_encoded_ca(self.name(), &[s], &[options.descending], &[options.nulls_last])
    }

    /// Set the outer nulls into the inner arrays, and clear the outer validity.
    pub(crate) fn propagate_nulls(&mut self) {
        // SAFETY:
        // We keep length and dtypes the same.
        unsafe {
            for arr in self.downcast_iter_mut() {
                *arr = arr.propagate_nulls()
            }
        }
    }

    /// Combine the validities of two structs.
    /// # Panics
    /// Panics if the chunks don't align.
    pub fn zip_outer_validity(&mut self, other: &StructChunked2) {
        if other.null_count > 0 {
            // SAFETY:
            // We keep length and dtypes the same.
            unsafe {
                for (a, b ) in self.downcast_iter_mut().zip(other.downcast_iter()) {
                    let new = combine_validities_and(a.validity(), b.validity());
                    a.set_validity(new)
                }
            }
        }
        self.compute_len();
    }

    pub(crate) fn set_outer_validity(&mut self, validity: Option<Bitmap>) {
        assert_eq!(self.chunks().len(), 1);
        unsafe {
            let arr = self.downcast_iter_mut().next().unwrap();
            arr.set_validity(validity)
        }
        self.compute_len();
    }

    pub fn unnest(mut self) -> DataFrame {
        self.propagate_nulls();

        // SAFETY: invariants for struct are the same
        unsafe { DataFrame::new_no_checks(self.fields_as_series()) }

    }

    /// Get access to one of this `[StructChunked]`'s fields
    pub fn field_by_name(&self, name: &str) -> PolarsResult<Series> {
        self.fields_as_series()
            .into_iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| polars_err!(StructFieldNotFound: "{}", name))
    }
}
