use arrow::array::StructArray;
use arrow::legacy::utils::CustomIterTools;
use polars_error::{polars_ensure, PolarsResult};
use polars_utils::aliases::PlHashMap;
use polars_utils::index::NullCount;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::ChunkedArray;
use crate::prelude::*;
use crate::series::Series;
use crate::utils::{Container, index_to_chunked_index};
use std::fmt::Write;
use crate::prelude::sort::arg_sort_multiple::_get_rows_encoded_ca;

pub type StructChunked2 = ChunkedArray<StructType>;

impl StructChunked2 {
    pub(crate) fn from_series(name: &str, fields: &[Series]) -> PolarsResult<Self> {
        polars_ensure!(fields.iter().map(|s| s.n_chunks()).all_equal(), InvalidOperation: "expected equal chunks in struct creation");

        let n_chunks = fields[0].n_chunks();
        let dtype = DataType::Struct(fields.iter().map(|s| s.field().into_owned()).collect());
        let arrow_dtype = dtype.to_arrow(CompatLevel::newest());

        let chunks = (0..n_chunks).map(|c_i| {
            let fields = fields.iter().map(|field| {
                field.chunks()[c_i].clone()
            }).collect::<Vec<_>>();

            polars_ensure!(fields.iter().map(|arr| arr.len()).all_equal(), InvalidOperation: "expected equal chunk lengths in struct creation");

            Ok(StructArray::new(arrow_dtype.clone(), fields, None))

        });

        StructChunked2::try_from_chunk_iter(name, chunks)

    }

    pub(crate) fn struct_fields(&self) -> &[Field] {
        let DataType::Struct(fields) = self.dtype() else {unreachable!()};
        fields
    }

    pub(crate) fn fields_as_series(&self) -> Vec<Series> {
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

               let mut ca = self.clone();
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

    pub(crate) fn _apply_fields<F>(&self, func: F) -> PolarsResult<Self>
    where
        F: FnMut(&Series) -> Series,
    {
        let fields = self.fields_as_series().iter().map(func).collect::<Vec<_>>();
        Self::from_series(self.name(), &fields)
    }
}
