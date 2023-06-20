mod from;

use std::collections::BTreeMap;
use std::io::Write;
use std::ops::BitAnd;

use arrow::bitmap::MutableBitmap;
use arrow::offset::OffsetsBuffer;
use polars_arrow::trusted_len::TrustedLenPush;
use smartstring::alias::String as SmartString;

use super::*;
use crate::datatypes::*;
use crate::utils::index_to_chunked_index2;

/// This is logical type [`StructChunked`] that
/// dispatches most logic to the `fields` implementations
///
/// Different from  [`StructArray`](arrow::array::StructArray), this
/// type does not have its own `validity`. That means some operations
/// will be a bit less efficient because we need to check validity of all
/// fields. However this does save a lot of code and compile times.
#[derive(Clone)]
pub struct StructChunked {
    fields: Vec<Series>,
    field: Field,
    chunks: Vec<ArrayRef>,
    null_count: usize,
}

fn arrays_to_fields(field_arrays: &[ArrayRef], fields: &[Series]) -> Vec<ArrowField> {
    field_arrays
        .iter()
        .zip(fields)
        .map(|(arr, s)| ArrowField::new(s.name(), arr.data_type().clone(), true))
        .collect()
}

fn fields_to_struct_array(fields: &[Series], physical: bool) -> (ArrayRef, Vec<Series>) {
    let fields = fields.iter().map(|s| s.rechunk()).collect::<Vec<_>>();

    let field_arrays = fields
        .iter()
        .map(|s| {
            let s = s.rechunk();
            match s.dtype() {
                #[cfg(feature = "object")]
                DataType::Object(_) => s.to_arrow(0),
                _ => {
                    if physical {
                        s.chunks()[0].clone()
                    } else {
                        s.to_arrow(0)
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    // we determine fields from arrays as there might be object arrays
    // where the dtype is bound to that single array
    let new_fields = arrays_to_fields(&field_arrays, &fields);
    let arr = StructArray::new(ArrowDataType::Struct(new_fields), field_arrays, None);
    (Box::new(arr), fields)
}

impl StructChunked {
    pub fn null_count(&self) -> usize {
        self.null_count
    }
    pub fn new(name: &str, fields: &[Series]) -> PolarsResult<Self> {
        let mut names = PlHashSet::with_capacity(fields.len());
        let first_len = fields.get(0).map(|s| s.len()).unwrap_or(0);
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
            Ok(Self::new_unchecked(name, &new_fields))
        } else if fields.is_empty() {
            let fields = &[Series::full_null("", 1, &DataType::Null)];
            Ok(Self::new_unchecked(name, fields))
        } else {
            Ok(Self::new_unchecked(name, fields))
        }
    }

    pub(crate) fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    pub fn rechunk(&mut self) {
        self.fields = self.fields.iter().map(|s| s.rechunk()).collect();
        self.update_chunks(0);
    }

    // Should be called after append or extend
    pub(crate) fn update_chunks(&mut self, offset: usize) {
        let n_chunks = self.fields[0].chunks().len();
        for i in offset..n_chunks {
            let field_arrays = self
                .fields
                .iter()
                .map(|s| match s.dtype() {
                    #[cfg(feature = "object")]
                    DataType::Object(_) => s.to_arrow(i),
                    _ => s.chunks()[i].clone(),
                })
                .collect::<Vec<_>>();

            // we determine fields from arrays as there might be object arrays
            // where the dtype is bound to that single array
            let new_fields = arrays_to_fields(&field_arrays, &self.fields);
            let arr = Box::new(StructArray::new(
                ArrowDataType::Struct(new_fields),
                field_arrays,
                None,
            )) as ArrayRef;
            match self.chunks.get_mut(i) {
                Some(a) => *a = arr,
                None => {
                    self.chunks.push(arr);
                }
            }
        }
        self.chunks.truncate(n_chunks);
        self.set_null_count()
    }

    /// Does not check the lengths of the fields
    pub(crate) fn new_unchecked(name: &str, fields: &[Series]) -> Self {
        let dtype = DataType::Struct(
            fields
                .iter()
                .map(|s| Field::new(s.name(), s.dtype().clone()))
                .collect(),
        );
        let field = Field::new(name, dtype);
        let (arrow_array, fields) = fields_to_struct_array(fields, true);

        let mut out = Self {
            fields,
            field,
            chunks: vec![arrow_array],
            null_count: 0,
        };
        out.set_null_count();
        out
    }

    fn set_null_count(&mut self) {
        let mut null_count = 0;
        let chunks_lens = self.fields()[0].chunks().len();

        // fast path
        // we early return if a column doesn't have nulls
        for i in 0..chunks_lens {
            for s in self.fields() {
                let arr = &s.chunks()[i];
                let has_nulls = arr.null_count() > 0 || matches!(s.dtype(), DataType::Null);
                if !has_nulls {
                    self.null_count = 0;
                    return;
                }
            }
        }

        // slow path
        // we bitand every null validity bitmask to determine
        // in which rows all values are null
        for i in 0..chunks_lens {
            let mut validity_agg = None;

            let mut all_null_array = true;
            for s in self.fields() {
                let arr = &s.chunks()[i];

                if !matches!(s.dtype(), DataType::Null) {
                    all_null_array = false;
                    match (&validity_agg, arr.validity()) {
                        (Some(agg), Some(validity)) => validity_agg = Some(validity.bitand(agg)),
                        (None, Some(validity)) => validity_agg = Some(validity.clone()),
                        _ => {}
                    }
                }
            }
            // we add the null count
            if let Some(validity) = &validity_agg {
                null_count += validity.unset_bits()
            }
            // all arrays are null arrays
            // we add the length of the chunk to the null_count
            else if all_null_array {
                null_count += self.fields()[0].chunks()[i].len()
            }
        }
        self.null_count = null_count
    }

    /// Get access to one of this `[StructChunked]`'s fields
    pub fn field_by_name(&self, name: &str) -> PolarsResult<Series> {
        self.fields
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| polars_err!(StructFieldNotFound: "{}", name))
            .map(|s| s.clone())
    }

    pub fn len(&self) -> usize {
        self.fields.get(0).map(|s| s.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the [`Field`] of array.
    pub fn ref_field(&self) -> &Field {
        &self.field
    }

    pub fn name(&self) -> &SmartString {
        self.field.name()
    }

    pub fn fields(&self) -> &[Series] {
        &self.fields
    }

    pub fn fields_mut(&mut self) -> &mut Vec<Series> {
        &mut self.fields
    }

    pub fn rename(&mut self, name: &str) {
        self.field.set_name(name.into())
    }

    pub(crate) fn try_apply_fields<F>(&self, func: F) -> PolarsResult<Self>
    where
        F: Fn(&Series) -> PolarsResult<Series>,
    {
        let fields = self
            .fields
            .iter()
            .map(func)
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(Self::new_unchecked(self.field.name(), &fields))
    }

    pub(crate) fn apply_fields<F>(&self, func: F) -> Self
    where
        F: FnMut(&Series) -> Series,
    {
        let fields = self.fields.iter().map(func).collect::<Vec<_>>();
        Self::new_unchecked(self.field.name(), &fields)
    }
    pub fn unnest(self) -> DataFrame {
        self.into()
    }

    pub(crate) fn to_arrow(&self, i: usize) -> ArrayRef {
        let values = self
            .fields
            .iter()
            .map(|s| s.to_arrow(i))
            .collect::<Vec<_>>();

        // we determine fields from arrays as there might be object arrays
        // where the dtype is bound to that single array
        let new_fields = arrays_to_fields(&values, &self.fields);
        Box::new(StructArray::new(
            ArrowDataType::Struct(new_fields),
            values,
            None,
        ))
    }

    unsafe fn cast_impl(&self, dtype: &DataType, unchecked: bool) -> PolarsResult<Series> {
        match dtype {
            DataType::Struct(dtype_fields) => {
                let map = BTreeMap::from_iter(self.fields().iter().map(|s| (s.name(), s)));
                let struct_len = self.len();
                let new_fields = dtype_fields
                    .iter()
                    .map(|new_field| match map.get(new_field.name().as_str()) {
                        Some(s) => {
                            if unchecked {
                                s.cast_unchecked(&new_field.dtype)
                            } else {
                                s.cast(&new_field.dtype)
                            }
                        }
                        None => Ok(Series::full_null(
                            new_field.name(),
                            struct_len,
                            &new_field.dtype,
                        )),
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                StructChunked::new(self.name(), &new_fields).map(|ca| ca.into_series())
            }
            DataType::Utf8 => {
                let mut ca = self.clone();
                ca.rechunk();
                let mut iters = ca.fields.iter().map(|s| s.iter()).collect::<Vec<_>>();
                let mut values = Vec::with_capacity(self.len() * 8);
                let mut offsets = Vec::with_capacity(ca.len() + 1);
                let has_nulls = self.fields.iter().any(|s| s.null_count() > 0) as usize;
                let cap = ca.len() * has_nulls;
                let mut bitmap = MutableBitmap::with_capacity(cap);
                bitmap.extend_constant(cap, true);

                let mut length_so_far = 0_i64;
                unsafe {
                    // safety: we have pre-allocated
                    offsets.push_unchecked(length_so_far);
                }
                for row in 0..ca.len() {
                    let mut row_has_nulls = false;

                    write!(values, "{{").unwrap();
                    for iter in &mut iters {
                        let av = unsafe { iter.next().unwrap_unchecked() };
                        row_has_nulls |= matches!(&av, AnyValue::Null);
                        write!(values, "{},", av).unwrap();
                    }

                    // replace latest comma with '|'
                    unsafe {
                        *values.last_mut().unwrap_unchecked() = b'}';

                        // safety: we have pre-allocated
                        length_so_far = values.len() as i64;
                        offsets.push_unchecked(length_so_far);
                    }
                    if row_has_nulls {
                        unsafe { bitmap.set_unchecked(row, false) }
                    }
                }
                let validity = if has_nulls == 1 {
                    Some(bitmap.into())
                } else {
                    None
                };
                unsafe {
                    let offsets = OffsetsBuffer::new_unchecked(offsets.into());
                    let array = Box::new(Utf8Array::new_unchecked(
                        ArrowDataType::LargeUtf8,
                        offsets,
                        values.into(),
                        validity,
                    )) as ArrayRef;
                    Series::try_from((ca.name().as_str(), array))
                }
            }
            _ => {
                let fields = self
                    .fields
                    .iter()
                    .map(|s| {
                        if unchecked {
                            s.cast_unchecked(dtype)
                        } else {
                            s.cast(dtype)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(Self::new_unchecked(self.field.name(), &fields).into_series())
            }
        }
    }

    pub(crate) unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        if dtype == self.dtype() {
            return Ok(self.clone().into_series());
        }
        self.cast_impl(dtype, true)
    }
}

impl LogicalType for StructChunked {
    fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Gets AnyValue from LogicalType
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        unsafe { Ok(self.get_any_value_unchecked(i)) }
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        let (chunk_idx, idx) = index_to_chunked_index2(&self.chunks, i);
        if let DataType::Struct(flds) = self.dtype() {
            // safety: we already have a single chunk and we are
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

    // in case of a struct, a cast will coerce the inner types
    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        unsafe { self.cast_impl(dtype, false) }
    }
}

#[cfg(feature = "object")]
impl Drop for StructChunked {
    fn drop(&mut self) {
        use crate::chunked_array::object::extension::drop::drop_object_array;
        use crate::chunked_array::object::extension::EXTENSION_NAME;
        if self
            .fields
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Object(_)))
        {
            for arr in std::mem::take(&mut self.chunks) {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                for arr in arr.values() {
                    match arr.data_type() {
                        ArrowDataType::Extension(name, _, _) if name == EXTENSION_NAME => unsafe {
                            drop_object_array(arr.as_ref())
                        },
                        _ => {}
                    }
                }
            }
        }
    }
}
