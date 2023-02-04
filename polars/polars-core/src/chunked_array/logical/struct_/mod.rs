mod from;

use std::collections::BTreeMap;

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
}

fn arrays_to_fields(field_arrays: &[ArrayRef], fields: &[Series]) -> Vec<ArrowField> {
    field_arrays
        .iter()
        .zip(fields)
        .map(|(arr, s)| ArrowField::new(s.name(), arr.data_type().clone(), true))
        .collect()
}

fn fields_to_struct_array(fields: &[Series]) -> (ArrayRef, Vec<Series>) {
    let fields = fields.iter().map(|s| s.rechunk()).collect::<Vec<_>>();

    let field_arrays = fields
        .iter()
        .map(|s| s.rechunk().to_arrow(0))
        .collect::<Vec<_>>();
    // we determine fields from arrays as there might be object arrays
    // where the dtype is bound to that single array
    let new_fields = arrays_to_fields(&field_arrays, &fields);
    let arr = StructArray::new(ArrowDataType::Struct(new_fields), field_arrays, None);
    (Box::new(arr), fields)
}

impl StructChunked {
    pub fn new(name: &str, fields: &[Series]) -> PolarsResult<Self> {
        let mut names = PlHashSet::with_capacity(fields.len());
        let first_len = fields.get(0).map(|s| s.len()).unwrap_or(0);
        let mut max_len = first_len;

        let mut all_equal_len = true;
        for s in fields {
            let s_len = s.len();
            max_len = std::cmp::max(max_len, s_len);

            if s_len != first_len {
                all_equal_len = false;
            }
            let name = s.name();
            if !names.insert(name) {
                return Err(PolarsError::Duplicate(
                    format!("multiple fields with name '{name}' found").into(),
                ));
            }
        }

        if !all_equal_len {
            let mut new_fields = Vec::with_capacity(fields.len());
            for s in fields {
                let s_len = s.len();
                if s_len == max_len {
                    new_fields.push(s.clone())
                } else if s_len == 1 {
                    new_fields.push(s.new_from_index(0, max_len))
                } else {
                    return Err(PolarsError::ShapeMisMatch(
                        "expected all fields to have equal length".into(),
                    ));
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
                .map(|s| s.to_arrow(i))
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
        let (arrow_array, fields) = fields_to_struct_array(fields);

        Self {
            fields,
            field,
            chunks: vec![arrow_array],
        }
    }

    /// Get access to one of this `[StructChunked]`'s fields
    pub fn field_by_name(&self, name: &str) -> PolarsResult<Series> {
        self.fields
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| PolarsError::StructFieldNotFound(name.to_string().into()))
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

    pub fn name(&self) -> &String {
        self.field.name()
    }

    pub fn fields(&self) -> &[Series] {
        &self.fields
    }

    pub fn fields_mut(&mut self) -> &mut Vec<Series> {
        &mut self.fields
    }

    pub fn rename(&mut self, name: &str) {
        self.field.set_name(name.to_string())
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
}

impl LogicalType for StructChunked {
    fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Gets AnyValue from LogicalType
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        if i >= self.len() {
            Err(PolarsError::ComputeError("Index out of bounds.".into()))
        } else {
            unsafe { Ok(self.get_any_value_unchecked(i)) }
        }
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
        match dtype {
            DataType::Struct(dtype_fields) => {
                let map = BTreeMap::from_iter(self.fields().iter().map(|s| (s.name(), s)));
                let struct_len = self.len();
                let new_fields = dtype_fields
                    .iter()
                    .map(|new_field| match map.get(new_field.name().as_str()) {
                        Some(s) => s.cast(&new_field.dtype),
                        None => Ok(Series::full_null(
                            new_field.name(),
                            struct_len,
                            &new_field.dtype,
                        )),
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                StructChunked::new(self.name(), &new_fields).map(|ca| ca.into_series())
            }
            _ => {
                let fields = self
                    .fields
                    .iter()
                    .map(|s| s.cast(dtype))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(Self::new_unchecked(self.field.name(), &fields).into_series())
            }
        }
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
