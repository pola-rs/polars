mod from;

use super::*;

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
    // needed by iterators
    arrow_array: ArrayRef,
}

fn fields_to_struct_array(fields: &[Series]) -> (ArrayRef, Vec<Series>) {
    let fields = fields.iter().map(|s| s.rechunk()).collect::<Vec<_>>();

    let new_fields = fields.iter().map(|s| s.field().to_arrow()).collect();
    let field_arrays = fields
        .iter()
        .map(|s| s.chunks()[0].clone() as ArrayRef)
        .collect::<Vec<_>>();
    let arr = StructArray::new(ArrowDataType::Struct(new_fields), field_arrays, None);
    (Arc::new(arr), fields)
}

impl StructChunked {
    pub fn new(name: &str, fields: &[Series]) -> Result<Self> {
        if !fields.iter().map(|s| s.len()).all_equal() {
            Err(PolarsError::ShapeMisMatch(
                "expected all fields to have equal length".into(),
            ))
        } else {
            Ok(Self::new_unchecked(name, fields))
        }
    }

    pub(crate) fn arrow_array(&self) -> &ArrayRef {
        &self.arrow_array
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
            arrow_array,
        }
    }

    /// Get access to one of this `[StructChunked]`'s fields
    pub fn field_by_name(&self, name: &str) -> Result<Series> {
        self.fields
            .iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| PolarsError::NotFound(name.to_string()))
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

    pub(crate) fn try_apply_fields<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&Series) -> Result<Series>,
    {
        let fields = self.fields.iter().map(func).collect::<Result<Vec<_>>>()?;
        Ok(Self::new_unchecked(self.field.name(), &fields))
    }

    pub(crate) fn apply_fields<F>(&self, func: F) -> Self
    where
        F: Fn(&Series) -> Series,
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
    fn get_any_value(&self, i: usize) -> AnyValue<'_> {
        AnyValue::Struct(self.fields.iter().map(|s| s.get(i)).collect())
    }

    // in case of a struct, a cast will coerce the inner types
    fn cast(&self, dtype: &DataType) -> Result<Series> {
        let fields = self
            .fields
            .iter()
            .map(|s| s.cast(dtype))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new_unchecked(self.field.name(), &fields).into_series())
    }
}
