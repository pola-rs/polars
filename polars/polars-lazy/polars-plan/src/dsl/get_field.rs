//! Utilitary functions for defining function output fields.
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;

/// set a dtype
pub fn with_dtype(
    dtype: DataType,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'static {
    move |fields: &[Field]| Ok(Field::new(fields[0].name(), dtype.clone()))
}

/// map a single dtype
pub fn map_dtype<'a>(
    func: impl Fn(&DataType) -> DataType + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| {
        let dtype = func(fields[0].data_type());
        Ok(Field::new(fields[0].name(), dtype))
    }
}

/// map dtypes
pub fn map_dtypes<'a>(
    func: impl Fn(&[&DataType]) -> DataType + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| {
        let mut fld = fields[0].clone();
        let dtypes = fields.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
        let new_type = func(&dtypes);
        fld.coerce(new_type);
        Ok(fld)
    }
}

/// map a single field
pub fn map_field<'a>(
    func: impl Fn(&Field) -> Field + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| Ok(func(&fields[0]))
}

/// map fields
pub fn map_fields<'a>(
    func: impl Fn(&[Field]) -> Field + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| Ok(func(fields))
}

/// map a single dtype
pub fn try_map_dtype<'a>(
    func: impl Fn(&DataType) -> PolarsResult<DataType> + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| {
        let dtype = func(fields[0].data_type())?;
        let out: PolarsResult<_> = Ok(Field::new(fields[0].name(), dtype));
        out
    }
}

/// map all dtypes
pub fn try_map_dtypes<'a>(
    func: impl Fn(&[&DataType]) -> PolarsResult<DataType> + Send + Sync + 'a,
) -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'a {
    move |fields: &[Field]| {
        let mut fld = fields[0].clone();
        let dtypes = fields.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
        let new_type = func(&dtypes)?;
        fld.coerce(new_type);
        Ok(fld)
    }
}

/// map to same type
pub fn same_type() -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'static {
    map_dtype(|dtype| dtype.clone())
}

/// get supertype of all types
pub fn super_type() -> impl Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'static {
    move |fields: &[Field]| {
        let mut first = fields[0].clone();
        let mut st = first.data_type().clone();
        for field in &fields[1..] {
            st = try_get_supertype(&st, field.data_type())?
        }
        first.coerce(st);
        Ok(first)
    }
}
