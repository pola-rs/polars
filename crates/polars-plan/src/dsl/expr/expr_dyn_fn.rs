use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use polars_core::utils::try_get_supertype;

use super::*;

/// A wrapper trait for any closure `Fn(Vec<Series>) -> PolarsResult<Series>`
pub trait ColumnsUdf: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any {
        unimplemented!("as_any not implemented for this 'opaque' function")
    }

    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this 'opaque' function")
    }
}

impl<F> ColumnsUdf for F
where
    F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>> {
        self(s)
    }
}

impl Debug for dyn ColumnsUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Column, Column) -> PolarsResult<Column>`
pub trait ColumnBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Column, b: Column) -> PolarsResult<Column>;
}

impl<F> ColumnBinaryUdf for F
where
    F: Fn(Column, Column) -> PolarsResult<Column> + Send + Sync,
{
    fn call_udf(&self, a: Column, b: Column) -> PolarsResult<Column> {
        self(a, b)
    }
}

impl Debug for dyn ColumnBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnBinaryUdf")
    }
}

impl Default for SpecialEq<Arc<dyn ColumnBinaryUdf>> {
    fn default() -> Self {
        panic!("implementation error");
    }
}

impl Default for SpecialEq<Arc<dyn BinaryUdfOutputField>> {
    fn default() -> Self {
        let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| None;
        SpecialEq::new(Arc::new(output_field))
    }
}

pub trait RenameAliasFn: Send + Sync {
    fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this renaming function")
    }
}

impl<F> RenameAliasFn for F
where
    F: Fn(&PlSmallStr) -> PolarsResult<PlSmallStr> + Send + Sync,
{
    fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr> {
        self(name)
    }
}

impl Debug for dyn RenameAliasFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenameAliasFn")
    }
}

#[derive(Clone)]
/// Wrapper type that has special equality properties
/// depending on the inner type specialization
pub struct SpecialEq<T>(T);

impl<T> SpecialEq<T> {
    pub fn new(val: T) -> Self {
        SpecialEq(val)
    }
}

impl<T: ?Sized> PartialEq for SpecialEq<Arc<T>> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for SpecialEq<Arc<T>> {}

impl PartialEq for SpecialEq<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Debug for SpecialEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for SpecialEq<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait BinaryUdfOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field>;
}

impl<F> BinaryUdfOutputField for F
where
    F: Fn(&Schema, Context, &Field, &Field) -> Option<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field> {
        self(input_schema, cntxt, field_a, field_b)
    }
}

pub trait FunctionOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this output field")
    }
}

pub type GetOutput = LazySerde<SpecialEq<Arc<dyn FunctionOutputField>>>;

impl Default for GetOutput {
    fn default() -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| Ok(fields[0].clone()),
        )))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn first() -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| Ok(fields[0].clone()),
        )))
    }

    pub fn from_type(dt: DataType) -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            move |_: &Schema, _: Context, flds: &[Field]| {
                Ok(Field::new(flds[0].name().clone(), dt.clone()))
            },
        )))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> PolarsResult<Field> + Send + Sync>(f: F) -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            move |_: &Schema, _: Context, flds: &[Field]| f(&flds[0]),
        )))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> PolarsResult<Field> + Send + Sync>(
        f: F,
    ) -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            move |_: &Schema, _: Context, flds: &[Field]| f(flds),
        )))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> PolarsResult<DataType> + Send + Sync>(
        f: F,
    ) -> Self {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            move |_: &Schema, _: Context, flds: &[Field]| {
                let mut fld = flds[0].clone();
                let new_type = f(fld.dtype())?;
                fld.coerce(new_type);
                Ok(fld)
            },
        )))
    }

    pub fn float_type() -> Self {
        Self::map_dtype(|dt| {
            Ok(match dt {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        })
    }

    pub fn super_type() -> Self {
        Self::map_dtypes(|dtypes| {
            let mut st = dtypes[0].clone();
            for dt in &dtypes[1..] {
                st = try_get_supertype(&st, dt)?;
            }
            Ok(st)
        })
    }

    pub fn map_dtypes<F>(f: F) -> Self
    where
        F: 'static + Fn(&[&DataType]) -> PolarsResult<DataType> + Send + Sync,
    {
        LazySerde::Deserialized(SpecialEq::new(Arc::new(
            move |_: &Schema, _: Context, flds: &[Field]| {
                let mut fld = flds[0].clone();
                let dtypes = flds.iter().map(|fld| fld.dtype()).collect::<Vec<_>>();
                let new_type = f(&dtypes)?;
                fld.coerce(new_type);
                Ok(fld)
            },
        )))
    }
}

impl<F> FunctionOutputField for F
where
    F: Fn(&Schema, Context, &[Field]) -> PolarsResult<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        self(input_schema, cntxt, fields)
    }
}

pub type OpaqueColumnUdf = LazySerde<SpecialEq<Arc<dyn ColumnsUdf>>>;
pub(crate) fn new_column_udf<F: ColumnsUdf + 'static>(func: F) -> OpaqueColumnUdf {
    LazySerde::Deserialized(SpecialEq::new(Arc::new(func)))
}

impl OpaqueColumnUdf {
    pub fn materialize(self) -> PolarsResult<SpecialEq<Arc<dyn ColumnsUdf>>> {
        match self {
            Self::Deserialized(t) => Ok(t),
            Self::Named {
                name: _,
                payload: _,
                value: _,
            } => {
                panic!("should not be hit")
            },
            Self::Bytes(_b) => {
                feature_gated!("serde";"python", {
                    serde_expr::deserialize_column_udf(_b.as_ref()).map(SpecialEq::new)
                })
            },
        }
    }
}

impl GetOutput {
    pub fn materialize(self) -> PolarsResult<SpecialEq<Arc<dyn FunctionOutputField>>> {
        match self {
            Self::Deserialized(t) => Ok(t),
            Self::Named {
                name: _,
                payload: _,
                value,
            } => value.ok_or_else(|| polars_err!(ComputeError: "GetOutput Value not set")),
            Self::Bytes(_b) => {
                polars_bail!(ComputeError: "should not be hit")
            },
        }
    }
}
