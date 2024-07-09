use std::fmt::Formatter;
use std::ops::Deref;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use serde::{Deserializer, Serializer};

use super::*;

/// A wrapper trait for any closure `Fn(Vec<Series>) -> PolarsResult<Series>`
pub trait SeriesUdf: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any {
        unimplemented!("as_any not implemented for this 'opaque' function")
    }

    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialize not supported for this 'opaque' function")
    }

    // Needed for python functions. After they are deserialized we first check if they
    // have a function that generates an output
    // This will be slower during optimization, so it is up to us to move
    // all expression to the known function architecture.
    fn get_output(&self) -> Option<GetOutput> {
        None
    }
}

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Arc<dyn SeriesUdf>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let mut buf = vec![];
        self.0
            .try_serialize(&mut buf)
            .map_err(|e| S::Error::custom(format!("{e}")))?;
        serializer.serialize_bytes(&buf)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        #[cfg(feature = "python")]
        {
            use crate::dsl::python_udf::MAGIC_BYTE_MARK;
            let buf = Vec::<u8>::deserialize(_deserializer)?;

            if buf.starts_with(MAGIC_BYTE_MARK) {
                let udf = python_udf::PythonUdfExpression::try_deserialize(&buf)
                    .map_err(|e| D::Error::custom(format!("{e}")))?;
                Ok(SpecialEq::new(udf))
            } else {
                Err(D::Error::custom(
                    "deserialize not supported for this 'opaque' function",
                ))
            }
        }
        #[cfg(not(feature = "python"))]
        {
            Err(D::Error::custom(
                "deserialize not supported for this 'opaque' function",
            ))
        }
    }
}

impl<F> SeriesUdf for F
where
    F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        self(s)
    }
}

impl Debug for dyn SeriesUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Series, Series) -> PolarsResult<Series>`
pub trait SeriesBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Series, b: Series) -> PolarsResult<Series>;
}

impl<F> SeriesBinaryUdf for F
where
    F: Fn(Series, Series) -> PolarsResult<Series> + Send + Sync,
{
    fn call_udf(&self, a: Series, b: Series) -> PolarsResult<Series> {
        self(a, b)
    }
}

impl Debug for dyn SeriesBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesBinaryUdf")
    }
}

impl Default for SpecialEq<Arc<dyn SeriesBinaryUdf>> {
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
    fn call(&self, name: &str) -> PolarsResult<String>;
}

impl<F: Fn(&str) -> PolarsResult<String> + Send + Sync> RenameAliasFn for F {
    fn call(&self, name: &str) -> PolarsResult<String> {
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

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Series> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for SpecialEq<Series> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = Series::deserialize(deserializer)?;
        Ok(SpecialEq(t))
    }
}

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Arc<DslPlan>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for SpecialEq<Arc<DslPlan>> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = DslPlan::deserialize(deserializer)?;
        Ok(SpecialEq(Arc::new(t)))
    }
}

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
}

pub type GetOutput = SpecialEq<Arc<dyn FunctionOutputField>>;

impl Default for GetOutput {
    fn default() -> Self {
        SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| Ok(fields[0].clone()),
        ))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn from_type(dt: DataType) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            Ok(Field::new(flds[0].name(), dt.clone()))
        }))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> PolarsResult<Field> + Send + Sync>(f: F) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(&flds[0])
        }))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> PolarsResult<Field> + Send + Sync>(
        f: F,
    ) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(flds)
        }))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> PolarsResult<DataType> + Send + Sync>(
        f: F,
    ) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let new_type = f(fld.data_type())?;
            fld.coerce(new_type);
            Ok(fld)
        }))
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
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let dtypes = flds.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
            let new_type = f(&dtypes)?;
            fld.coerce(new_type);
            Ok(fld)
        }))
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
