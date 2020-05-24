use super::chunked_array::ChunkedArray;
use crate::series::chunked_array::{ChunkOps, SeriesOps};
use crate::{
    datatypes,
    datatypes::ArrowDataType,
    error::{PolarsError, Result},
};
use arrow::array::ArrayRef;
use arrow::compute::TakeOptions;
use arrow::datatypes::{ArrowPrimitiveType, Field};
use std::mem;
use std::ops::Deref;

#[derive(Clone)]
pub enum Series {
    Int32(ChunkedArray<datatypes::Int32Type>),
    Int64(ChunkedArray<datatypes::Int64Type>),
    Float32(ChunkedArray<datatypes::Float32Type>),
    Float64(ChunkedArray<datatypes::Float64Type>),
    Utf8(ChunkedArray<datatypes::Utf8Type>),
    Bool(ChunkedArray<datatypes::BooleanType>),
}

macro_rules! apply_method {
    ($self:ident, $method:ident, $($args:ident),*) => {
        match $self {
            Series::Int32(a) => a.$method($($args),*),
            Series::Int64(a) => a.$method($($args),*),
            Series::Float32(a) => a.$method($($args),*),
            Series::Float64(a) => a.$method($($args),*),
            Series::Utf8(a) => a.$method($($args),*),
            Series::Bool(a) => a.$method($($args),*),
            _ => unimplemented!(),
        }
    }
}

macro_rules! apply_method_and_return {
    ($self:ident, $method:ident, [$($args:expr),*], $($opt_question_mark:tt)*) => {
        match $self {
            Series::Int32(a) => Series::Int32(a.$method($($args),*)$($opt_question_mark)*),
            Series::Int64(a) => Series::Int64(a.$method($($args),*)$($opt_question_mark)*),
            Series::Float32(a) => Series::Float32(a.$method($($args),*)$($opt_question_mark)*),
            Series::Float64(a) => Series::Float64(a.$method($($args),*)$($opt_question_mark)*),
            Series::Utf8(a) => Series::Utf8(a.$method($($args),*)$($opt_question_mark)*),
            Series::Bool(a) => Series::Bool(a.$method($($args),*)$($opt_question_mark)*),
            _ => unimplemented!(),
        }
    }
}

impl Series {
    pub fn name(&self) -> &str {
        apply_method!(self, name,)
    }

    pub fn field(&self) -> &Field {
        apply_method!(self, ref_field,)
    }

    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        apply_method!(self, append_array, other)
    }

    pub fn as_series_ops(&self) -> &dyn SeriesOps {
        match self {
            Series::Int32(arr) => arr,
            Series::Int64(arr) => arr,
            Series::Float32(arr) => arr,
            Series::Float64(arr) => arr,
            Series::Utf8(arr) => arr,
            Series::Bool(arr) => arr,
        }
    }

    pub fn limit(&self, num_elements: usize) -> Result<Self> {
        Ok(apply_method_and_return!(self, limit, [num_elements], ?))
    }
    pub fn filter(&self, filter: &ChunkedArray<datatypes::BooleanType>) -> Result<Self> {
        Ok(apply_method_and_return!(self, filter, [filter], ?))
    }
    pub fn take(
        &self,
        indices: &ChunkedArray<datatypes::UInt32Type>,
        options: Option<TakeOptions>,
    ) -> Result<Self> {
        Ok(apply_method_and_return!(self, take, [indices, options], ?))
    }

    pub fn len(&self) -> usize {
        apply_method!(self, len,)
    }

    pub fn rechunk(&mut self) {
        apply_method!(self, rechunk,)
    }

    pub fn cast<N>(&self) -> Result<Self>
    where
        N: ArrowPrimitiveType,
    {
        let s = match self {
            Series::Int32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Int64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Float32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Float64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            _ => return Err(PolarsError::DataTypeMisMatch),
        };
        Ok(s)
    }
}

fn pack_ca_to_series<N: ArrowPrimitiveType>(ca: ChunkedArray<N>) -> Series {
    unsafe {
        match N::get_data_type() {
            ArrowDataType::Int32 => Series::Int32(mem::transmute(ca)),
            ArrowDataType::Int64 => Series::Int64(mem::transmute(ca)),
            ArrowDataType::Float32 => Series::Float32(mem::transmute(ca)),
            ArrowDataType::Float64 => Series::Float64(mem::transmute(ca)),
            _ => unimplemented!(),
        }
    }
}

pub trait NamedFrom<T> {
    fn from(name: &str, _: T) -> Self;
}

macro_rules! impl_named_from {
    ($type:ty, $series_var:ident, $method:ident) => {
        impl NamedFrom<&[$type]> for Series {
            fn from(name: &str, v: &[$type]) -> Self {
                Series::$series_var(ChunkedArray::$method(name, v))
            }
        }
    };
}

impl_named_from!(&str, Utf8, new_utf8_from_slice);
impl_named_from!(String, Utf8, new_utf8_from_slice);
impl_named_from!(bool, Bool, new_from_slice);
impl_named_from!(i32, Int32, new_from_slice);
impl_named_from!(i64, Int64, new_from_slice);
impl_named_from!(f32, Float32, new_from_slice);
impl_named_from!(f64, Float64, new_from_slice);

mod test {
    use super::*;

    #[test]
    fn cast() {
        let ar = ChunkedArray::<datatypes::Int32Type>::new_from_slice("a", &[1, 2]);
        let s = Series::Int32(ar);
        let s2 = s.cast::<datatypes::Int64Type>().unwrap();
        match s2 {
            Series::Int64(_) => assert!(true),
            _ => assert!(false),
        }
        let s2 = s.cast::<datatypes::Float32Type>().unwrap();
        match s2 {
            Series::Float32(_) => assert!(true),
            _ => assert!(false),
        }
    }
}
