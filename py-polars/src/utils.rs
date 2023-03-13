use polars::prelude::*;

pub fn reinterpret(s: &Series, signed: bool) -> PolarsResult<Series> {
    Ok(match (s.dtype(), signed) {
        (DataType::UInt64, true) => s.u64().unwrap().reinterpret_signed().into_series(),
        (DataType::UInt64, false) => s.clone(),
        (DataType::Int64, false) => s.i64().unwrap().reinterpret_unsigned().into_series(),
        (DataType::Int64, true) => s.clone(),
        (DataType::UInt32, true) => s.u32().unwrap().reinterpret_signed().into_series(),
        (DataType::UInt32, false) => s.clone(),
        (DataType::Int32, false) => s.i32().unwrap().reinterpret_unsigned().into_series(),
        (DataType::Int32, true) => s.clone(),
        _ => polars_bail!(
            ComputeError:
            "reinterpret is only allowed for 64-bit/32-bit integers types, use cast otherwise"
        ),
    })
}

// was redefined because I could not get feature flags activated?
#[macro_export]
macro_rules! apply_method_all_arrow_series2 {
    ($self:expr, $method:ident, $($args:expr),*) => {
        match $self.dtype() {
            DataType::Boolean => $self.bool().unwrap().$method($($args),*),
            DataType::Utf8 => $self.utf8().unwrap().$method($($args),*),
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Date => $self.date().unwrap().$method($($args),*),
            DataType::Datetime(_, _) => $self.datetime().unwrap().$method($($args),*),
            DataType::List(_) => $self.list().unwrap().$method($($args),*),
            DataType::Struct(_) => $self.struct_().unwrap().$method($($args),*),
            dt => panic!("dtype {:?} not supported", dt)
        }
    }
}
