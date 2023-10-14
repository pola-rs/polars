use std::sync::Arc;

use super::*;
use crate::datatypes::PhysicalType;
use crate::{match_integer_type, with_match_primitive_type};

impl PartialEq for dyn Scalar + '_ {
    fn eq(&self, that: &dyn Scalar) -> bool {
        equal(self, that)
    }
}

impl PartialEq<dyn Scalar> for Arc<dyn Scalar + '_> {
    fn eq(&self, that: &dyn Scalar) -> bool {
        equal(&**self, that)
    }
}

impl PartialEq<dyn Scalar> for Box<dyn Scalar + '_> {
    fn eq(&self, that: &dyn Scalar) -> bool {
        equal(&**self, that)
    }
}

macro_rules! dyn_eq {
    ($ty:ty, $lhs:expr, $rhs:expr) => {{
        let lhs = $lhs.as_any().downcast_ref::<$ty>().unwrap();
        let rhs = $rhs.as_any().downcast_ref::<$ty>().unwrap();
        lhs == rhs
    }};
}

fn equal(lhs: &dyn Scalar, rhs: &dyn Scalar) -> bool {
    if lhs.data_type() != rhs.data_type() {
        return false;
    }

    use PhysicalType::*;
    match lhs.data_type().to_physical_type() {
        Null => dyn_eq!(NullScalar, lhs, rhs),
        Boolean => dyn_eq!(BooleanScalar, lhs, rhs),
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            dyn_eq!(PrimitiveScalar<$T>, lhs, rhs)
        }),
        LargeUtf8 => dyn_eq!(Utf8Scalar<i64>, lhs, rhs),
        LargeBinary => dyn_eq!(BinaryScalar<i64>, lhs, rhs),
        LargeList => dyn_eq!(ListScalar<i64>, lhs, rhs),
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            dyn_eq!(DictionaryScalar<$T>, lhs, rhs)
        }),
        Struct => dyn_eq!(StructScalar, lhs, rhs),
        FixedSizeBinary => dyn_eq!(FixedSizeBinaryScalar, lhs, rhs),
        FixedSizeList => dyn_eq!(FixedSizeListScalar, lhs, rhs),
        Union => dyn_eq!(UnionScalar, lhs, rhs),
        Map => dyn_eq!(MapScalar, lhs, rhs),
        _ => unimplemented!(),
    }
}
