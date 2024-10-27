use std::borrow::Cow;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
use crate::datatypes::{AnyValue, PolarsNumericType};

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum LambdaExpression {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    StaticStr(Cow<'static, str>),
    Variable(usize),
    GreaterThan(Box<Self>, Box<Self>),
    LessThan(Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    CaseWhen(Vec<(Box<Self>, Box<Self>)>, Box<Self>)
}

impl LambdaExpression {
    #[inline]
    pub(crate) fn eval_numeric<T: PolarsNumericType>(&self, args: &[&T::Native]) -> AnyValue {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                (*args[*idx]).into()
            }
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) > right.eval_numeric::<T>(args))
            }
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) < right.eval_numeric::<T>(args))
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_numeric::<T>(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_numeric::<T>(args)
                } else {
                    falsy.eval_numeric::<T>(args)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_numeric::<T>(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_numeric::<T>(args)
                    }
                }
                otherwise.eval_numeric::<T>(args)
            }
        }
    }

    #[inline]
    pub(crate) fn eval_bool(&self, args: &[&bool]) -> AnyValue {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                (*args[*idx]).into()
            }
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) > right.eval_bool(args))
            }
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) < right.eval_bool(args))
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_bool(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_bool(args)
                } else {
                    falsy.eval_bool(args)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_bool(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_bool(args)
                    }
                }
                otherwise.eval_bool(args)
            }
        }
    }

    #[inline]
    pub(crate) fn eval_slice<'a>(&'a self, args: &'a[&'a[u8]]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                AnyValue::Binary(args[*idx])
            }
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_slice(args).gt(&right.eval_slice(args)).into()
            }
            LambdaExpression::LessThan(left, right) => {
                left.eval_slice(args).lt(&right.eval_slice(args)).into()
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_slice(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_slice(args)
                } else {
                    falsy.eval_slice(args)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_slice(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_slice(args)
                    }
                }
                otherwise.eval_slice(args)
            },
        }
    }
}
