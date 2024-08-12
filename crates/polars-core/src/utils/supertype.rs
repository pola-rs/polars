use bitflags::bitflags;
use num_traits::Signed;

use super::*;

/// Given two data types, determine the data type that both types can safely be cast to.
///
/// Returns a [`PolarsError::ComputeError`] if no such data type exists.
pub fn try_get_supertype(l: &DataType, r: &DataType) -> PolarsResult<DataType> {
    get_supertype(l, r).ok_or_else(
        || polars_err!(SchemaMismatch: "failed to determine supertype of {} and {}", l, r),
    )
}

bitflags! {
    #[repr(transparent)]
    #[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
    pub struct SuperTypeFlags: u8 {
        /// Implode lists to match nesting types.
        const ALLOW_IMPLODE_LIST = 1 << 0;
        /// Allow casting of primitive types (numeric, bools) to strings
        const ALLOW_PRIMITIVE_TO_STRING = 1 << 1;
    }
}

impl Default for SuperTypeFlags {
    fn default() -> Self {
        SuperTypeFlags::from_bits_truncate(0) | SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
pub struct SuperTypeOptions {
    pub flags: SuperTypeFlags,
}

impl From<SuperTypeFlags> for SuperTypeOptions {
    fn from(flags: SuperTypeFlags) -> Self {
        SuperTypeOptions { flags }
    }
}

impl SuperTypeOptions {
    pub fn allow_implode_list(&self) -> bool {
        self.flags.contains(SuperTypeFlags::ALLOW_IMPLODE_LIST)
    }

    pub fn allow_primitive_to_string(&self) -> bool {
        self.flags
            .contains(SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING)
    }
}

pub fn get_supertype(l: &DataType, r: &DataType) -> Option<DataType> {
    get_supertype_with_options(l, r, SuperTypeOptions::default())
}

/// Given two data types, determine the data type that both types can safely be cast to.
///
/// Returns [`None`] if no such data type exists.
pub fn get_supertype_with_options(
    l: &DataType,
    r: &DataType,
    options: SuperTypeOptions,
) -> Option<DataType> {
    fn inner(l: &DataType, r: &DataType, options: SuperTypeOptions) -> Option<DataType> {
        use DataType::*;
        if l == r {
            return Some(l.clone());
        }
        match (l, r) {
            #[cfg(feature = "dtype-i8")]
            (Int8, Boolean) => Some(Int8),
            //(Int8, Int8) => Some(Int8),
            #[cfg(all(feature = "dtype-i8", feature = "dtype-i16"))]
            (Int8, Int16) => Some(Int16),
            #[cfg(feature = "dtype-i8")]
            (Int8, Int32) => Some(Int32),
            #[cfg(feature = "dtype-i8")]
            (Int8, Int64) => Some(Int64),
            #[cfg(all(feature = "dtype-i8", feature = "dtype-i16"))]
            (Int8, UInt8) => Some(Int16),
            #[cfg(all(feature = "dtype-i8", feature = "dtype-u16"))]
            (Int8, UInt16) => Some(Int32),
            #[cfg(feature = "dtype-i8")]
            (Int8, UInt32) => Some(Int64),
            #[cfg(feature = "dtype-i8")]
            (Int8, UInt64) => Some(Float64), // Follow numpy
            #[cfg(feature = "dtype-i8")]
            (Int8, Float32) => Some(Float32),
            #[cfg(feature = "dtype-i8")]
            (Int8, Float64) => Some(Float64),

            #[cfg(feature = "dtype-i16")]
            (Int16, Boolean) => Some(Int16),
            #[cfg(all(feature = "dtype-i16", feature = "dtype-i8"))]
            (Int16, Int8) => Some(Int16),
            //(Int16, Int16) => Some(Int16),
            #[cfg(feature = "dtype-i16")]
            (Int16, Int32) => Some(Int32),
            #[cfg(feature = "dtype-i16")]
            (Int16, Int64) => Some(Int64),
            #[cfg(all(feature = "dtype-i16", feature = "dtype-u8"))]
            (Int16, UInt8) => Some(Int16),
            #[cfg(all(feature = "dtype-i16", feature = "dtype-u16"))]
            (Int16, UInt16) => Some(Int32),
            #[cfg(feature = "dtype-i16")]
            (Int16, UInt32) => Some(Int64),
            #[cfg(feature = "dtype-i16")]
            (Int16, UInt64) => Some(Float64), // Follow numpy
            #[cfg(feature = "dtype-i16")]
            (Int16, Float32) => Some(Float32),
            #[cfg(feature = "dtype-i16")]
            (Int16, Float64) => Some(Float64),

            (Int32, Boolean) => Some(Int32),
            #[cfg(feature = "dtype-i8")]
            (Int32, Int8) => Some(Int32),
            #[cfg(feature = "dtype-i16")]
            (Int32, Int16) => Some(Int32),
            //(Int32, Int32) => Some(Int32),
            (Int32, Int64) => Some(Int64),
            #[cfg(feature = "dtype-u8")]
            (Int32, UInt8) => Some(Int32),
            #[cfg(feature = "dtype-u16")]
            (Int32, UInt16) => Some(Int32),
            (Int32, UInt32) => Some(Int64),
            #[cfg(not(feature = "bigidx"))]
            (Int32, UInt64) => Some(Float64), // Follow numpy
            #[cfg(feature = "bigidx")]
            (Int32, UInt64) => Some(Int64), // Needed for bigidx
            (Int32, Float32) => Some(Float64), // Follow numpy
            (Int32, Float64) => Some(Float64),

            (Int64, Boolean) => Some(Int64),
            #[cfg(feature = "dtype-i8")]
            (Int64, Int8) => Some(Int64),
            #[cfg(feature = "dtype-i16")]
            (Int64, Int16) => Some(Int64),
            (Int64, Int32) => Some(Int64),
            //(Int64, Int64) => Some(Int64),
            #[cfg(feature = "dtype-u8")]
            (Int64, UInt8) => Some(Int64),
            #[cfg(feature = "dtype-u16")]
            (Int64, UInt16) => Some(Int64),
            (Int64, UInt32) => Some(Int64),
            #[cfg(not(feature = "bigidx"))]
            (Int64, UInt64) => Some(Float64), // Follow numpy
            #[cfg(feature = "bigidx")]
            (Int64, UInt64) => Some(Int64), // Needed for bigidx
            (Int64, Float32) => Some(Float64), // Follow numpy
            (Int64, Float64) => Some(Float64),

            #[cfg(all(feature = "dtype-u16", feature = "dtype-u8"))]
            (UInt16, UInt8) => Some(UInt16),
            #[cfg(feature = "dtype-u16")]
            (UInt16, UInt32) => Some(UInt32),
            #[cfg(feature = "dtype-u16")]
            (UInt16, UInt64) => Some(UInt64),

            #[cfg(feature = "dtype-u8")]
            (UInt8, UInt32) => Some(UInt32),
            #[cfg(feature = "dtype-u8")]
            (UInt8, UInt64) => Some(UInt64),

            (UInt32, UInt64) => Some(UInt64),

            #[cfg(feature = "dtype-u8")]
            (Boolean, UInt8) => Some(UInt8),
            #[cfg(feature = "dtype-u16")]
            (Boolean, UInt16) => Some(UInt16),
            (Boolean, UInt32) => Some(UInt32),
            (Boolean, UInt64) => Some(UInt64),

            #[cfg(feature = "dtype-u8")]
            (Float32, UInt8) => Some(Float32),
            #[cfg(feature = "dtype-u16")]
            (Float32, UInt16) => Some(Float32),
            (Float32, UInt32) => Some(Float64),
            (Float32, UInt64) => Some(Float64),

            #[cfg(feature = "dtype-u8")]
            (Float64, UInt8) => Some(Float64),
            #[cfg(feature = "dtype-u16")]
            (Float64, UInt16) => Some(Float64),
            (Float64, UInt32) => Some(Float64),
            (Float64, UInt64) => Some(Float64),

            (Float64, Float32) => Some(Float64),

            // Time related dtypes
            #[cfg(feature = "dtype-date")]
            (Date, UInt32) => Some(Int64),
            #[cfg(feature = "dtype-date")]
            (Date, UInt64) => Some(Int64),
            #[cfg(feature = "dtype-date")]
            (Date, Int32) => Some(Int32),
            #[cfg(feature = "dtype-date")]
            (Date, Int64) => Some(Int64),
            #[cfg(feature = "dtype-date")]
            (Date, Float32) => Some(Float32),
            #[cfg(feature = "dtype-date")]
            (Date, Float64) => Some(Float64),
            #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
            (Date, Datetime(tu, tz)) => Some(Datetime(*tu, tz.clone())),

            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), UInt32) => Some(Int64),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), UInt64) => Some(Int64),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), Int32) => Some(Int64),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), Int64) => Some(Int64),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), Float32) => Some(Float64),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(_, _), Float64) => Some(Float64),
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Datetime(tu, tz), Date) => Some(Datetime(*tu, tz.clone())),

            (Boolean, Float32) => Some(Float32),
            (Boolean, Float64) => Some(Float64),

            #[cfg(feature = "dtype-duration")]
            (Duration(_), UInt32) => Some(Int64),
            #[cfg(feature = "dtype-duration")]
            (Duration(_), UInt64) => Some(Int64),
            #[cfg(feature = "dtype-duration")]
            (Duration(_), Int32) => Some(Int64),
            #[cfg(feature = "dtype-duration")]
            (Duration(_), Int64) => Some(Int64),
            #[cfg(feature = "dtype-duration")]
            (Duration(_), Float32) => Some(Float64),
            #[cfg(feature = "dtype-duration")]
            (Duration(_), Float64) => Some(Float64),

            #[cfg(feature = "dtype-time")]
            (Time, Int32) => Some(Int64),
            #[cfg(feature = "dtype-time")]
            (Time, Int64) => Some(Int64),
            #[cfg(feature = "dtype-time")]
            (Time, Float32) => Some(Float64),
            #[cfg(feature = "dtype-time")]
            (Time, Float64) => Some(Float64),

            // Every known type can be cast to a string except binary
            (dt, String) if !matches!(dt, Unknown(UnknownKind::Any)) && dt != &Binary && options.allow_primitive_to_string() || !dt.to_physical().is_primitive() => Some(String),
            (String, Binary) => Some(Binary),
            (dt, Null) => Some(dt.clone()),

            #[cfg(all(feature = "dtype-duration", feature = "dtype-datetime"))]
            (Duration(lu), Datetime(ru, Some(tz))) | (Datetime(lu, Some(tz)), Duration(ru)) => {
                if tz.is_empty() {
                    Some(Datetime(get_time_units(lu, ru), None))
                } else {
                    Some(Datetime(get_time_units(lu, ru), Some(tz.clone())))
                }
            }
            #[cfg(all(feature = "dtype-duration", feature = "dtype-datetime"))]
            (Duration(lu), Datetime(ru, None)) | (Datetime(lu, None), Duration(ru)) => {
                Some(Datetime(get_time_units(lu, ru), None))
            }
            #[cfg(all(feature = "dtype-duration", feature = "dtype-date"))]
            (Duration(_), Date) | (Date, Duration(_)) => Some(Date),
            #[cfg(feature = "dtype-duration")]
            (Duration(lu), Duration(ru)) => Some(Duration(get_time_units(lu, ru))),

            // both None or both Some("<tz>") timezones
            // we cast from more precision to higher precision as that always fits with occasional loss of precision
            #[cfg(feature = "dtype-datetime")]
            (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r)) if
                // both are none
                (tz_l.is_none() && tz_r.is_none())
                // both have the same time zone
                || (tz_l.is_some() && (tz_l == tz_r)) => {
                let tu = get_time_units(tu_l, tu_r);
                Some(Datetime(tu, tz_r.clone()))
            }
            (List(inner_left), List(inner_right)) => {
                let st = get_supertype(inner_left, inner_right)?;
                Some(List(Box::new(st)))
            }
            #[cfg(feature = "dtype-array")]
            (List(inner_left), Array(inner_right, _)) | (Array(inner_left, _), List(inner_right)) => {
                let st = get_supertype(inner_left, inner_right)?;
                Some(List(Box::new(st)))
            }
            #[cfg(feature = "dtype-array")]
            (Array(inner_left, width_left), Array(inner_right, width_right)) if *width_left == *width_right => {
                let st = get_supertype(inner_left, inner_right)?;
                Some(Array(Box::new(st), *width_left))
            }
            (List(inner), other) | (other, List(inner)) if options.allow_implode_list() => {
                let st = get_supertype(inner, other)?;
                Some(List(Box::new(st)))
            }
            #[cfg(feature = "dtype-array")]
            (Array(inner_left, _), Array(inner_right, _)) => {
                let st = get_supertype(inner_left, inner_right)?;
                Some(List(Box::new(st)))
            }
            #[cfg(feature = "dtype-struct")]
            (Struct(inner), right @ Unknown(UnknownKind::Float | UnknownKind::Int(_))) => {
                match inner.first() {
                    Some(inner) => get_supertype(&inner.dtype, right),
                    None => None
                }
            },
            (dt, Unknown(kind)) => {
                match kind {
                    UnknownKind::Float | UnknownKind::Int(_) if  dt.is_string() => {
                        if options.allow_primitive_to_string() {
                            Some(dt.clone())
                        } else {
                            None
                        }
                    },
                    // numeric vs float|str -> always float|str|decimal
                    UnknownKind::Float | UnknownKind::Int(_) if dt.is_float() | dt.is_decimal() => Some(dt.clone()),
                    UnknownKind::Float if dt.is_integer() => Some(Unknown(UnknownKind::Float)),
                    // Materialize float to float or decimal
                    UnknownKind::Float if dt.is_float() | dt.is_decimal() => Some(dt.clone()),
                    // Materialize str
                    UnknownKind::Str if dt.is_string() | dt.is_enum() => Some(dt.clone()),
                    // Materialize str
                    #[cfg(feature = "dtype-categorical")]
                    UnknownKind::Str if dt.is_categorical()  => {
                        let Categorical(_, ord) = dt else { unreachable!()};
                        Some(Categorical(None, *ord))
                    },
                    // Keep unknown
                    dynam if dt.is_null() => Some(Unknown(*dynam)),
                    // Find integers sizes
                    UnknownKind::Int(v) if dt.is_numeric() => {
                        // Both dyn int
                        if let Unknown(UnknownKind::Int(v_other)) = dt {
                            // Take the maximum value to ensure we bubble up the required minimal size.
                            Some(Unknown(UnknownKind::Int(std::cmp::max(*v, *v_other))))
                        }
                        // dyn int vs number
                        else {
                            let smallest_fitting_dtype = if dt.is_unsigned_integer() && !v.is_negative() {
                                materialize_dyn_int_pos(*v).dtype()
                            } else {
                                materialize_smallest_dyn_int(*v).dtype()
                            };
                            match dt {
                                UInt64 if smallest_fitting_dtype.is_signed_integer() => {
                                    // Ensure we don't cast to float when dealing with dynamic literals
                                    Some(Int64)
                                },
                                _ => {
                                    get_supertype(dt, &smallest_fitting_dtype)
                                }
                            }
                        }
                    }
                    UnknownKind::Int(_) if dt.is_decimal() => Some(dt.clone()),
                    _ => Some(Unknown(UnknownKind::Any))
                }
            },
            #[cfg(feature = "dtype-struct")]
            (Struct(fields_a), Struct(fields_b)) => {
                super_type_structs(fields_a, fields_b)
            }
            #[cfg(feature = "dtype-struct")]
            (Struct(fields_a), rhs) if rhs.is_numeric() => {
                let mut new_fields = Vec::with_capacity(fields_a.len());
                for a in fields_a {
                    let st = get_supertype(&a.dtype, rhs)?;
                    new_fields.push(Field::new(&a.name, st))
                }
                Some(Struct(new_fields))
            }
            #[cfg(feature = "dtype-decimal")]
            (Decimal(p1, s1), Decimal(p2, s2)) => {
                Some(Decimal((*p1).zip(*p2).map(|(p1, p2)| p1.max(p2)), (*s1).max(*s2)))
            }
            #[cfg(feature = "dtype-decimal")]
            (Decimal(_, _), f @ (Float32 | Float64)) => Some(f.clone()),
            #[cfg(feature = "dtype-decimal")]
            (d @ Decimal(_, _), dt) if dt.is_signed_integer() || dt.is_unsigned_integer() => Some(d.clone()),
            _ => None,
        }
    }

    inner(l, r, options).or_else(|| inner(r, l, options))
}

/// Given multiple data types, determine the data type that all types can safely be cast to.
///
/// Returns [`DataType::Null`] if no data types were passed.
pub fn dtypes_to_supertype<'a, I>(dtypes: I) -> PolarsResult<DataType>
where
    I: IntoIterator<Item = &'a DataType>,
{
    dtypes
        .into_iter()
        .try_fold(DataType::Null, |supertype, dtype| {
            try_get_supertype(&supertype, dtype)
        })
}

#[cfg(feature = "dtype-struct")]
fn union_struct_fields(fields_a: &[Field], fields_b: &[Field]) -> Option<DataType> {
    let (longest, shortest) = {
        // if equal length we also take the lhs
        // so that the lhs determines the order of the fields
        if fields_a.len() >= fields_b.len() {
            (fields_a, fields_b)
        } else {
            (fields_b, fields_a)
        }
    };
    let mut longest_map =
        PlIndexMap::from_iter(longest.iter().map(|fld| (&fld.name, fld.dtype.clone())));
    for field in shortest {
        let dtype_longest = longest_map
            .entry(&field.name)
            .or_insert_with(|| field.dtype.clone());
        if &field.dtype != dtype_longest {
            let st = get_supertype(&field.dtype, dtype_longest)?;
            *dtype_longest = st
        }
    }
    let new_fields = longest_map
        .into_iter()
        .map(|(name, dtype)| Field::new(name, dtype))
        .collect::<Vec<_>>();
    Some(DataType::Struct(new_fields))
}

#[cfg(feature = "dtype-struct")]
fn super_type_structs(fields_a: &[Field], fields_b: &[Field]) -> Option<DataType> {
    if fields_a.len() != fields_b.len() {
        union_struct_fields(fields_a, fields_b)
    } else {
        let mut new_fields = Vec::with_capacity(fields_a.len());
        for (a, b) in fields_a.iter().zip(fields_b) {
            if a.name != b.name {
                return union_struct_fields(fields_a, fields_b);
            }
            let st = get_supertype(&a.dtype, &b.dtype)?;
            new_fields.push(Field::new(&a.name, st))
        }
        Some(DataType::Struct(new_fields))
    }
}

pub fn materialize_dyn_int(v: i128) -> AnyValue<'static> {
    // Try to get the "smallest" fitting value.
    // TODO! next breaking go to true smallest.
    match i32::try_from(v).ok() {
        Some(v) => AnyValue::Int32(v),
        None => match i64::try_from(v).ok() {
            Some(v) => AnyValue::Int64(v),
            None => match u64::try_from(v).ok() {
                Some(v) => AnyValue::UInt64(v),
                None => AnyValue::Null,
            },
        },
    }
}
fn materialize_dyn_int_pos(v: i128) -> AnyValue<'static> {
    // Try to get the "smallest" fitting value.
    // TODO! next breaking go to true smallest.
    match u8::try_from(v).ok() {
        Some(v) => AnyValue::UInt8(v),
        None => match u16::try_from(v).ok() {
            Some(v) => AnyValue::UInt16(v),
            None => match u32::try_from(v).ok() {
                Some(v) => AnyValue::UInt32(v),
                None => match u64::try_from(v).ok() {
                    Some(v) => AnyValue::UInt64(v),
                    None => AnyValue::Null,
                },
            },
        },
    }
}

fn materialize_smallest_dyn_int(v: i128) -> AnyValue<'static> {
    match i8::try_from(v).ok() {
        Some(v) => AnyValue::Int8(v),
        None => match i16::try_from(v).ok() {
            Some(v) => AnyValue::Int16(v),
            None => match i32::try_from(v).ok() {
                Some(v) => AnyValue::Int32(v),
                None => match i64::try_from(v).ok() {
                    Some(v) => AnyValue::Int64(v),
                    None => match u64::try_from(v).ok() {
                        Some(v) => AnyValue::UInt64(v),
                        None => AnyValue::Null,
                    },
                },
            },
        },
    }
}
