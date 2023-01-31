use super::*;

/// Given two datatypes, determine the supertype that both types can safely be cast to
#[cfg(feature = "private")]
pub fn try_get_supertype(l: &DataType, r: &DataType) -> PolarsResult<DataType> {
    get_supertype(l, r).ok_or_else(|| {
        PolarsError::ComputeError(
            format!("Failed to determine supertype of {l:?} and {r:?}").into(),
        )
    })
}

/// Given two datatypes, determine the supertype that both types can safely be cast to
#[cfg(feature = "private")]
pub fn get_supertype(l: &DataType, r: &DataType) -> Option<DataType> {
    fn inner(l: &DataType, r: &DataType) -> Option<DataType> {
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
            #[cfg(all(feature = "dtype-datetime", feature = "dtype=date"))]
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

            // every known type can be casted to a string except binary
            #[cfg(feature = "dtype-binary")]
            (dt, Utf8) if dt != &DataType::Unknown && dt != &DataType::Binary => Some(Utf8),

            #[cfg(not(feature = "dtype-binary"))]
            (dt, Utf8) if dt != &DataType::Unknown => Some(Utf8),

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

            // None and Some("") timezones
            // we cast from more precision to higher precision as that always fits with occasional loss of precision
            #[cfg(feature = "dtype-datetime")]
            (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r))
                if (tz_l.is_none() || tz_l.as_deref() == Some(""))
                    && (tz_r.is_none() || tz_r.as_deref() == Some("")) =>
            {
                let tu = get_time_units(tu_l, tu_r);
                Some(Datetime(tu, None))
            }
            // None and Some("<tz>") timezones
            // we cast from more precision to higher precision as that always fits with occasional loss of precision
            #[cfg(feature = "dtype-datetime")]
            (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r)) if
                // both are none
                tz_l.is_none() && tz_r.is_some()
                // both have the same time zone
                || (tz_l.is_some() && (tz_l == tz_r)) => {
                let tu = get_time_units(tu_l, tu_r);
                Some(Datetime(tu, tz_r.clone()))
            }
            (List(inner_left), List(inner_right)) => {
                let st = get_supertype(inner_left, inner_right)?;
                Some(DataType::List(Box::new(st)))
            }
            // todo! check if can be removed
            (List(inner), other) | (other, List(inner)) => {
                let st = get_supertype(inner, other)?;
                Some(DataType::List(Box::new(st)))
            }
            (_, Unknown) => Some(Unknown),
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
            _ => None,
        }
    }

    inner(l, r).or_else(|| inner(r, l))
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
