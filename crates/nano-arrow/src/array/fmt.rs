use std::fmt::{Result, Write};

use crate::bitmap::Bitmap;

use super::Array;

/// Returns a function that writes the value of the element of `array`
/// at position `index` to a [`Write`],
/// writing `null` in the null slots.
pub fn get_value_display<'a, F: Write + 'a>(
    array: &'a dyn Array,
    null: &'static str,
) -> Box<dyn Fn(&mut F, usize) -> Result + 'a> {
    use crate::datatypes::PhysicalType::*;
    match array.data_type().to_physical_type() {
        Null => Box::new(move |f, _| write!(f, "{null}")),
        Boolean => Box::new(|f, index| {
            super::boolean::fmt::write_value(array.as_any().downcast_ref().unwrap(), index, f)
        }),
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let writer = super::primitive::fmt::get_write_value::<$T, _>(
                array.as_any().downcast_ref().unwrap(),
            );
            Box::new(move |f, index| writer(f, index))
        }),
        Binary => Box::new(|f, index| {
            super::binary::fmt::write_value::<i32, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                f,
            )
        }),
        FixedSizeBinary => Box::new(|f, index| {
            super::fixed_size_binary::fmt::write_value(
                array.as_any().downcast_ref().unwrap(),
                index,
                f,
            )
        }),
        LargeBinary => Box::new(|f, index| {
            super::binary::fmt::write_value::<i64, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                f,
            )
        }),
        Utf8 => Box::new(|f, index| {
            super::utf8::fmt::write_value::<i32, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                f,
            )
        }),
        LargeUtf8 => Box::new(|f, index| {
            super::utf8::fmt::write_value::<i64, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                f,
            )
        }),
        List => Box::new(move |f, index| {
            super::list::fmt::write_value::<i32, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                null,
                f,
            )
        }),
        FixedSizeList => Box::new(move |f, index| {
            super::fixed_size_list::fmt::write_value(
                array.as_any().downcast_ref().unwrap(),
                index,
                null,
                f,
            )
        }),
        LargeList => Box::new(move |f, index| {
            super::list::fmt::write_value::<i64, _>(
                array.as_any().downcast_ref().unwrap(),
                index,
                null,
                f,
            )
        }),
        Struct => Box::new(move |f, index| {
            super::struct_::fmt::write_value(array.as_any().downcast_ref().unwrap(), index, null, f)
        }),
        Union => Box::new(move |f, index| {
            super::union::fmt::write_value(array.as_any().downcast_ref().unwrap(), index, null, f)
        }),
        Map => Box::new(move |f, index| {
            super::map::fmt::write_value(array.as_any().downcast_ref().unwrap(), index, null, f)
        }),
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            Box::new(move |f, index| {
                super::dictionary::fmt::write_value::<$T,_>(array.as_any().downcast_ref().unwrap(), index, null, f)
            })
        }),
    }
}

/// Returns a function that writes the element of `array`
/// at position `index` to a [`Write`], writing `null` to the null slots.
pub fn get_display<'a, F: Write + 'a>(
    array: &'a dyn Array,
    null: &'static str,
) -> Box<dyn Fn(&mut F, usize) -> Result + 'a> {
    let value_display = get_value_display(array, null);
    Box::new(move |f, row| {
        if array.is_null(row) {
            f.write_str(null)
        } else {
            value_display(f, row)
        }
    })
}

pub fn write_vec<D, F>(
    f: &mut F,
    d: D,
    validity: Option<&Bitmap>,
    len: usize,
    null: &'static str,
    new_lines: bool,
) -> Result
where
    D: Fn(&mut F, usize) -> Result,
    F: Write,
{
    f.write_char('[')?;
    write_list(f, d, validity, len, null, new_lines)?;
    f.write_char(']')?;
    Ok(())
}

fn write_list<D, F>(
    f: &mut F,
    d: D,
    validity: Option<&Bitmap>,
    len: usize,
    null: &'static str,
    new_lines: bool,
) -> Result
where
    D: Fn(&mut F, usize) -> Result,
    F: Write,
{
    for index in 0..len {
        if index != 0 {
            f.write_char(',')?;
            f.write_char(if new_lines { '\n' } else { ' ' })?;
        }
        if let Some(val) = validity {
            if val.get_bit(index) {
                d(f, index)
            } else {
                write!(f, "{null}")
            }
        } else {
            d(f, index)
        }?;
    }
    Ok(())
}

pub fn write_map<D, F>(
    f: &mut F,
    d: D,
    validity: Option<&Bitmap>,
    len: usize,
    null: &'static str,
    new_lines: bool,
) -> Result
where
    D: Fn(&mut F, usize) -> Result,
    F: Write,
{
    f.write_char('{')?;
    write_list(f, d, validity, len, null, new_lines)?;
    f.write_char('}')?;
    Ok(())
}
