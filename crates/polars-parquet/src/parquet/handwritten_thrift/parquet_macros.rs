// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// These macros are adapted from Jörn Horstmann's thrift macros at
// https://github.com/jhorstmann/compact-thrift
// They allow for pasting sections of the Parquet thrift IDL file
// into a macro to generate rust structures and implementations.

//! This is a collection of macros used to parse Thrift IDL descriptions of structs,
//! unions, and enums into their corresponding Rust types. These macros will also
//! generate the code necessary to serialize and deserialize to/from the [Thrift compact]
//! protocol.
//!
//! Further details of how to use them (and other aspects of the Thrift serialization process)
//! can be found in [THRIFT.md].
//!
//! [Thrift compact]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#list-and-set
//! [THRIFT.md]: https://github.com/apache/arrow-rs/blob/main/parquet/THRIFT.md

#[doc(hidden)]
#[macro_export]
#[allow(clippy::crate_in_macro_def)]
/// Macro used to generate rust enums from a Thrift `enum` definition.
///
/// When utilizing this macro the Thrift serialization traits and structs need to be in scope.
macro_rules! thrift_enum {
    ($(#[$($def_attrs:tt)*])* enum $identifier:ident { $($(#[$($field_attrs:tt)*])* $field_name:ident = $field_value:literal;)* }) => {
        $(#[$($def_attrs)*])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[allow(non_camel_case_types)]
        #[allow(missing_docs)]
        pub enum $identifier {
            $($(#[cfg_attr(not(doctest), $($field_attrs)*)])* $field_name = $field_value,)*
        }

        impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for $identifier {
            #[allow(deprecated)]
            fn read_thrift(prot: &mut R) -> Result<Self> {
                let val = prot.read_i32()?;
                match val {
                    $($field_value => Ok(Self::$field_name),)*
                    _ => Err(general_err!("Unexpected {} {}", stringify!($identifier), val)),
                }
            }
        }

        impl fmt::Display for $identifier {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{self:?}")
            }
        }

        impl WriteThrift for $identifier {
            const ELEMENT_TYPE: ElementType = ElementType::I32;

            fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
                writer.write_i32(*self as i32)
            }
        }

        impl WriteThriftField for $identifier {
            fn write_thrift_field<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>, field_id: i16, last_field_id: i16) -> Result<i16> {
                writer.write_field_begin(FieldType::I32, field_id, last_field_id)?;
                self.write_thrift(writer)?;
                Ok(field_id)
            }
        }
    }
}

/// Macro used to generate Rust enums for Thrift unions in which all variants are typed with empty
/// structs.
///
/// Because the compact protocol does not write any struct type information, these empty structs
/// become a single `0` (end-of-fields marker) upon serialization. Rather than trying to deserialize
/// an empty struct, we can instead simply read the `0` and discard it.
///
/// The resulting Rust enum will have all unit variants.
///
/// When utilizing this macro the Thrift serialization traits and structs need to be in scope.
#[doc(hidden)]
#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! thrift_union_all_empty {
    ($(#[$($def_attrs:tt)*])* union $identifier:ident { $($(#[$($field_attrs:tt)*])* $field_id:literal : $field_type:ident $(< $element_type:ident >)? $field_name:ident $(;)?)* }) => {
        $(#[cfg_attr(not(doctest), $($def_attrs)*)])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        #[allow(missing_docs)]
        pub enum $identifier {
            $($(#[cfg_attr(not(doctest), $($field_attrs)*)])* $field_name),*
        }

        impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for $identifier {
            fn read_thrift(prot: &mut R) -> Result<Self> {
                let field_ident = prot.read_field_begin(0)?;
                if field_ident.field_type == FieldType::Stop {
                    return Err(general_err!("Received empty union from remote {}", stringify!($identifier)));
                }
                let ret = match field_ident.id {
                    $($field_id => {
                        prot.skip_empty_struct()?;
                        Self::$field_name
                    }
                    )*
                    _ => {
                        return Err(general_err!("Unexpected {} {}", stringify!($identifier), field_ident.id));
                    }
                };
                let field_ident = prot.read_field_begin(field_ident.id)?;
                if field_ident.field_type != FieldType::Stop {
                    return Err(general_err!(
                        "Received multiple fields for union from remote {}", stringify!($identifier)
                    ));
                }
                Ok(ret)
            }
        }

        impl WriteThrift for $identifier {
            const ELEMENT_TYPE: ElementType = ElementType::Struct;

            fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
                match *self {
                    $(Self::$field_name => writer.write_empty_struct($field_id, 0)?,)*
                };
                // write end of struct for this union
                writer.write_struct_end()
            }
        }

        impl WriteThriftField for $identifier {
            fn write_thrift_field<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>, field_id: i16, last_field_id: i16) -> Result<i16> {
                writer.write_field_begin(FieldType::Struct, field_id, last_field_id)?;
                self.write_thrift(writer)?;
                Ok(field_id)
            }
        }
    }
}

/// Macro used to generate Rust enums for Thrift unions where variants are a mix of unit and
/// tuple types.
///
/// Use of this macro requires modifying the thrift IDL. For variants with empty structs as their
/// type, delete the typename (i.e. `1: EmptyStruct Var1;` becomes `1: Var1`). For variants with a
/// non-empty type, the typename must be contained within parens (e.g. `1: MyType Var1;` becomes
/// `1: (MyType) Var1;`).
///
/// This macro allows for specifying lifetime annotations for the resulting `enum` and its fields.
///
/// When utilizing this macro the Thrift serialization traits and structs need to be in scope.
#[doc(hidden)]
#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! thrift_union {
    ($(#[$($def_attrs:tt)*])* union $identifier:ident $(< $lt:lifetime >)? { $($(#[$($field_attrs:tt)*])* $field_id:literal : $( ( $field_type:ident $(< $element_type:ident >)? $(< $field_lt:lifetime >)?) )? $field_name:ident $(;)?)* }) => {
        $(#[cfg_attr(not(doctest), $($def_attrs)*)])*
        #[derive(Clone, Debug, Eq, PartialEq)]
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        #[allow(missing_docs)]
        pub enum $identifier $(<$lt>)? {
            $($(#[cfg_attr(not(doctest), $($field_attrs)*)])* $field_name $( ( $crate::__thrift_union_type!{$field_type $($field_lt)? $($element_type)?} ) )?),*
        }

        impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for $identifier $(<$lt>)? {
            fn read_thrift(prot: &mut R) -> Result<Self> {
                let field_ident = prot.read_field_begin(0)?;
                if field_ident.field_type == FieldType::Stop {
                    return Err(general_err!("Received empty union from remote {}", stringify!($identifier)));
                }
                let ret = match field_ident.id {
                    $($field_id => {
                        let val = $crate::__thrift_read_variant!(prot, $field_name $($field_type $($element_type)?)?);
                        val
                    })*
                    _ => {
                        return Err(general_err!("Unexpected {} {}", stringify!($identifier), field_ident.id));
                    }
                };
                let field_ident = prot.read_field_begin(field_ident.id)?;
                if field_ident.field_type != FieldType::Stop {
                    return Err(general_err!(
                        concat!("Received multiple fields for union from remote {}", stringify!($identifier))
                    ));
                }
                Ok(ret)
            }
        }

        impl $(<$lt>)? WriteThrift for $identifier $(<$lt>)? {
            const ELEMENT_TYPE: ElementType = ElementType::Struct;

            fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
                match self {
                    $($crate::__thrift_write_variant_lhs!($field_name $($field_type)?, variant_val) =>
                      $crate::__thrift_write_variant_rhs!($field_id $($field_type)?, writer, variant_val),)*
                };
                writer.write_struct_end()
            }
        }

        impl $(<$lt>)? WriteThriftField for $identifier $(<$lt>)? {
            fn write_thrift_field<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>, field_id: i16, last_field_id: i16) -> Result<i16> {
                writer.write_field_begin(FieldType::Struct, field_id, last_field_id)?;
                self.write_thrift(writer)?;
                Ok(field_id)
            }
        }
    }
}

/// Macro used to generate Rust structs from a Thrift `struct` definition.
///
/// This macro allows for specifying lifetime annotations for the resulting `struct` and its fields.
///
/// When utilizing this macro the Thrift serialization traits and structs need to be in scope.
#[doc(hidden)]
#[macro_export]
macro_rules! thrift_struct {
    ($(#[$($def_attrs:tt)*])* $vis:vis struct $identifier:ident $(< $lt:lifetime >)? { $($(#[$($field_attrs:tt)*])* $field_id:literal : $required_or_optional:ident $field_type:ident $(< $field_lt:lifetime >)? $(< $element_type:ident >)? $field_name:ident $(= $default_value:literal)? $(;)?)* }) => {
        $(#[cfg_attr(not(doctest), $($def_attrs)*)])*
        #[derive(Clone, Debug, Eq, PartialEq)]
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        #[allow(missing_docs)]
        $vis struct $identifier $(<$lt>)? {
            $($(#[cfg_attr(not(doctest), $($field_attrs)*)])* $vis $field_name: $crate::__thrift_required_or_optional!($required_or_optional $crate::__thrift_field_type!($field_type $($field_lt)? $($element_type)?))),*
        }

        impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for $identifier $(<$lt>)? {
            fn read_thrift(prot: &mut R) -> Result<Self> {
                $(let mut $field_name: Option<$crate::__thrift_field_type!($field_type $($field_lt)? $($element_type)?)> = None;)*
                let mut last_field_id = 0i16;
                loop {
                    let field_ident = prot.read_field_begin(last_field_id)?;
                    if field_ident.field_type == FieldType::Stop {
                        break;
                    }
                    match field_ident.id {
                        $($field_id => {
                            let val = $crate::__thrift_read_field!(prot, field_ident, $field_type $($field_lt)? $($element_type)?);
                            $field_name = Some(val);
                        })*
                        _ => {
                            prot.skip(field_ident.field_type)?;
                        }
                    };
                    last_field_id = field_ident.id;
                }
                $($crate::__thrift_result_required_or_optional!($required_or_optional $field_name);)*
                Ok(Self {
                    $($field_name),*
                })
            }
        }

        impl $(<$lt>)? WriteThrift for $identifier $(<$lt>)? {
            const ELEMENT_TYPE: ElementType = ElementType::Struct;

            #[allow(unused_assignments)]
            fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
                #[allow(unused_mut, unused_variables)]
                let mut last_field_id = 0i16;
                $($crate::__thrift_write_required_or_optional_field!($required_or_optional $field_name, $field_id, $field_type, self, writer, last_field_id);)*
                writer.write_struct_end()
            }
        }

        impl $(<$lt>)? WriteThriftField for $identifier $(<$lt>)? {
            fn write_thrift_field<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>, field_id: i16, last_field_id: i16) -> Result<i16> {
                writer.write_field_begin(FieldType::Struct, field_id, last_field_id)?;
                self.write_thrift(writer)?;
                Ok(field_id)
            }
        }
    }
}

#[doc(hidden)]
#[macro_export]
/// Generate `WriteThriftField` implementation for a struct.
macro_rules! write_thrift_field {
    ($identifier:ident $(< $lt:lifetime >)?, $fld_type:expr) => {
        impl $(<$lt>)? WriteThriftField for $identifier $(<$lt>)? {
            fn write_thrift_field<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>, field_id: i16, last_field_id: i16) -> Result<i16> {
                writer.write_field_begin($fld_type, field_id, last_field_id)?;
                self.write_thrift(writer)?;
                Ok(field_id)
            }
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_write_required_or_optional_field {
    (required $field_name:ident, $field_id:literal, $field_type:ident, $self:tt, $writer:tt, $last_id:tt) => {
        $crate::__thrift_write_required_field!(
            $field_type,
            $field_name,
            $field_id,
            $self,
            $writer,
            $last_id
        )
    };
    (optional $field_name:ident, $field_id:literal, $field_type:ident, $self:tt, $writer:tt, $last_id:tt) => {
        $crate::__thrift_write_optional_field!(
            $field_type,
            $field_name,
            $field_id,
            $self,
            $writer,
            $last_id
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_write_required_field {
    (binary, $field_name:ident, $field_id:literal, $self:ident, $writer:ident, $last_id:ident) => {
        $writer.write_field_begin(FieldType::Binary, $field_id, $last_id)?;
        $writer.write_bytes($self.$field_name)?;
        $last_id = $field_id;
    };
    ($field_type:ident, $field_name:ident, $field_id:literal, $self:ident, $writer:ident, $last_id:ident) => {
        $last_id = $self
            .$field_name
            .write_thrift_field($writer, $field_id, $last_id)?;
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_write_optional_field {
    (binary, $field_name:ident, $field_id:literal, $self:ident, $writer:tt, $last_id:tt) => {
        if $self.$field_name.is_some() {
            $writer.write_field_begin(FieldType::Binary, $field_id, $last_id)?;
            $writer.write_bytes($self.$field_name.as_ref().unwrap())?;
            $last_id = $field_id;
        }
    };
    ($field_type:ident, $field_name:ident, $field_id:literal, $self:ident, $writer:tt, $last_id:tt) => {
        if $self.$field_name.is_some() {
            $last_id = $self
                .$field_name
                .as_ref()
                .unwrap()
                .write_thrift_field($writer, $field_id, $last_id)?;
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_required_or_optional {
    (required $field_type:ty) => { $field_type };
    (optional $field_type:ty) => { Option<$field_type> };
}

// Performance note: using `expect` here is about 4% faster on the page index bench,
// but we want to propagate errors. Using `ok_or` is *much* slower.
#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_result_required_or_optional {
    (required $field_name:ident) => {
        let Some($field_name) = $field_name else {
            return Err(general_err!(concat!(
                "Required field ",
                stringify!($field_name),
                " is missing",
            )));
        };
    };
    (optional $field_name:ident) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_read_field {
    ($prot:tt, $field_ident:tt, list $lt:lifetime binary) => {
        read_thrift_vec::<&'a [u8], R>(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, list $lt:lifetime $element_type:ident) => {
        read_thrift_vec::<$element_type, R>(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, list string) => {
        read_thrift_vec::<String, R>(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, list $element_type:ident) => {
        read_thrift_vec::<$element_type, R>(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, string $lt:lifetime) => {
        <&$lt str>::read_thrift(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, binary $lt:lifetime) => {
        <&$lt [u8]>::read_thrift(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, $field_type:ident $lt:lifetime) => {
        $field_type::read_thrift(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, string) => {
        String::read_thrift(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, binary) => {
        // this one needs to not conflict with `list<i8>`
        $prot.read_bytes_owned()?
    };
    ($prot:tt, $field_ident:tt, double) => {
        // Polars integration: arrow puts `parquet_thrift` at crate root; polars nests it.
        $crate::parquet::handwritten_thrift::parquet_thrift::OrderedF64::read_thrift(&mut *$prot)?
    };
    ($prot:tt, $field_ident:tt, bool) => {
        $field_ident.bool_val.unwrap()
    };
    ($prot:tt, $field_ident:tt, $field_type:ident) => {
        $field_type::read_thrift(&mut *$prot)?
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_field_type {
    (binary $lt:lifetime) => { &$lt [u8] };
    (string $lt:lifetime) => { &$lt str };
    ($field_type:ident $lt:lifetime) => { $field_type<$lt> };
    (list $lt:lifetime $element_type:ident) => { Vec< $crate::__thrift_field_type!($element_type $lt) > };
    (list string) => { Vec<String> };
    (list $element_type:ident) => { Vec< $crate::__thrift_field_type!($element_type) > };
    (binary) => { Vec<u8> };
    (string) => { String };
    // Polars integration: arrow puts `parquet_thrift` at crate root; polars nests it.
    (double) => { $crate::parquet::handwritten_thrift::parquet_thrift::OrderedF64 };
    ($field_type:ty) => { $field_type };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_union_type {
    (binary $lt:lifetime) => { &$lt [u8] };
    (string $lt:lifetime) => { &$lt str };
    ($field_type:ident $lt:lifetime) => { $field_type<$lt> };
    ($field_type:ident) => { $field_type };
    (list $field_type:ident) => { Vec<$field_type> };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_read_variant {
    ($prot:tt, $field_name:ident $field_type:ident) => {
        Self::$field_name($field_type::read_thrift(&mut *$prot)?)
    };
    ($prot:tt, $field_name:ident list $field_type:ident) => {
        Self::$field_name(Vec::<$field_type>::read_thrift(&mut *$prot)?)
    };
    ($prot:tt, $field_name:ident) => {{
        $prot.skip_empty_struct()?;
        Self::$field_name
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_write_variant_lhs {
    ($field_name:ident $field_type:ident, $val:tt) => {
        Self::$field_name($val)
    };
    ($field_name:ident, $val:tt) => {
        Self::$field_name
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrift_write_variant_rhs {
    ($field_id:literal $field_type:ident, $writer:tt, $val:ident) => {
        $val.write_thrift_field($writer, $field_id, 0)?
    };
    ($field_id:literal, $writer:tt, $val:tt) => {
        $writer.write_empty_struct($field_id, 0)?
    };
}
