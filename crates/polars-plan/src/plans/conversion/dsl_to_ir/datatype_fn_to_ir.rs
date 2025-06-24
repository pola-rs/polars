use polars_core::error::{PolarsResult, feature_gated, polars_bail, polars_ensure};
use polars_core::prelude::{AnyValue, DataType, Field, NamedFrom};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use crate::constants::get_literal_name;
use crate::dsl::{DataTypeFunction, DataTypeKind, StructDataTypeFunction};
use crate::plans::{AExpr, LiteralValue};

pub fn datatype_fn_to_aexpr(
    f: DataTypeFunction,
    schema: &Schema,
    _arena: &mut Arena<AExpr>,
) -> PolarsResult<(AExpr, PlSmallStr)> {
    use DataTypeFunction as DTF;
    Ok(match f {
        DTF::ToString(dt) => (
            AExpr::Literal(LiteralValue::Scalar(Scalar::from(PlSmallStr::from_string(
                dt.into_datatype(schema)?.to_string(),
            )))),
            get_literal_name().clone(),
        ),
        DTF::Eq(l, r) => (
            AExpr::Literal(LiteralValue::Scalar(Scalar::from(
                l.into_datatype(schema)? == r.into_datatype(schema)?,
            ))),
            get_literal_name().clone(),
        ),
        DTF::IsKind(dt, kind) => {
            let dt = dt.into_datatype(schema)?;

            use DataTypeKind as DTK;
            let is_kind = match kind {
                DTK::Numeric => dt.is_numeric(),
                DTK::Integer => dt.is_integer(),
                DTK::SignedInteger => dt.is_signed_integer(),
                DTK::UnsignedInteger => dt.is_unsigned_integer(),
                DTK::Float => dt.is_float(),
                DTK::Decimal => dt.is_decimal(),

                DTK::Categorical => dt.is_categorical(),
                DTK::Enum => dt.is_enum(),

                DTK::Nested => dt.is_nested(),
                DTK::Array(kind_width) => {
                    #[cfg(feature = "dtype-array")]
                    {
                        matches!(dt, DataType::Array(_, width) if kind_width.is_none_or(|w| width == w))
                    }

                    #[cfg(not(feature = "dtype-array"))]
                    {
                        false
                    }
                },

                DTK::List => dt.is_list(),
                DTK::Struct => dt.is_struct(),

                DTK::Temporal => dt.is_temporal(),
                DTK::Datetime => dt.is_datetime(),
                DTK::Duration => dt.is_duration(),

                DTK::Object => dt.is_object(),
            };
            let is_kind = Scalar::from(is_kind);
            let is_kind = LiteralValue::Scalar(is_kind);
            let is_kind = AExpr::Literal(is_kind);

            (is_kind, get_literal_name().clone())
        },
        DTF::ElementBitSize(dt_expr) => {
            let dt = dt_expr.into_datatype(schema)?;

            let Some(num_bits) = dt.num_bits_per_element() else {
                polars_bail!(InvalidOperation: "`{dt}` does not have a static amount of bits per element");
            };
            let num_bits = Scalar::from(num_bits);
            let num_bits = LiteralValue::Scalar(num_bits);
            let num_bits = AExpr::Literal(num_bits);

            (num_bits, get_literal_name().clone())
        },
        DTF::Array(dt_expr, f) => {
            let (inner, width): (DataType, usize) = match dt_expr.into_datatype(schema)? {
                #[cfg(feature = "dtype-array")]
                DataType::Array(inner, width) => (*inner, width),
                dt => polars_bail!(InvalidOperation: "`{dt}` is not an Array"),
            };

            feature_gated!("dtype-array", {
                use crate::dsl::ArrayDataTypeFunction;
                let value = match f {
                    ArrayDataTypeFunction::Width => Scalar::from(width as u32),
                    ArrayDataTypeFunction::Dimensions => {
                        let mut dims = vec![width as u32];
                        let mut inner = inner;
                        while let DataType::Array(new_inner, width) = inner {
                            dims.push(width as u32);
                            inner = *new_inner;
                        }
                        Scalar::new(
                            DataType::List(Box::new(DataType::UInt32)),
                            AnyValue::List(Series::new(PlSmallStr::EMPTY, dims)),
                        )
                    },
                };
                let value = LiteralValue::Scalar(value);
                let value = AExpr::Literal(value);
                (value, get_literal_name().clone())
            })
        },
        DTF::Struct(dt_expr, f) => {
            let fields: Vec<Field> = match dt_expr.into_datatype(schema)? {
                #[cfg(feature = "dtype-struct")]
                DataType::Struct(fields) => fields,
                dt => polars_bail!(InvalidOperation: "`{dt}` is not an Struct"),
            };

            let value = match f {
                StructDataTypeFunction::NumFields => Scalar::from(fields.len() as u32),
                StructDataTypeFunction::FieldNames => Scalar::new(
                    DataType::List(Box::new(DataType::String)),
                    AnyValue::List(Series::new(
                        PlSmallStr::EMPTY,
                        fields
                            .iter()
                            .map(|f| f.name.as_str())
                            .collect::<Vec<&str>>(),
                    )),
                ),
                StructDataTypeFunction::FieldName { idx, raise_on_oob } => {
                    let offset = if idx < 0 {
                        usize::try_from(idx.abs_diff(0))
                            .ok()
                            .and_then(|idx| fields.len().checked_sub(idx))
                    } else {
                        usize::try_from(idx).ok()
                    };
                    let offset = offset.filter(|o| *o < fields.len());
                    match offset {
                        None => {
                            polars_ensure!(
                                !raise_on_oob,
                                InvalidOperation: "`struct` has {} fields, but field {idx} was requested",
                                fields.len()
                            );
                            Scalar::null(DataType::String)
                        },
                        Some(offset) => Scalar::from(fields[offset].name().clone()),
                    }
                },
                StructDataTypeFunction::FieldIndex {
                    name,
                    raise_on_missing,
                } => {
                    let index = fields.iter().position(|f| f.name() == &name);
                    match index {
                        None => {
                            polars_ensure!(!raise_on_missing, InvalidOperation: "`struct` does not contain field `{name}`");
                            Scalar::null(DataType::UInt32)
                        },
                        Some(index) => Scalar::from(index as u32),
                    }
                },
            };
            let value = LiteralValue::Scalar(value);
            let value = AExpr::Literal(value);
            (value, get_literal_name().clone())
        },
        DTF::Enum(dt_expr, f) => feature_gated!("dtype-categorical", {
            use crate::dsl::EnumDataTypeFunction;
            let revmap = match dt_expr.into_datatype(schema)? {
                DataType::Enum(revmap, _) => revmap,
                dt => polars_bail!(InvalidOperation: "`{dt}` is not an Enum"),
            };

            use polars_core::prelude::RevMapping;
            let revmap = revmap.unwrap();
            let RevMapping::Local(categories, _) = revmap.as_ref() else {
                unreachable!("enums cannot be global");
            };

            let value = match f {
                EnumDataTypeFunction::NumCategories => Scalar::from(categories.len() as u32),
                EnumDataTypeFunction::Categories => Scalar::new(
                    DataType::List(Box::new(DataType::String)),
                    AnyValue::List(
                        Series::from_chunk_and_dtype(
                            PlSmallStr::EMPTY,
                            categories.clone().boxed(),
                            &DataType::String,
                        )
                        .unwrap(),
                    ),
                ),
                EnumDataTypeFunction::GetCategory { idx, raise_on_oob } => {
                    let offset = if idx < 0 {
                        usize::try_from(idx.abs_diff(0))
                            .ok()
                            .and_then(|idx| categories.len().checked_sub(idx))
                    } else {
                        usize::try_from(idx).ok()
                    };
                    let offset = offset.filter(|o| *o < categories.len());
                    match offset {
                        None => {
                            polars_ensure!(
                                !raise_on_oob,
                                InvalidOperation: "`struct` has {} fields, but field {idx} was requested",
                                categories.len()
                            );

                            Scalar::null(DataType::String)
                        },
                        Some(offset) => {
                            Scalar::from(PlSmallStr::from_str(categories.get(offset).unwrap()))
                        },
                    }
                },
                EnumDataTypeFunction::IndexOfCategory {
                    cat,
                    raise_on_missing: raise,
                } => match categories.values_iter().position(|v| v == cat.as_str()) {
                    None if raise => {
                        polars_bail!(InvalidOperation: "`enum` does not contain category `{cat}`")
                    },
                    None => Scalar::null(DataType::UInt32),
                    Some(idx) => Scalar::from(idx as u32),
                },
            };
            let value = LiteralValue::Scalar(value);
            let value = AExpr::Literal(value);
            (value, get_literal_name().clone())
        }),
    })
}
