use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum IRUnionFunction {
    Tag,
    Field(PlSmallStr),
    /// Every row holds the input as the variant `variant` of the union `dtype`.
    Value {
        dtype: DataType,
        variant: PlSmallStr,
    },
}

impl IRUnionFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        if let Self::Value { dtype, variant } = self {
            return mapper.try_map_dtype(|input| {
                let DataType::Union(fields) = dtype else {
                    unreachable!()
                };
                let expected = fields[fields.try_index_of(variant)?].dtype();
                // Type coercion casts these to the variant's dtype.
                polars_ensure!(
                    input == expected || input.is_null() || literal_fits(input, expected),
                    SchemaMismatch: "variant `{variant}` of `{dtype}` has dtype `{expected}`, \
                    got a value of dtype `{input}`; cast the value first"
                );
                Ok(dtype.clone())
            });
        }
        mapper.try_map_dtype(|dtype| {
            let DataType::Union(fields) = dtype else {
                polars_bail!(InvalidOperation: "`{self}` requires a Union dtype, got `{dtype}`");
            };
            Ok(match self {
                Self::Tag => fields.tag_dtype(),
                Self::Field(name) => fields[fields.try_index_of(name)?].dtype().clone(),
                Self::Value { .. } => unreachable!(),
            })
        })
    }

    pub fn function_options(&self) -> FunctionOptions {
        FunctionOptions::elementwise()
    }
}

impl Display for IRUnionFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tag => write!(f, "union.tag"),
            Self::Field(name) => write!(f, "union.field({name})"),
            Self::Value { variant, .. } => write!(f, "union.value({variant})"),
        }
    }
}

impl From<IRUnionFunction> for IRFunctionExpr {
    fn from(value: IRUnionFunction) -> Self {
        Self::UnionExpr(value)
    }
}

/// Whether a dynamic literal of dtype `input` can take the dtype `to` without changing kind.
pub(crate) fn literal_fits(input: &DataType, to: &DataType) -> bool {
    match input {
        DataType::Unknown(UnknownKind::Int(_)) => to.is_primitive_numeric() || to.is_decimal(),
        DataType::Unknown(UnknownKind::Float) => to.is_float() || to.is_decimal(),
        DataType::Unknown(UnknownKind::Str) => {
            to.is_string() || to.is_categorical() || to.is_enum()
        },
        _ => false,
    }
}

/// The output dtype of `union.match`, from a label and the dtype of each arm and the default.
///
/// The dtypes must be equal, except that `Null` takes the common dtype and a dynamic literal one
/// of its kind.
pub(crate) fn union_match_dtype(branches: &[(String, DataType)]) -> PolarsResult<DataType> {
    let mismatch = |(a, a_dt): &(String, DataType), (b, b_dt): &(String, DataType)| {
        polars_err!(
            SchemaMismatch: "the arms of `union.match` must have the same dtype, \
            but the arm for {a} gives `{a_dt}` and the arm for {b} gives `{b_dt}`"
        )
    };
    if let Some(first) = branches
        .iter()
        .find(|(_, dt)| !dt.is_null() && !dt.is_unknown())
    {
        for b in branches {
            let dt = &b.1;
            if !(dt.is_null() || dt == &first.1 || literal_fits(dt, &first.1)) {
                return Err(mismatch(first, b));
            }
        }
        return Ok(first.1.clone());
    }
    let mut literals = branches.iter().filter(|(_, dt)| dt.is_unknown());
    let Some(first) = literals.next() else {
        return Ok(DataType::Null);
    };
    let mut dtype = first.1.clone();
    for b in literals {
        dtype =
            polars_core::utils::try_get_supertype(&dtype, &b.1).map_err(|_| mismatch(first, b))?;
    }
    dtype.materialize_unknown(false)
}
