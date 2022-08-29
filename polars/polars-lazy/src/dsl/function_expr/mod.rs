#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "round_series")]
mod clip;
mod fill_null;
#[cfg(feature = "is_in")]
mod is_in;
#[cfg(any(feature = "is_in", feature = "list"))]
mod list;
mod nan;
mod pow;
#[cfg(all(feature = "rolling_window", feature = "moment"))]
mod rolling;
#[cfg(feature = "row_hash")]
mod row_hash;
#[cfg(feature = "search_sorted")]
mod search_sorted;
mod shift_and_fill;
#[cfg(feature = "sign")]
mod sign;
#[cfg(feature = "strings")]
mod strings;
#[cfg(feature = "dtype-struct")]
mod struct_;
#[cfg(any(feature = "temporal", feature = "date_offset"))]
mod temporal;
#[cfg(feature = "trigonometry")]
mod trigonometry;

#[cfg(feature = "list")]
pub(super) use list::ListFunction;
use polars_core::prelude::*;
use polars_core::utils::slice_offsets;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(super) use self::nan::NanFunction;
#[cfg(feature = "strings")]
pub(super) use self::strings::StringFunction;
#[cfg(feature = "dtype-struct")]
pub(super) use self::struct_::StructFunction;
use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum FunctionExpr {
    NullCount,
    Pow,
    #[cfg(feature = "row_hash")]
    Hash(u64, u64, u64, u64),
    #[cfg(feature = "is_in")]
    IsIn,
    #[cfg(feature = "arg_where")]
    ArgWhere,
    #[cfg(feature = "search_sorted")]
    SearchSorted,
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    #[cfg(feature = "date_offset")]
    DateOffset(Duration),
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
    #[cfg(feature = "sign")]
    Sign,
    FillNull {
        super_type: DataType,
    },
    #[cfg(feature = "is_in")]
    ListContains,
    #[cfg(all(feature = "rolling_window", feature = "moment"))]
    // if we add more, make a sub enum
    RollingSkew {
        window_size: usize,
        bias: bool,
    },
    ShiftAndFill {
        periods: i64,
    },
    Nan(NanFunction),
    #[cfg(feature = "round_series")]
    Clip {
        min: Option<AnyValue<'static>>,
        max: Option<AnyValue<'static>>,
    },
    #[cfg(feature = "list")]
    ListExpr(ListFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "top_k")]
    TopK {
        k: usize,
        reverse: bool,
    },
}

#[cfg(feature = "trigonometry")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
    ArcSin,
    ArcCos,
    ArcTan,
    Sinh,
    Cosh,
    Tanh,
    ArcSinh,
    ArcCosh,
    ArcTanh,
}

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> Result<Field> {
        // set a dtype
        let with_dtype = |dtype: DataType| Ok(Field::new(fields[0].name(), dtype));

        // map a single dtype
        let map_dtype = |func: &dyn Fn(&DataType) -> DataType| {
            let dtype = func(fields[0].data_type());
            Ok(Field::new(fields[0].name(), dtype))
        };

        // map all dtypes
        #[cfg(feature = "list")]
        let map_dtypes = |func: &dyn Fn(&[&DataType]) -> DataType| {
            let mut fld = fields[0].clone();
            let dtypes = fields.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
            let new_type = func(&dtypes);
            fld.coerce(new_type);
            Ok(fld)
        };

        #[cfg(any(feature = "rolling_window", feature = "trigonometry"))]
        // set float supertype
        let float_dtype = || {
            map_dtype(&|dtype| match dtype {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        };

        // map to same type
        let same_type = || map_dtype(&|dtype| dtype.clone());

        // get supertype of all types
        let super_type = || {
            let mut first = fields[0].clone();
            let mut st = first.data_type().clone();
            for field in &fields[1..] {
                st = get_supertype(&st, field.data_type())?
            }
            first.coerce(st);
            Ok(first)
        };

        // inner super type of lists
        #[cfg(feature = "list")]
        let inner_super_type_list = || {
            map_dtypes(&|dts| {
                let mut super_type_inner = None;

                for dt in dts {
                    match dt {
                        DataType::List(inner) => match super_type_inner {
                            None => super_type_inner = Some(*inner.clone()),
                            Some(st_inner) => {
                                super_type_inner = get_supertype(&st_inner, inner).ok()
                            }
                        },
                        dt => match super_type_inner {
                            None => super_type_inner = Some((*dt).clone()),
                            Some(st_inner) => super_type_inner = get_supertype(&st_inner, dt).ok(),
                        },
                    }
                }
                DataType::List(Box::new(super_type_inner.unwrap()))
            })
        };

        use FunctionExpr::*;
        match self {
            NullCount => with_dtype(IDX_DTYPE),
            Pow => super_type(),
            #[cfg(feature = "row_hash")]
            Hash(..) => with_dtype(DataType::UInt64),
            #[cfg(feature = "is_in")]
            IsIn => with_dtype(DataType::Boolean),
            #[cfg(feature = "arg_where")]
            ArgWhere => with_dtype(IDX_DTYPE),
            #[cfg(feature = "search_sorted")]
            SearchSorted => with_dtype(IDX_DTYPE),
            #[cfg(feature = "strings")]
            StringExpr(s) => {
                use StringFunction::*;
                match s {
                    Contains { .. } | EndsWith(_) | StartsWith(_) => with_dtype(DataType::Boolean),
                    Extract { .. } => same_type(),
                    ExtractAll(_) => with_dtype(DataType::List(Box::new(DataType::Utf8))),
                    CountMatch(_) => with_dtype(DataType::UInt32),
                    #[cfg(feature = "string_justify")]
                    Zfill { .. } | LJust { .. } | RJust { .. } => same_type(),
                    #[cfg(feature = "temporal")]
                    Strptime(options) => with_dtype(options.date_dtype.clone()),
                    #[cfg(feature = "concat_str")]
                    Concat(_) => with_dtype(DataType::Utf8),
                    #[cfg(feature = "regex")]
                    Replace { .. } => with_dtype(DataType::Utf8),
                    Uppercase | Lowercase => with_dtype(DataType::Utf8),
                }
            }

            #[cfg(feature = "date_offset")]
            DateOffset(_) => same_type(),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => float_dtype(),
            #[cfg(feature = "sign")]
            Sign => with_dtype(DataType::Int64),
            FillNull { super_type, .. } => with_dtype(super_type.clone()),
            #[cfg(feature = "is_in")]
            ListContains => with_dtype(DataType::Boolean),
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { .. } => float_dtype(),
            ShiftAndFill { .. } => same_type(),
            Nan(n) => n.get_field(fields),
            #[cfg(feature = "round_series")]
            Clip { .. } => same_type(),
            #[cfg(feature = "list")]
            ListExpr(l) => {
                use ListFunction::*;
                match l {
                    Concat => inner_super_type_list(),
                }
            }
            #[cfg(feature = "dtype-struct")]
            StructExpr(s) => {
                use StructFunction::*;
                match s {
                    FieldByIndex(index) => {
                        let (index, _) = slice_offsets(*index, 0, fields.len());
                        fields.get(index).cloned().ok_or_else(|| {
                            PolarsError::ComputeError(
                                "index out of bounds in 'struct.field'".into(),
                            )
                        })
                    }
                    FieldByName(name) => {
                        if let DataType::Struct(flds) = &fields[0].dtype {
                            let fld = flds
                                .iter()
                                .find(|fld| fld.name() == name.as_ref())
                                .ok_or_else(|| PolarsError::NotFound(name.as_ref().to_string()))?;
                            Ok(fld.clone())
                        } else {
                            Err(PolarsError::NotFound(name.as_ref().to_string()))
                        }
                    }
                }
            }
            #[cfg(feature = "top_k")]
            TopK { .. } => same_type(),
        }
    }
}

macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };
}

// Fn(&[Series], args)
// all expression arguments are in the slice.
// the first element is the root expression.
macro_rules! map_as_slice {
    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// Fn(&Series)
#[macro_export(super)]
macro_rules! map_without_args {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// FnOnce(Series)
// FnOnce(Series, args)
#[macro_export(super)]
macro_rules! map_owned {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// Fn(&Series, args)
macro_rules! map_with_args {
    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

impl From<FunctionExpr> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: FunctionExpr) -> Self {
        use FunctionExpr::*;
        match func {
            NullCount => {
                let f = |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(Series::new(s.name(), [s.null_count() as IdxSize]))
                };
                wrap!(f)
            }
            Pow => {
                wrap!(pow::pow)
            }
            #[cfg(feature = "row_hash")]
            Hash(k0, k1, k2, k3) => {
                map_with_args!(row_hash::row_hash, k0, k1, k2, k3)
            }
            #[cfg(feature = "is_in")]
            IsIn => {
                wrap!(is_in::is_in)
            }
            #[cfg(feature = "arg_where")]
            ArgWhere => {
                wrap!(arg_where::arg_where)
            }
            #[cfg(feature = "search_sorted")]
            SearchSorted => {
                wrap!(search_sorted::search_sorted_impl)
            }
            #[cfg(feature = "strings")]
            StringExpr(s) => s.into(),

            #[cfg(feature = "date_offset")]
            DateOffset(offset) => {
                map_owned!(temporal::date_offset, offset)
            }
            #[cfg(feature = "trigonometry")]
            Trigonometry(trig_function) => {
                map_with_args!(trigonometry::apply_trigonometric_function, trig_function)
            }
            #[cfg(feature = "sign")]
            Sign => {
                map_without_args!(sign::sign)
            }
            FillNull { super_type } => {
                map_as_slice!(fill_null::fill_null, &super_type)
            }

            #[cfg(feature = "is_in")]
            ListContains => {
                wrap!(list::contains)
            }
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { window_size, bias } => {
                map_with_args!(rolling::rolling_skew, window_size, bias)
            }
            ShiftAndFill { periods } => {
                map_as_slice!(shift_and_fill::shift_and_fill, periods)
            }
            Nan(n) => n.into(),
            #[cfg(feature = "round_series")]
            Clip { min, max } => {
                map_owned!(clip::clip, min.clone(), max.clone())
            }
            #[cfg(feature = "list")]
            ListExpr(lf) => {
                use ListFunction::*;
                match lf {
                    Concat => wrap!(list::concat),
                }
            }
            #[cfg(feature = "dtype-struct")]
            StructExpr(sf) => {
                use StructFunction::*;
                match sf {
                    FieldByIndex(index) => map_with_args!(struct_::get_by_index, index),
                    FieldByName(name) => map_with_args!(struct_::get_by_name, name.clone()),
                }
            }
            #[cfg(feature = "top_k")]
            TopK { k, reverse } => {
                map_with_args!(top_k, k, reverse)
            }
        }
    }
}

#[cfg(feature = "strings")]
impl From<StringFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StringFunction) -> Self {
        use StringFunction::*;
        match func {
            Contains { pat, literal } => {
                map_with_args!(strings::contains, &pat, literal)
            }
            EndsWith(sub) => {
                map_with_args!(strings::ends_with, &sub)
            }
            StartsWith(sub) => {
                map_with_args!(strings::starts_with, &sub)
            }
            Extract { pat, group_index } => {
                map_with_args!(strings::extract, &pat, group_index)
            }
            ExtractAll(pat) => {
                map_with_args!(strings::extract_all, &pat)
            }
            CountMatch(pat) => {
                map_with_args!(strings::count_match, &pat)
            }
            #[cfg(feature = "string_justify")]
            Zfill(alignment) => {
                map_with_args!(strings::zfill, alignment)
            }
            #[cfg(feature = "string_justify")]
            LJust { width, fillchar } => {
                map_with_args!(strings::ljust, width, fillchar)
            }
            #[cfg(feature = "string_justify")]
            RJust { width, fillchar } => {
                map_with_args!(strings::rjust, width, fillchar)
            }
            #[cfg(feature = "temporal")]
            Strptime(options) => {
                map_with_args!(strings::strptime, &options)
            }
            #[cfg(feature = "concat_str")]
            Concat(delimiter) => map_with_args!(strings::concat, &delimiter),
            #[cfg(feature = "regex")]
            Replace { all, literal } => map_as_slice!(strings::replace, literal, all),
            Uppercase => map_without_args!(strings::uppercase),
            Lowercase => map_without_args!(strings::lowercase),
        }
    }
}
