use polars_core::POOL;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::*;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ListToStructArgs {
    FixedWidth(Arc<[PlSmallStr]>),
    InferWidth {
        infer_field_strategy: ListToStructWidthStrategy,
        get_index_name: Option<NameGenerator>,
        /// If this is 0, it means unbounded.
        max_fields: usize,
    },
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ListToStructWidthStrategy {
    FirstNonNull,
    MaxWidth,
}

impl ListToStructArgs {
    pub fn get_output_dtype(&self, input_dtype: &DataType) -> PolarsResult<DataType> {
        let DataType::List(inner_dtype) = input_dtype else {
            polars_bail!(
                InvalidOperation:
                "attempted list to_struct on non-list dtype: {}",
                input_dtype
            );
        };
        let inner_dtype = inner_dtype.as_ref();

        match self {
            Self::FixedWidth(names) => Ok(DataType::Struct(
                names
                    .iter()
                    .map(|x| Field::new(x.clone(), inner_dtype.clone()))
                    .collect::<Vec<_>>(),
            )),
            Self::InferWidth {
                get_index_name,
                max_fields,
                ..
            } if *max_fields > 0 => {
                let get_index_name_func = get_index_name.as_ref().map_or(
                    &_default_struct_name_gen as &dyn Fn(usize) -> PlSmallStr,
                    |x| x.0.as_ref(),
                );
                Ok(DataType::Struct(
                    (0..*max_fields)
                        .map(|i| Field::new(get_index_name_func(i), inner_dtype.clone()))
                        .collect::<Vec<_>>(),
                ))
            },
            Self::InferWidth { .. } => Ok(DataType::Unknown(UnknownKind::Any)),
        }
    }

    fn det_n_fields(&self, ca: &ListChunked) -> usize {
        match self {
            Self::FixedWidth(v) => v.len(),
            Self::InferWidth {
                infer_field_strategy,
                max_fields,
                ..
            } => {
                let inferred = match infer_field_strategy {
                    ListToStructWidthStrategy::MaxWidth => {
                        let mut max = 0;

                        ca.downcast_iter().for_each(|arr| {
                            let offsets = arr.offsets().as_slice();
                            let mut last = offsets[0];
                            for o in &offsets[1..] {
                                let len = (*o - last) as usize;
                                max = std::cmp::max(max, len);
                                last = *o;
                            }
                        });
                        max
                    },
                    ListToStructWidthStrategy::FirstNonNull => {
                        let mut len = 0;
                        for arr in ca.downcast_iter() {
                            let offsets = arr.offsets().as_slice();
                            let mut last = offsets[0];
                            for o in &offsets[1..] {
                                len = (*o - last) as usize;
                                if len > 0 {
                                    break;
                                }
                                last = *o;
                            }
                            if len > 0 {
                                break;
                            }
                        }
                        len
                    },
                };

                if *max_fields > 0 {
                    inferred.min(*max_fields)
                } else {
                    inferred
                }
            },
        }
    }

    fn set_output_names(&self, columns: &mut [Series]) {
        match self {
            Self::FixedWidth(v) => {
                assert_eq!(columns.len(), v.len());

                for (c, name) in columns.iter_mut().zip(v.iter()) {
                    c.rename(name.clone());
                }
            },
            Self::InferWidth { get_index_name, .. } => {
                let get_index_name_func = get_index_name.as_ref().map_or(
                    &_default_struct_name_gen as &dyn Fn(usize) -> PlSmallStr,
                    |x| x.0.as_ref(),
                );

                for (i, c) in columns.iter_mut().enumerate() {
                    c.rename(get_index_name_func(i));
                }
            },
        }
    }
}

#[derive(Clone)]
pub struct NameGenerator(pub Arc<dyn Fn(usize) -> PlSmallStr + Send + Sync>);

impl NameGenerator {
    pub fn from_func(func: impl Fn(usize) -> PlSmallStr + Send + Sync + 'static) -> Self {
        Self(Arc::new(func))
    }
}

impl std::fmt::Debug for NameGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "list::to_struct::NameGenerator function at 0x{:016x}",
            self.0.as_ref() as *const _ as *const () as usize
        )
    }
}

impl Eq for NameGenerator {}

impl PartialEq for NameGenerator {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::hash::Hash for NameGenerator {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize)
    }
}

pub fn _default_struct_name_gen(idx: usize) -> PlSmallStr {
    format_pl_smallstr!("field_{idx}")
}

pub trait ToStruct: AsList {
    fn to_struct(&self, args: &ListToStructArgs) -> PolarsResult<StructChunked> {
        let ca = self.as_list();
        let n_fields = args.det_n_fields(ca);

        let mut fields = POOL.install(|| {
            (0..n_fields)
                .into_par_iter()
                .map(|i| ca.lst_get(i as i64, true))
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        args.set_output_names(&mut fields);

        StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
    }
}

impl ToStruct for ListChunked {}

#[cfg(feature = "serde")]
mod _serde_impl {
    use super::*;

    impl serde::Serialize for NameGenerator {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::Error;
            Err(S::Error::custom(
                "cannot serialize name generator function for to_struct, \
                consider passing a list of field names instead.",
            ))
        }
    }

    impl<'de> serde::Deserialize<'de> for NameGenerator {
        fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            use serde::de::Error;
            Err(D::Error::custom(
                "invalid data: attempted to deserialize list::to_struct::NameGenerator",
            ))
        }
    }
}
