use std::any::Any;

use parquet_format_safe::ColumnIndex;

use crate::parquet_bridge::BoundaryOrder;
use crate::schema::types::PrimitiveType;
use crate::{error::Error, schema::types::PhysicalType, types::NativeType};

/// Trait object representing a [`ColumnIndex`] in Rust's native format.
///
/// See [`NativeIndex`], [`ByteIndex`] and [`FixedLenByteIndex`] for concrete implementations.
pub trait Index: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;

    fn physical_type(&self) -> &PhysicalType;
}

impl PartialEq for dyn Index + '_ {
    fn eq(&self, that: &dyn Index) -> bool {
        equal(self, that)
    }
}

impl Eq for dyn Index + '_ {}

fn equal(lhs: &dyn Index, rhs: &dyn Index) -> bool {
    if lhs.physical_type() != rhs.physical_type() {
        return false;
    }

    match lhs.physical_type() {
        PhysicalType::Boolean => {
            lhs.as_any().downcast_ref::<BooleanIndex>().unwrap()
                == rhs.as_any().downcast_ref::<BooleanIndex>().unwrap()
        }
        PhysicalType::Int32 => {
            lhs.as_any().downcast_ref::<NativeIndex<i32>>().unwrap()
                == rhs.as_any().downcast_ref::<NativeIndex<i32>>().unwrap()
        }
        PhysicalType::Int64 => {
            lhs.as_any().downcast_ref::<NativeIndex<i64>>().unwrap()
                == rhs.as_any().downcast_ref::<NativeIndex<i64>>().unwrap()
        }
        PhysicalType::Int96 => {
            lhs.as_any()
                .downcast_ref::<NativeIndex<[u32; 3]>>()
                .unwrap()
                == rhs
                    .as_any()
                    .downcast_ref::<NativeIndex<[u32; 3]>>()
                    .unwrap()
        }
        PhysicalType::Float => {
            lhs.as_any().downcast_ref::<NativeIndex<f32>>().unwrap()
                == rhs.as_any().downcast_ref::<NativeIndex<f32>>().unwrap()
        }
        PhysicalType::Double => {
            lhs.as_any().downcast_ref::<NativeIndex<f64>>().unwrap()
                == rhs.as_any().downcast_ref::<NativeIndex<f64>>().unwrap()
        }
        PhysicalType::ByteArray => {
            lhs.as_any().downcast_ref::<ByteIndex>().unwrap()
                == rhs.as_any().downcast_ref::<ByteIndex>().unwrap()
        }
        PhysicalType::FixedLenByteArray(_) => {
            lhs.as_any().downcast_ref::<FixedLenByteIndex>().unwrap()
                == rhs.as_any().downcast_ref::<FixedLenByteIndex>().unwrap()
        }
    }
}

/// An index of a column of [`NativeType`] physical representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeIndex<T: NativeType> {
    /// The primitive type
    pub primitive_type: PrimitiveType,
    /// The indexes, one item per page
    pub indexes: Vec<PageIndex<T>>,
    /// the order
    pub boundary_order: BoundaryOrder,
}

impl<T: NativeType> NativeIndex<T> {
    /// Creates a new [`NativeIndex`]
    pub(crate) fn try_new(
        index: ColumnIndex,
        primitive_type: PrimitiveType,
    ) -> Result<Self, Error> {
        let len = index.min_values.len();

        let null_counts = index
            .null_counts
            .map(|x| x.into_iter().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![None; len]);

        let indexes = index
            .min_values
            .iter()
            .zip(index.max_values.into_iter())
            .zip(index.null_pages.into_iter())
            .zip(null_counts.into_iter())
            .map(|(((min, max), is_null), null_count)| {
                let (min, max) = if is_null {
                    (None, None)
                } else {
                    let min = min.as_slice().try_into()?;
                    let max = max.as_slice().try_into()?;
                    (Some(T::from_le_bytes(min)), Some(T::from_le_bytes(max)))
                };
                Ok(PageIndex {
                    min,
                    max,
                    null_count,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            primitive_type,
            indexes,
            boundary_order: index.boundary_order.try_into()?,
        })
    }
}

/// The index of a page, containing the min and max values of the page.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PageIndex<T> {
    /// The minimum value in the page. It is None when all values are null
    pub min: Option<T>,
    /// The maximum value in the page. It is None when all values are null
    pub max: Option<T>,
    /// The number of null values in the page
    pub null_count: Option<i64>,
}

impl<T: NativeType> Index for NativeIndex<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &T::TYPE
    }
}

/// An index of a column of bytes physical type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteIndex {
    /// The [`PrimitiveType`].
    pub primitive_type: PrimitiveType,
    /// The indexes, one item per page
    pub indexes: Vec<PageIndex<Vec<u8>>>,
    pub boundary_order: BoundaryOrder,
}

impl ByteIndex {
    pub(crate) fn try_new(
        index: ColumnIndex,
        primitive_type: PrimitiveType,
    ) -> Result<Self, Error> {
        let len = index.min_values.len();

        let null_counts = index
            .null_counts
            .map(|x| x.into_iter().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![None; len]);

        let indexes = index
            .min_values
            .into_iter()
            .zip(index.max_values.into_iter())
            .zip(index.null_pages.into_iter())
            .zip(null_counts.into_iter())
            .map(|(((min, max), is_null), null_count)| {
                let (min, max) = if is_null {
                    (None, None)
                } else {
                    (Some(min), Some(max))
                };
                Ok(PageIndex {
                    min,
                    max,
                    null_count,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            primitive_type,
            indexes,
            boundary_order: index.boundary_order.try_into()?,
        })
    }
}

impl Index for ByteIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &PhysicalType::ByteArray
    }
}

/// An index of a column of fixed len byte physical type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedLenByteIndex {
    /// The [`PrimitiveType`].
    pub primitive_type: PrimitiveType,
    /// The indexes, one item per page
    pub indexes: Vec<PageIndex<Vec<u8>>>,
    pub boundary_order: BoundaryOrder,
}

impl FixedLenByteIndex {
    pub(crate) fn try_new(
        index: ColumnIndex,
        primitive_type: PrimitiveType,
    ) -> Result<Self, Error> {
        let len = index.min_values.len();

        let null_counts = index
            .null_counts
            .map(|x| x.into_iter().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![None; len]);

        let indexes = index
            .min_values
            .into_iter()
            .zip(index.max_values.into_iter())
            .zip(index.null_pages.into_iter())
            .zip(null_counts.into_iter())
            .map(|(((min, max), is_null), null_count)| {
                let (min, max) = if is_null {
                    (None, None)
                } else {
                    (Some(min), Some(max))
                };
                Ok(PageIndex {
                    min,
                    max,
                    null_count,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            primitive_type,
            indexes,
            boundary_order: index.boundary_order.try_into()?,
        })
    }
}

impl Index for FixedLenByteIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &self.primitive_type.physical_type
    }
}

/// An index of a column of boolean physical type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BooleanIndex {
    /// The indexes, one item per page
    pub indexes: Vec<PageIndex<bool>>,
    pub boundary_order: BoundaryOrder,
}

impl BooleanIndex {
    pub(crate) fn try_new(index: ColumnIndex) -> Result<Self, Error> {
        let len = index.min_values.len();

        let null_counts = index
            .null_counts
            .map(|x| x.into_iter().map(Some).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![None; len]);

        let indexes = index
            .min_values
            .into_iter()
            .zip(index.max_values.into_iter())
            .zip(index.null_pages.into_iter())
            .zip(null_counts.into_iter())
            .map(|(((min, max), is_null), null_count)| {
                let (min, max) = if is_null {
                    (None, None)
                } else {
                    let min = min[0] == 1;
                    let max = max[0] == 1;
                    (Some(min), Some(max))
                };
                Ok(PageIndex {
                    min,
                    max,
                    null_count,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            indexes,
            boundary_order: index.boundary_order.try_into()?,
        })
    }
}

impl Index for BooleanIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn physical_type(&self) -> &PhysicalType {
        &PhysicalType::Boolean
    }
}
