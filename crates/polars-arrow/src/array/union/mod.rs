use polars_error::{polars_bail, polars_err, PolarsResult};

use super::{new_empty_array, new_null_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::{ArrowDataType, Field, UnionMode};
use crate::scalar::{new_scalar, Scalar};

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;

type UnionComponents<'a> = (&'a [Field], Option<&'a [i32]>, UnionMode);

/// [`UnionArray`] represents an array whose each slot can contain different values.
///
// How to read a value at slot i:
// ```
// let index = self.types()[i] as usize;
// let field = self.fields()[index];
// let offset = self.offsets().map(|x| x[index]).unwrap_or(i);
// let field = field.as_any().downcast to correct type;
// let value = field.value(offset);
// ```
#[derive(Clone)]
pub struct UnionArray {
    // Invariant: every item in `types` is `> 0 && < fields.len()`
    types: Buffer<i8>,
    // Invariant: `map.len() == fields.len()`
    // Invariant: every item in `map` is `> 0 && < fields.len()`
    map: Option<[usize; 127]>,
    fields: Vec<Box<dyn Array>>,
    // Invariant: when set, `offsets.len() == types.len()`
    offsets: Option<Buffer<i32>>,
    data_type: ArrowDataType,
    offset: usize,
}

impl UnionArray {
    /// Returns a new [`UnionArray`].
    /// # Errors
    /// This function errors iff:
    /// * `data_type`'s physical type is not [`crate::datatypes::PhysicalType::Union`].
    /// * the fields's len is different from the `data_type`'s children's length
    /// * The number of `fields` is larger than `i8::MAX`
    /// * any of the values's data type is different from its corresponding children' data type
    pub fn try_new(
        data_type: ArrowDataType,
        types: Buffer<i8>,
        fields: Vec<Box<dyn Array>>,
        offsets: Option<Buffer<i32>>,
    ) -> PolarsResult<Self> {
        let (f, ids, mode) = Self::try_get_all(&data_type)?;

        if f.len() != fields.len() {
            polars_bail!(ComputeError: "the number of `fields` must equal the number of children fields in DataType::Union")
        };
        let number_of_fields: i8 = fields.len().try_into().map_err(
            |_| polars_err!(ComputeError: "the number of `fields` cannot be larger than i8::MAX"),
        )?;

        f
            .iter().map(|a| a.data_type())
            .zip(fields.iter().map(|a| a.data_type()))
            .enumerate()
            .try_for_each(|(index, (data_type, child))| {
                if data_type != child {
                    polars_bail!(ComputeError:
                        "the children DataTypes of a UnionArray must equal the children data types.
                         However, the field {index} has data type {data_type:?} but the value has data type {child:?}"
                    )
                } else {
                    Ok(())
                }
            })?;

        if let Some(offsets) = &offsets {
            if offsets.len() != types.len() {
                polars_bail!(ComputeError:
                "in a UnionArray, the offsets' length must be equal to the number of types"
                )
            }
        }
        if offsets.is_none() != mode.is_sparse() {
            polars_bail!(ComputeError:
            "in a sparse UnionArray, the offsets must be set (and vice-versa)",
                )
        }

        // build hash
        let map = if let Some(&ids) = ids.as_ref() {
            if ids.len() != fields.len() {
                polars_bail!(ComputeError:
                "in a union, when the ids are set, their length must be equal to the number of fields",
                )
            }

            // example:
            // * types = [5, 7, 5, 7, 7, 7, 5, 7, 7, 5, 5]
            // * ids = [5, 7]
            // => hash = [0, 0, 0, 0, 0, 0, 1, 0, ...]
            let mut hash = [0; 127];

            for (pos, &id) in ids.iter().enumerate() {
                if !(0..=127).contains(&id) {
                    polars_bail!(ComputeError:
                        "in a union, when the ids are set, every id must belong to [0, 128[",
                    )
                }
                hash[id as usize] = pos;
            }

            types.iter().try_for_each(|&type_| {
                if type_ < 0 {
                    polars_bail!(ComputeError:
                        "in a union, when the ids are set, every type must be >= 0"
                    )
                }
                let id = hash[type_ as usize];
                if id >= fields.len() {
                    polars_bail!(ComputeError:
    "in a union, when the ids are set, each id must be smaller than the number of fields."
                    )
                } else {
                    Ok(())
                }
            })?;

            Some(hash)
        } else {
            // SAFETY: every type in types is smaller than number of fields
            let mut is_valid = true;
            for &type_ in types.iter() {
                if type_ < 0 || type_ >= number_of_fields {
                    is_valid = false
                }
            }
            if !is_valid {
                polars_bail!(ComputeError:
                    "every type in `types` must be larger than 0 and smaller than the number of fields.",
                )
            }

            None
        };

        Ok(Self {
            data_type,
            map,
            fields,
            offsets,
            types,
            offset: 0,
        })
    }

    /// Returns a new [`UnionArray`].
    /// # Panics
    /// This function panics iff:
    /// * `data_type`'s physical type is not [`crate::datatypes::PhysicalType::Union`].
    /// * the fields's len is different from the `data_type`'s children's length
    /// * any of the values's data type is different from its corresponding children' data type
    pub fn new(
        data_type: ArrowDataType,
        types: Buffer<i8>,
        fields: Vec<Box<dyn Array>>,
        offsets: Option<Buffer<i32>>,
    ) -> Self {
        Self::try_new(data_type, types, fields, offsets).unwrap()
    }

    /// Creates a new null [`UnionArray`].
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        if let ArrowDataType::Union(f, _, mode) = &data_type {
            let fields = f
                .iter()
                .map(|x| new_null_array(x.data_type().clone(), length))
                .collect();

            let offsets = if mode.is_sparse() {
                None
            } else {
                Some((0..length as i32).collect::<Vec<_>>().into())
            };

            // all from the same field
            let types = vec![0i8; length].into();

            Self::new(data_type, types, fields, offsets)
        } else {
            panic!("Union struct must be created with the corresponding Union DataType")
        }
    }

    /// Creates a new empty [`UnionArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        if let ArrowDataType::Union(f, _, mode) = data_type.to_logical_type() {
            let fields = f
                .iter()
                .map(|x| new_empty_array(x.data_type().clone()))
                .collect();

            let offsets = if mode.is_sparse() {
                None
            } else {
                Some(Buffer::default())
            };

            Self {
                data_type,
                map: None,
                fields,
                offsets,
                types: Buffer::new(),
                offset: 0,
            }
        } else {
            panic!("Union struct must be created with the corresponding Union DataType")
        }
    }
}

impl UnionArray {
    /// Returns a slice of this [`UnionArray`].
    /// # Implementation
    /// This operation is `O(F)` where `F` is the number of fields.
    /// # Panic
    /// This function panics iff `offset + length > self.len()`.
    #[inline]
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new array cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Returns a slice of this [`UnionArray`].
    /// # Implementation
    /// This operation is `O(F)` where `F` is the number of fields.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    #[inline]
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.len());

        self.types.slice_unchecked(offset, length);
        if let Some(offsets) = self.offsets.as_mut() {
            offsets.slice_unchecked(offset, length)
        }
        self.offset += offset;
    }

    impl_sliced!();
    impl_into_array!();
}

impl UnionArray {
    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// The optional offsets.
    pub fn offsets(&self) -> Option<&Buffer<i32>> {
        self.offsets.as_ref()
    }

    /// The fields.
    pub fn fields(&self) -> &Vec<Box<dyn Array>> {
        &self.fields
    }

    /// The types.
    pub fn types(&self) -> &Buffer<i8> {
        &self.types
    }

    #[inline]
    unsafe fn field_slot_unchecked(&self, index: usize) -> usize {
        self.offsets()
            .as_ref()
            .map(|x| *x.get_unchecked(index) as usize)
            .unwrap_or(index + self.offset)
    }

    /// Returns the index and slot of the field to select from `self.fields`.
    #[inline]
    pub fn index(&self, index: usize) -> (usize, usize) {
        assert!(index < self.len());
        unsafe { self.index_unchecked(index) }
    }

    /// Returns the index and slot of the field to select from `self.fields`.
    /// The first value is guaranteed to be `< self.fields().len()`
    ///
    /// # Safety
    /// This function is safe iff `index < self.len`.
    #[inline]
    pub unsafe fn index_unchecked(&self, index: usize) -> (usize, usize) {
        debug_assert!(index < self.len());
        // SAFETY: assumption of the function
        let type_ = unsafe { *self.types.get_unchecked(index) };
        // SAFETY: assumption of the struct
        let type_ = self
            .map
            .as_ref()
            .map(|map| unsafe { *map.get_unchecked(type_ as usize) })
            .unwrap_or(type_ as usize);
        // SAFETY: assumption of the function
        let index = self.field_slot_unchecked(index);
        (type_, index)
    }

    /// Returns the slot `index` as a [`Scalar`].
    /// # Panics
    /// iff `index >= self.len()`
    pub fn value(&self, index: usize) -> Box<dyn Scalar> {
        assert!(index < self.len());
        unsafe { self.value_unchecked(index) }
    }

    /// Returns the slot `index` as a [`Scalar`].
    ///
    /// # Safety
    /// This function is safe iff `i < self.len`.
    pub unsafe fn value_unchecked(&self, index: usize) -> Box<dyn Scalar> {
        debug_assert!(index < self.len());
        let (type_, index) = self.index_unchecked(index);
        // SAFETY: assumption of the struct
        debug_assert!(type_ < self.fields.len());
        let field = self.fields.get_unchecked(type_).as_ref();
        new_scalar(field, index)
    }
}

impl Array for UnionArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        None
    }

    fn with_validity(&self, _: Option<Bitmap>) -> Box<dyn Array> {
        panic!("cannot set validity of a union array")
    }
}

impl UnionArray {
    fn try_get_all(data_type: &ArrowDataType) -> PolarsResult<UnionComponents> {
        match data_type.to_logical_type() {
            ArrowDataType::Union(fields, ids, mode) => {
                Ok((fields, ids.as_ref().map(|x| x.as_ref()), *mode))
            },
            _ => polars_bail!(ComputeError:
                "The UnionArray requires a logical type of DataType::Union",
            ),
        }
    }

    fn get_all(data_type: &ArrowDataType) -> (&[Field], Option<&[i32]>, UnionMode) {
        Self::try_get_all(data_type).unwrap()
    }

    /// Returns all fields from [`ArrowDataType::Union`].
    /// # Panic
    /// Panics iff `data_type`'s logical type is not [`ArrowDataType::Union`].
    pub fn get_fields(data_type: &ArrowDataType) -> &[Field] {
        Self::get_all(data_type).0
    }

    /// Returns whether the [`ArrowDataType::Union`] is sparse or not.
    /// # Panic
    /// Panics iff `data_type`'s logical type is not [`ArrowDataType::Union`].
    pub fn is_sparse(data_type: &ArrowDataType) -> bool {
        Self::get_all(data_type).2.is_sparse()
    }
}

impl Splitable for UnionArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_types, rhs_types) = unsafe { self.types.split_at_unchecked(offset) };
        let (lhs_offsets, rhs_offsets) = self.offsets.as_ref().map_or((None, None), |v| {
            let (lhs, rhs) = unsafe { v.split_at_unchecked(offset) };
            (Some(lhs), Some(rhs))
        });

        (
            Self {
                types: lhs_types,
                map: self.map,
                fields: self.fields.clone(),
                offsets: lhs_offsets,
                data_type: self.data_type.clone(),
                offset: self.offset,
            },
            Self {
                types: rhs_types,
                map: self.map,
                fields: self.fields.clone(),
                offsets: rhs_offsets,
                data_type: self.data_type.clone(),
                offset: self.offset + offset,
            },
        )
    }
}
