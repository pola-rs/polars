//! Building a union from one variant, and reading the variant and the values of a union.
use polars_arrow::array::{ChildPositions, LargeUnionArray, PrimitiveArray, new_empty_array};
use polars_buffer::Buffer;
use polars_compute::gather::gather_child;

use crate::prelude::*;

impl UnionFields {
    /// The dtype of [`UnionChunked::tag`]: an `Enum` of the variant names, in variant order.
    #[cfg(feature = "dtype-categorical")]
    pub fn tag_dtype(&self) -> DataType {
        let names = self.iter().map(|f| f.name().as_str());
        // Unique by construction, and at most `MAX_UNION_VARIANTS` fit in `u8`.
        DataType::from_frozen_categories(FrozenCategories::new(names).unwrap())
    }

    /// The index of the variant called `name`.
    pub fn try_index_of(&self, name: &str) -> PolarsResult<usize> {
        self.iter().position(|f| f.name() == name).ok_or_else(|| {
            polars_err!(
                StructFieldNotFound: "union has no variant `{name}`; the variants are [{}]",
                self.iter().map(|f| format!("`{}`", f.name())).collect::<Vec<_>>().join(", ")
            )
        })
    }
}

impl UnionChunked {
    /// A union of `dtype` in which every row holds `values` as the variant at index `variant`;
    /// null values give null rows.
    pub fn from_variant(dtype: &DataType, variant: usize, values: &Series) -> PolarsResult<Self> {
        let DataType::Union(fields) = dtype else {
            polars_bail!(InvalidOperation: "expected a Union dtype, got `{dtype}`");
        };
        let field = &fields[variant];
        polars_ensure!(
            values.dtype() == field.dtype(),
            SchemaMismatch: "variant `{}` of `{dtype}` has dtype `{}`, got `{}`",
            field.name(), field.dtype(), values.dtype()
        );
        let arrow_dtype = dtype.to_physical_arrow(CompatLevel::newest());
        let empty: Vec<ArrayRef> = fields
            .iter()
            .map(|f| new_empty_array(f.dtype().to_physical_arrow(CompatLevel::newest())))
            .collect();
        let chunks = values
            .chunks()
            .iter()
            .map(|values| {
                let n = values.len();
                let types: Buffer<i8> = vec![variant as i8; n].into();
                let offsets: Buffer<i64> = (0..n as i64).collect();
                let validity = values.validity().cloned();
                let mut children = empty.clone();
                children[variant] = values.clone();
                // SAFETY: slot `i` points at value `i` of the variant's child, which has the
                // variant's physical dtype, and null values are null rows.
                unsafe {
                    LargeUnionArray::new_unchecked(
                        arrow_dtype.clone(),
                        types,
                        offsets,
                        children,
                        validity,
                    )
                }
                .boxed()
            })
            .collect();
        // SAFETY: the chunks have the physical arrow dtype of `dtype`.
        Ok(unsafe {
            Self::from_chunks_and_dtype_unchecked(values.name().clone(), chunks, dtype.clone())
        })
    }

    /// The variant of every row, as an `Enum` of the variant names; null for null rows.
    #[cfg(feature = "dtype-categorical")]
    pub fn tag(&self) -> Series {
        let dtype = self.union_fields().tag_dtype();
        let chunks = self.downcast_iter().map(|arr| {
            // Tags are non-negative, so the cast keeps them.
            let tags = arr.types().iter().map(|&t| t as u8).collect::<Vec<_>>();
            PrimitiveArray::from_vec(tags).with_validity(arr.validity().cloned())
        });
        let ca = UInt8Chunked::from_chunk_iter(self.name().clone(), chunks);
        // SAFETY: every tag is the index of a variant, so a valid category of `dtype`.
        unsafe { CategoricalChunked::<Categorical8Type>::from_cats_and_dtype_unchecked(ca, dtype) }
            .into_series()
    }

    /// The values of `variant`: null for null rows and rows of other variants.
    pub fn variant(&self, variant: usize) -> Series {
        let field = &self.union_fields()[variant];
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let child = arr.fields()[variant].as_ref();
                let mut positions = ChildPositions::new(child.len());
                for (i, (&t, &o)) in arr.types().iter().zip(arr.offsets().iter()).enumerate() {
                    let hit = t as usize == variant && arr.is_valid(i);
                    positions.push(hit.then_some(o as usize));
                }
                // SAFETY: the offsets of the union are in bounds of its children.
                unsafe { gather_child(child, positions) }
            })
            .collect();
        // SAFETY: the chunks have the physical dtype of the variant.
        unsafe {
            Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, field.dtype())
        }
    }

    /// Splits the non-null rows into `n_groups` groups, sending the rows of variant `t` to group
    /// `group_of[t]` and dropping them for `None`.
    ///
    /// Returns the rows of every group in row order, and for every variant `t` with
    /// `values_of[t]` its values at its rows, in row order.
    pub fn split_by_variant(
        &self,
        group_of: &[Option<usize>],
        n_groups: usize,
        values_of: &[bool],
    ) -> (Vec<Vec<IdxSize>>, Vec<Option<Series>>) {
        const NONE: u8 = u8::MAX;
        let fields = self.union_fields();
        let n_variants = fields.len();
        assert!(group_of.len() == n_variants && values_of.len() == n_variants);
        assert!(n_groups < NONE as usize);
        let group: Vec<u8> = group_of
            .iter()
            .map(|g| g.map_or(NONE, |g| g as u8))
            .collect();

        let mut rows: Vec<Vec<IdxSize>> = vec![Vec::new(); n_groups];
        let mut chunks: Vec<Vec<ArrayRef>> = vec![Vec::new(); n_variants];
        let mut base = 0 as IdxSize;
        for arr in self.downcast_iter() {
            // Until a variant's rows stop pointing at its child's slots in order, its values are a
            // prefix of the child and need no gather.
            let mut next = vec![0usize; n_variants];
            let mut positions: Vec<Option<ChildPositions>> =
                (0..n_variants).map(|_| None).collect();
            let mut visit = |i: usize| {
                let t = arr.types()[i] as usize;
                let g = group[t];
                if g != NONE {
                    rows[g as usize].push(base + i as IdxSize);
                }
                if values_of[t] {
                    let o = arr.offsets()[i] as usize;
                    if positions[t].is_none() && o != next[t] {
                        let mut p = ChildPositions::new(arr.fields()[t].len());
                        (0..next[t]).for_each(|j| p.push(Some(j)));
                        positions[t] = Some(p);
                    }
                    if let Some(p) = &mut positions[t] {
                        p.push(Some(o));
                    }
                    next[t] += 1;
                }
            };
            match arr.validity() {
                None => (0..arr.len()).for_each(&mut visit),
                Some(v) => v.true_idx_iter().for_each(&mut visit),
            }
            for t in (0..n_variants).filter(|&t| values_of[t]) {
                let child = arr.fields()[t].as_ref();
                chunks[t].push(match positions[t].take() {
                    None if next[t] == child.len() => child.to_boxed(),
                    None => child.sliced(0, next[t]),
                    // SAFETY: the offsets of the union are in bounds of its children.
                    Some(p) => unsafe { gather_child(child, p) },
                });
            }
            base += arr.len() as IdxSize;
        }

        let values = chunks
            .into_iter()
            .zip(fields.iter())
            .zip(values_of)
            .map(|((chunks, field), &wanted)| {
                // SAFETY: the chunks have the physical dtype of the variant.
                wanted.then(|| unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        self.name().clone(),
                        chunks,
                        field.dtype(),
                    )
                })
            })
            .collect();
        (rows, values)
    }
}

impl Series {
    /// Replaces every union in this Series by a struct with one nullable member per variant,
    /// holding each row's value in its variant's member; `None` if it holds no union.
    ///
    /// This is how formats without a union type, such as Parquet, store unions.
    #[cfg(feature = "dtype-struct")]
    pub fn unions_to_structs(&self) -> PolarsResult<Option<Series>> {
        if !self.dtype().contains_union() {
            return Ok(None);
        }
        let rec = |s: Series| -> PolarsResult<Series> { Ok(s.unions_to_structs()?.unwrap_or(s)) };
        let out = match self.dtype() {
            DataType::Union(fields) => {
                let ca = self.union()?;
                let members = (0..fields.len())
                    .map(|t| rec(ca.variant(t).with_name(fields[t].name().clone())))
                    .collect::<PolarsResult<Vec<_>>>()?;
                // A null row is a null struct rather than one with null members, so that readers
                // that see the struct see a null member for every variant but the row's.
                let validity = self
                    .is_not_null()
                    .rechunk()
                    .downcast_as_array()
                    .values()
                    .clone();
                StructChunked::from_series(self.name().clone(), self.len(), members.iter())?
                    .with_outer_validity(Some(validity))
                    .into_series()
            },
            DataType::List(_) => self.list()?.apply_to_inner(&rec)?.into_series(),
            #[cfg(feature = "dtype-array")]
            DataType::Array(..) => self.array()?.apply_to_inner(&rec)?.into_series(),
            DataType::Struct(_) => self
                .struct_()?
                .try_apply_fields(|f| rec(f.clone()))?
                .into_series(),
            #[cfg(feature = "dtype-map")]
            DataType::Map(..) => {
                let storage = rec(self.map()?.storage().clone())?;
                let DataType::List(entries) = storage.dtype() else {
                    unreachable!()
                };
                let DataType::Struct(kv) = entries.as_ref() else {
                    unreachable!()
                };
                let dtype = DataType::Map(
                    Box::new(kv[0].dtype().clone()),
                    Box::new(kv[1].dtype().clone()),
                );
                // SAFETY: replacing unions by structs keeps the entries, and with them the keys.
                unsafe { MapChunked::from_storage_unchecked(dtype, storage) }.into_series()
            },
            dt => polars_bail!(
                InvalidOperation: "cannot store a union nested in `{dt}` as a struct"
            ),
        };
        Ok(Some(out))
    }
}
