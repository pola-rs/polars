use super::*;
use crate::chunked_array::strings::*;
#[allow(unused_imports)]
use crate::prelude::*;

pub trait CategoricalStringNameSpace: Sized {
    fn str_slice(&self, offset: &Column, length: &Column) -> PolarsResult<Self>;
    fn str_head(&self, n: &Column) -> PolarsResult<Self>;
    fn str_tail(&self, n: &Column) -> PolarsResult<Self>;
}

/// Apply operation to categories of a CategoricalChunked and remap physical & rev map.
fn fast_remap_ca<F>(ca: &CategoricalChunked, mut op: F) -> PolarsResult<CategoricalChunked>
where
    F: FnMut(&StringChunked) -> PolarsResult<StringChunked>,
{
    // Determine physical and categories.
    let rev_map = &**ca.get_rev_map();
    let (categories, old_phys) = match rev_map {
        RevMapping::Local(c, _) => (c, (0..c.len() as u32).collect::<Vec<u32>>()),
        RevMapping::Global(m, c, _) => {
            let mut idx = m.iter().collect::<Vec<(&u32, &u32)>>();
            // Set index to same ordering as categories.
            idx.sort_by_key(|&(_, key)| key);
            let idx = idx.iter().map(|v| *v.0).collect::<Vec<u32>>();
            (c, idx)
        },
    };
    let categories = StringChunked::with_chunk(PlSmallStr::EMPTY, categories.clone());

    // Apply function and retrieve new physical.
    let new_categories = op(&categories)?;
    let mapped = new_categories.cast(&DataType::Categorical(None, Default::default()))?;
    let new_phys = mapped.cast(&DataType::UInt32)?.u32()?.to_vec();

    // Create mapping from old physical => new physical.
    // Some values may map to null if the op returns null.
    let phys_mapper = old_phys
        .iter()
        .zip(new_phys.iter())
        .collect::<PlHashMap<&u32, &Option<u32>>>();

    // Apply mapping to physical.
    let new_phys = ca
        .physical()
        .apply(|opt| opt.and_then(|idx| phys_mapper.get(&idx).and_then(|&mapped| *mapped)));
    let new_rev_map = mapped.categorical()?.get_rev_map();

    // SAFETY: new rev_map is valid.
    let out = unsafe {
        CategoricalChunked::from_cats_and_rev_map_unchecked(
            new_phys,
            new_rev_map.clone(),
            ca.is_enum(),
            Default::default(),
        )
    };
    Ok(out)
}

/// Apply function to a categorical array by casting to String and back.
fn slow_remap_ca<F>(ca: &CategoricalChunked, mut op: F) -> PolarsResult<CategoricalChunked>
where
    F: FnMut(&StringChunked) -> PolarsResult<StringChunked>,
{
    let s = ca.cast(&DataType::String).unwrap();
    let s = op(s.str()?)?;
    let out = s.cast(&DataType::Categorical(None, Default::default()))?;
    let out = out.categorical()?;
    Ok(out.clone())
}

impl CategoricalStringNameSpace for CategoricalChunked {
    fn str_slice(&self, offset: &Column, length: &Column) -> PolarsResult<Self> {
        match (offset, length) {
            (Column::Scalar(_), Column::Scalar(_)) => {
                fast_remap_ca(self, |ca| ca.str_slice(offset, length))
            },
            _ => slow_remap_ca(self, |ca| ca.str_slice(offset, length)),
        }
    }

    fn str_head(&self, n: &Column) -> PolarsResult<Self> {
        match n {
            Column::Scalar(_) => fast_remap_ca(self, |ca| ca.str_head(n)),
            _ => slow_remap_ca(self, |ca| ca.str_head(n)),
        }
    }

    fn str_tail(&self, n: &Column) -> PolarsResult<Self> {
        match n {
            Column::Scalar(_) => fast_remap_ca(self, |ca| ca.str_tail(n)),
            _ => slow_remap_ca(self, |ca| ca.str_tail(n)),
        }
    }
}
