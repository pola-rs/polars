use polars_core::prelude::{ChunkExpandAtIndex, GroupsIdx, GroupsType, ListChunked};
use polars_core::series::Series;
use polars_utils::{IdxSize, UnitVec};

use super::ReturnValue;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SliceGroup {
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
#[derive(Clone)]
pub struct SliceGroups(pub Box<[SliceGroup]>);

#[repr(C)]
#[derive(Clone)]
pub struct IndexGroups {
    pub index: Box<[u64]>,
    pub ends: Box<[u64]>,
}

#[repr(C)]
#[derive(Clone)]
pub enum GroupPositions {
    /// Every group has all the data values.
    SharedAcrossGroups {
        num_groups: usize,
    },
    /// Every group has 1 value in sequential order of the data.
    ScalarPerGroup,
    Slice(SliceGroups),
    Index(IndexGroups),
}

pub struct CowGroupPositions<'a> {
    groups: &'a GroupPositions,
    drop: Option<unsafe extern "C" fn(*mut GroupPositions) -> u32>,
}

impl<'a> AsRef<GroupPositions> for CowGroupPositions<'a> {
    fn as_ref(&self) -> &GroupPositions {
        self.groups
    }
}

impl<'a> Drop for CowGroupPositions<'a> {
    fn drop(&mut self) {
        if let Some(drop) = &self.drop {
            let rv = unsafe { (drop)(self.groups as *const GroupPositions as *mut GroupPositions) };
            match ReturnValue::from(rv) {
                ReturnValue::Ok => {},
                ReturnValue::Panic => panic!("plugin panicked"),
                _ => panic!("did not expect error"),
            }
        }
    }
}

impl GroupPositions {
    pub unsafe fn aggregate_to_list(&self, data: &Series) -> ListChunked {
        match self {
            Self::SharedAcrossGroups { num_groups } => {
                data.implode().unwrap().new_from_index(0, *num_groups)
            },
            Self::ScalarPerGroup => data.as_list(),
            Self::Slice(groups) => {
                let groups = groups.to_core();
                let groups = GroupsType::Slice {
                    groups,
                    overlapping: true,
                };
                unsafe { data.agg_list(&groups) }.list().unwrap().clone()
            },
            Self::Index(groups) => {
                let groups = groups.to_core();
                let groups = GroupsType::Idx(groups);
                unsafe { data.agg_list(&groups) }.list().unwrap().clone()
            },
        }
    }
}

impl SliceGroups {
    pub fn to_core(&self) -> Vec<[IdxSize; 2]> {
        self.0
            .iter()
            .map(|s| [s.offset as IdxSize, s.length as IdxSize])
            .collect::<Vec<_>>()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn lengths(&self) -> impl Iterator<Item = usize> {
        self.0.iter().map(|g| g.length as usize)
    }

    pub fn iter(&self) -> impl Iterator<Item = &SliceGroup> {
        self.0.iter()
    }
}

impl IndexGroups {
    pub fn lengths(&self) -> impl Iterator<Item = usize> {
        (0..self.ends.len()).map(|i| {
            let start = i.checked_sub(1).map_or(0, |i| self.ends[i]);
            let end = self.ends[i];
            (end - start) as usize
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u64]> {
        (0..self.ends.len()).map(|i| {
            let start = i.checked_sub(1).map_or(0, |i| self.ends[i]);
            let end = self.ends[i];

            &self.index[start as usize..end as usize]
        })
    }

    pub fn to_core(&self) -> GroupsIdx {
        self.iter()
            .map(|v| {
                (
                    v.first().copied().unwrap_or(0) as IdxSize,
                    UnitVec::from_iter(v.iter().copied().map(|v| v as IdxSize)),
                )
            })
            .collect::<Vec<_>>()
            .into()
    }
}

pub mod callee {
    use std::borrow::Cow;
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    use super::GroupPositions;
    use crate::version_0::{SeriesExport, export_series, import_series};
    use crate::version_1::_callee::wrap_callee_function;
    use crate::version_1::{DataPtr, PolarsPlugin};

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn evaluate_on_groups<Data: PolarsPlugin>(
        data: DataPtr,
        inputs_ptr: *mut (SeriesExport, NonNull<GroupPositions>),
        inputs_len: usize,
        output_series: NonNull<MaybeUninit<SeriesExport>>,
        output_groups_owned: NonNull<MaybeUninit<bool>>,
        output_groups: NonNull<MaybeUninit<NonNull<GroupPositions>>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let mut collected_inputs = Vec::with_capacity(inputs_len);
            for i in 0..inputs_len {
                let (series, groups) = unsafe { std::ptr::read(inputs_ptr.add(i)) };
                let series = unsafe { import_series(series)? };
                let groups = unsafe { groups.as_ref() };
                collected_inputs.push((series, groups));
            }
            let (out_data, out_groups) = data.evaluate_on_groups(&collected_inputs)?;

            let is_owned = matches!(out_groups, Cow::Owned(_));
            unsafe { output_groups_owned.write(MaybeUninit::new(is_owned)) };
            let out_groups = match out_groups {
                Cow::Borrowed(ptr) => NonNull::from_ref(ptr),
                Cow::Owned(out_groups) => {
                    NonNull::new(Box::into_raw(Box::new(out_groups))).unwrap()
                },
            };
            unsafe {
                output_groups.write(MaybeUninit::new(out_groups));
            }
            let out_data = export_series(&out_data);
            unsafe { output_series.write(MaybeUninit::new(out_data)) };

            Ok(())
        })
    }

    pub unsafe extern "C" fn drop_box_group_positions(ptr: *mut GroupPositions) -> u32 {
        wrap_callee_function(|| {
            let buffer = unsafe { Box::from_raw(ptr) };
            drop(buffer);
            Ok(())
        })
    }
}

mod caller {
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    use polars_core::series::Series;
    use polars_error::PolarsResult;

    use super::{CowGroupPositions, GroupPositions};
    use crate::version_0::{export_series, import_series};
    use crate::version_1::{DataPtr, VTable};

    impl VTable {
        pub unsafe fn evaluate_on_groups<'a>(
            &self,
            data: DataPtr,
            inputs: &[(Series, &'a GroupPositions)],
        ) -> PolarsResult<(Series, CowGroupPositions<'a>)> {
            let mut out_series = MaybeUninit::uninit();
            let mut out_groups_owned = MaybeUninit::uninit();
            let mut out_groups = MaybeUninit::uninit();

            let mut inputs_export = Vec::with_capacity(inputs.len());
            for (series, groups) in inputs {
                let series = export_series(series);
                let groups = NonNull::from_ref(*groups);
                inputs_export.push((series, groups));
            }

            let rv = unsafe {
                (self._evaluate_on_groups)(
                    data,
                    inputs_export.as_mut_ptr(),
                    inputs.len(),
                    NonNull::from_mut(&mut out_series),
                    NonNull::from_mut(&mut out_groups_owned),
                    NonNull::from_mut(&mut out_groups),
                )
            };
            // Already deallocated in step function
            unsafe { inputs_export.set_len(0) };
            self.handle_return_value(rv)?;

            let out_series = unsafe { out_series.assume_init() };
            let out_series = unsafe { import_series(out_series) }?;
            let out_groups_owned = unsafe { out_groups_owned.assume_init() };
            let out_groups = unsafe { out_groups.assume_init() };
            let drop_fn = out_groups_owned.then_some(self._drop_box_group_positions);
            Ok((
                out_series,
                CowGroupPositions {
                    groups: unsafe { out_groups.as_ref() },
                    drop: drop_fn,
                },
            ))
        }
    }
}
