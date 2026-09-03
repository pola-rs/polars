#![allow(unsafe_op_in_unsafe_fn)]
use polars_compute::moment::{CovState, PearsonState};
use polars_core::prelude::*;
use polars_core::utils::{align_chunks_binary, try_get_supertype};

use super::*;

fn out_dtype(dtype_x: &DataType, dtype_y: &DataType) -> DataType {
    let st = try_get_supertype(dtype_x, dtype_y).unwrap_or(DataType::Float64);
    match st {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => DataType::Float16,
        DataType::Float32 => DataType::Float32,
        _ => DataType::Float64,
    }
}

pub fn new_cov_reduction(
    dtype_x: DataType,
    dtype_y: DataType,
    ddof: u8,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    polars_ensure!(
        dtype_x.is_primitive_numeric(),
        InvalidOperation: "`cov` operation not supported for dtype `{dtype_x}`"
    );
    polars_ensure!(
        dtype_y.is_primitive_numeric(),
        InvalidOperation: "`cov` operation not supported for dtype `{dtype_y}`"
    );
    let out_dtype = out_dtype(&dtype_x, &dtype_y);
    Ok(Box::new(CovGroupedReduction {
        values: Vec::new(),
        evicted_values: Vec::new(),
        ddof,
        out_dtype,
    }))
}

struct CovGroupedReduction {
    values: Vec<CovState>,
    evicted_values: Vec<CovState>,
    ddof: u8,
    out_dtype: DataType,
}

impl GroupedReduction for CovGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            ddof: self.ddof,
            out_dtype: self.out_dtype.clone(),
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, CovState::default());
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        let sx = values[0].cast(&DataType::Float64)?;
        let sy = values[1].cast(&DataType::Float64)?;
        let cx = sx.f64().unwrap();
        let cy = sy.f64().unwrap();
        let (cx, cy) = align_chunks_binary(cx, cy);
        let state = &mut self.values[group_idx as usize];
        for (ax, ay) in cx.downcast_iter().zip(cy.downcast_iter()) {
            state.combine(&polars_compute::moment::cov(ax, ay));
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        assert!(subset.len() == group_idxs.len());
        let sx = values[0]
            .take_slice_unchecked(subset)
            .cast(&DataType::Float64)?;
        let sy = values[1]
            .take_slice_unchecked(subset)
            .cast(&DataType::Float64)?;
        let cx = sx.f64().unwrap();
        let cy = sy.f64().unwrap();
        let ax = cx.downcast_as_array();
        let ay = cy.downcast_as_array();
        if ax.has_nulls() || ay.has_nulls() {
            for ((ox, oy), g) in ax.iter().zip(ay.iter()).zip(group_idxs) {
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    let old = core::mem::take(grp);
                    self.evicted_values.push(old);
                }
                if let (Some(x), Some(y)) = (ox, oy) {
                    grp.insert_one(*x, *y);
                }
            }
        } else {
            for ((x, y), g) in ax.values().iter().zip(ay.values().iter()).zip(group_idxs) {
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    let old = core::mem::take(grp);
                    self.evicted_values.push(old);
                }
                grp.insert_one(*x, *y);
            }
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        for (i, g) in subset.iter().zip(group_idxs) {
            let v = other.values.get_unchecked(*i as usize);
            let grp = self.values.get_unchecked_mut(*g as usize);
            grp.combine(v);
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            evicted_values: Vec::new(),
            ddof: self.ddof,
            out_dtype: self.out_dtype.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let ddof = self.ddof;
        let ca: Float64Chunked = v
            .into_iter()
            .map(|s| s.finalize(ddof))
            .collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(&self.out_dtype)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub fn new_pearson_corr_reduction(
    dtype_x: DataType,
    dtype_y: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    polars_ensure!(
        dtype_x.is_primitive_numeric(),
        InvalidOperation: "`corr` operation not supported for dtype `{dtype_x}`"
    );
    polars_ensure!(
        dtype_y.is_primitive_numeric(),
        InvalidOperation: "`corr` operation not supported for dtype `{dtype_y}`"
    );
    let out_dtype = out_dtype(&dtype_x, &dtype_y);
    Ok(Box::new(PearsonCorrGroupedReduction {
        values: Vec::new(),
        evicted_values: Vec::new(),
        out_dtype,
    }))
}

struct PearsonCorrGroupedReduction {
    values: Vec<PearsonState>,
    evicted_values: Vec<PearsonState>,
    out_dtype: DataType,
}

impl GroupedReduction for PearsonCorrGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            out_dtype: self.out_dtype.clone(),
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values
            .resize(num_groups as usize, PearsonState::default());
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        let sx = values[0].cast(&DataType::Float64)?;
        let sy = values[1].cast(&DataType::Float64)?;
        let cx = sx.f64().unwrap();
        let cy = sy.f64().unwrap();
        let (cx, cy) = align_chunks_binary(cx, cy);
        let state = &mut self.values[group_idx as usize];
        for (ax, ay) in cx.downcast_iter().zip(cy.downcast_iter()) {
            state.combine(&polars_compute::moment::pearson_corr(ax, ay));
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        assert!(subset.len() == group_idxs.len());
        let sx = values[0]
            .take_slice_unchecked(subset)
            .cast(&DataType::Float64)?;
        let sy = values[1]
            .take_slice_unchecked(subset)
            .cast(&DataType::Float64)?;
        let cx = sx.f64().unwrap();
        let cy = sy.f64().unwrap();
        let ax = cx.downcast_as_array();
        let ay = cy.downcast_as_array();
        if ax.has_nulls() || ay.has_nulls() {
            for ((ox, oy), g) in ax.iter().zip(ay.iter()).zip(group_idxs) {
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    let old = core::mem::take(grp);
                    self.evicted_values.push(old);
                }
                if let (Some(x), Some(y)) = (ox, oy) {
                    grp.insert_one(*x, *y);
                }
            }
        } else {
            for ((x, y), g) in ax.values().iter().zip(ay.values().iter()).zip(group_idxs) {
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    let old = core::mem::take(grp);
                    self.evicted_values.push(old);
                }
                grp.insert_one(*x, *y);
            }
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        for (i, g) in subset.iter().zip(group_idxs) {
            let v = other.values.get_unchecked(*i as usize);
            let grp = self.values.get_unchecked_mut(*g as usize);
            grp.combine(v);
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            evicted_values: Vec::new(),
            out_dtype: self.out_dtype.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let ca: Float64Chunked = v
            .into_iter()
            .map(|s| {
                if s.weight() == 0.0 {
                    None
                } else {
                    Some(s.finalize())
                }
            })
            .collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(&self.out_dtype)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
