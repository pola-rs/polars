use ndarray::prelude::*;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::POOL;
use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IndexOrder {
    C,
    #[default]
    Fortran,
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// If data is aligned in a single chunk and has no Null values a zero copy view is returned
    /// as an [ndarray]
    pub fn to_ndarray(&self) -> PolarsResult<ArrayView1<'_, T::Native>> {
        let slice = self.cont_slice()?;
        Ok(aview1(slice))
    }
}

impl ListChunked {
    /// If all nested [`Series`] have the same length, a 2 dimensional [`ndarray::Array`] is returned.
    pub fn to_ndarray<N>(&self) -> PolarsResult<Array2<N::Native>>
    where
        N: PolarsNumericType,
    {
        polars_ensure!(
            self.null_count() == 0,
            ComputeError: "creation of ndarray with null values is not supported"
        );

        // first iteration determine the size
        let mut iter = self.into_no_null_iter();
        let series = iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "unable to create ndarray of empty ListChunked"))?;

        let width = series.len();
        let mut row_idx = 0;
        let mut ndarray = ndarray::Array::uninit((self.len(), width));

        let series = series.cast(&N::get_static_dtype())?;
        let ca = series.unpack::<N>()?;
        let a = ca.to_ndarray()?;
        let mut row = ndarray.slice_mut(s![row_idx, ..]);
        a.assign_to(&mut row);
        row_idx += 1;

        for series in iter {
            polars_ensure!(
                series.len() == width,
                ShapeMismatch: "unable to create a 2-D array, series have different lengths"
            );
            let series = series.cast(&N::get_static_dtype())?;
            let ca = series.unpack::<N>()?;
            let a = ca.to_ndarray()?;
            let mut row = ndarray.slice_mut(s![row_idx, ..]);
            a.assign_to(&mut row);
            row_idx += 1;
        }

        debug_assert_eq!(row_idx, self.len());
        // SAFETY:
        // We have assigned to every row and element of the array
        unsafe { Ok(ndarray.assume_init()) }
    }
}

impl DataFrame {
    /// Create a 2D [`ndarray::Array`] from this [`DataFrame`]. This requires all columns in the
    /// [`DataFrame`] to be non-null and numeric. They will be cast to the same data type
    /// (if they aren't already).
    ///
    /// For floating point data we implicitly convert `None` to `NaN` without failure.
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// let a = UInt32Chunked::new("a".into(), &[1, 2, 3]).into_column();
    /// let b = Float64Chunked::new("b".into(), &[10., 8., 6.]).into_column();
    ///
    /// let df = DataFrame::new_infer_height(vec![a, b]).unwrap();
    /// let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
    /// println!("{:?}", ndarray);
    /// ```
    /// Outputs:
    /// ```text
    /// [[1.0, 10.0],
    ///  [2.0, 8.0],
    ///  [3.0, 6.0]], shape=[3, 2], strides=[1, 3], layout=Ff (0xa), const ndim=2
    /// ```
    pub fn to_ndarray<N>(&self, ordering: IndexOrder) -> PolarsResult<Array2<N::Native>>
    where
        N: PolarsNumericType,
    {
        let shape = self.shape();
        let height = self.height();
        let columns = self.columns();
        let num_cols = columns.len();

        let mut membuf: Vec<N::Native> = Vec::with_capacity(shape.0 * shape.1);
        let ptr = membuf.as_ptr() as usize;

        // Cast a column to N's dtype, replace float nulls with NaN, error on remaining
        // nulls. Shared by both writer paths.
        let cast_to_target = |s: &Column| -> PolarsResult<Series> {
            let s = s.as_materialized_series().cast(&N::get_static_dtype())?;
            let s = match s.dtype() {
                DataType::Float32 => s.f32().unwrap().none_to_nan().into_series(),
                DataType::Float64 => s.f64().unwrap().none_to_nan().into_series(),
                _ => s,
            };
            polars_ensure!(
                s.null_count() == 0,
                ComputeError: "creation of ndarray with null values is not supported"
            );
            Ok(s)
        };

        match (ordering, num_cols) {
            // F-order, or C-order with 0/1 columns (same memory layout). Per-column
            // parallel writer; each column owns a disjoint contiguous stripe of the
            // output buffer.
            (IndexOrder::Fortran, _) | (IndexOrder::C, 0 | 1) => {
                POOL.install(|| {
                    columns.par_iter().enumerate().try_for_each(
                        |(col_idx, s)| -> PolarsResult<()> {
                            let s = cast_to_target(s)?;
                            let ca = s.unpack::<N>()?;

                            let mut chunk_offset = 0;
                            for arr in ca.downcast_iter() {
                                let vals = arr.values();

                                // SAFETY:
                                // We get parallel access to the vector by offsetting index access
                                // accordingly. We only operate on n contiguous elements, offset by
                                // n * the column index.
                                unsafe {
                                    let offset_ptr = (ptr as *mut N::Native)
                                        .add(col_idx * height + chunk_offset);
                                    // SAFETY:
                                    // this is uninitialized memory, so we must never read from this
                                    // data; copy_from_slice does not read.
                                    let buf =
                                        std::slice::from_raw_parts_mut(offset_ptr, vals.len());
                                    buf.copy_from_slice(vals)
                                }
                                chunk_offset += vals.len();
                            }

                            Ok(())
                        },
                    )
                })?;
            },
            // C-order with > 1 column. Cache-blocked transpose writer: row-block
            // parallel work units own disjoint output rows; each column's row-block
            // is gathered into a register-blocked sub-tile then bulk-written
            // contiguously. Avoids the per-column strided write that would touch
            // one cache line per element and false-share between threads.
            (IndexOrder::C, _) => {
                // Sequential below ~1M cells (~4 MB f32) skips rayon dispatch overhead.
                const PARALLEL_THRESHOLD: usize = 1_000_000;
                let parallel = num_cols.saturating_mul(height) >= PARALLEL_THRESHOLD;

                // The cast result must outlive Phase 2; the per-chunk slices below
                // borrow into it. No-op when source dtype already matches and the
                // column has no nulls. When a cast is needed it costs an extra
                // memory pass over the source; fusing it into the sub-tile gather
                // would halve the cast-required traffic but needs per-source-dtype
                // specialisation of the gather.
                let cast_columns: Vec<Series> = if parallel {
                    POOL.install(|| {
                        columns
                            .par_iter()
                            .map(cast_to_target)
                            .collect::<PolarsResult<_>>()
                    })?
                } else {
                    columns
                        .iter()
                        .map(cast_to_target)
                        .collect::<PolarsResult<_>>()?
                };

                const COL_BLOCK: usize = 64;
                const TARGET_BLOCK_CELLS: usize = 32_768;
                const MIN_ROW_BLOCK: usize = 64;
                // row_block scales inversely with num_cols so each work unit carries
                // about TARGET_BLOCK_CELLS cells, clamped at MIN_ROW_BLOCK for narrow
                // frames.
                let row_block = (TARGET_BLOCK_CELLS / num_cols.max(1)).max(MIN_ROW_BLOCK);
                let num_blocks = height.div_ceil(row_block);

                let column_chunks: Vec<Vec<&[N::Native]>> = cast_columns
                    .iter()
                    .map(|s| {
                        s.unpack::<N>()
                            .unwrap()
                            .downcast_iter()
                            .map(|arr| arr.values().as_slice())
                            .collect()
                    })
                    .collect();

                // Cursor into one column's chunk list. Advanced only at chunk boundaries.
                #[derive(Clone, Copy, Default)]
                struct Cursor {
                    idx: usize, // chunk index
                    off: usize, // first row of the chunk in the column's row space
                    end: usize, // first row past the chunk
                }

                // Sub-tile dimension for the register-blocked transpose. SUB=32 keeps
                // the SUB*SUB stack scratch under L1 (4 KB read + 4 KB write working
                // set per sub-tile) and makes each row write 128 B for f32, which
                // lowers to ~8 SIMD stores via copy_nonoverlapping.
                const SUB: usize = 32;

                let writer = |block: usize| {
                    let row_start = block * row_block;
                    let row_end = (row_start + row_block).min(height);
                    let block_rows = row_end - row_start;

                    let mut cursors = [Cursor::default(); COL_BLOCK];

                    for col_start in (0..num_cols).step_by(COL_BLOCK) {
                        let block_cols = (num_cols - col_start).min(COL_BLOCK);

                        // Position each cursor at row_start. The length-accumulator
                        // walk skips zero-length chunks naturally.
                        for ci_offset in 0..block_cols {
                            let chunks = &column_chunks[col_start + ci_offset];
                            let mut acc = 0usize;
                            let mut idx = 0;
                            for (i, c) in chunks.iter().enumerate() {
                                if acc + c.len() > row_start {
                                    idx = i;
                                    break;
                                }
                                acc += c.len();
                            }
                            cursors[ci_offset] = Cursor {
                                idx,
                                off: acc,
                                end: acc + chunks[idx].len(),
                            };
                        }

                        for sr in (0..block_rows).step_by(SUB) {
                            let tile_rows = (block_rows - sr).min(SUB);
                            let abs_row_start = row_start + sr;

                            for sc in (0..block_cols).step_by(SUB) {
                                let tile_cols = (block_cols - sc).min(SUB);

                                // Resolve each column's slice and starting offset
                                // for this sub-tile. all_simple stays true when
                                // every column's current chunk fully covers
                                // tile_rows; the inner gather is then a tight loop
                                // with no per-cell cursor work. It falls to false
                                // when a chunk boundary lands inside this sub-tile.
                                let mut col_slices: [&[N::Native]; SUB] = [&[]; SUB];
                                let mut col_offs = [0usize; SUB];
                                let mut all_simple = true;
                                for ci in 0..tile_cols {
                                    let chunks = &column_chunks[col_start + sc + ci];
                                    let cur = &mut cursors[sc + ci];
                                    // `while` (not `if`) skips zero-length chunks.
                                    while abs_row_start >= cur.end {
                                        cur.idx += 1;
                                        cur.off = cur.end;
                                        cur.end = cur.off + chunks[cur.idx].len();
                                    }
                                    if abs_row_start + tile_rows <= cur.end {
                                        col_slices[ci] = chunks[cur.idx];
                                        col_offs[ci] = abs_row_start - cur.off;
                                    } else {
                                        all_simple = false;
                                        break;
                                    }
                                }

                                let mut buf = [N::Native::default(); SUB * SUB];

                                if all_simple {
                                    for ri in 0..tile_rows {
                                        let buf_row = ri * SUB;
                                        for ci in 0..tile_cols {
                                            // SAFETY: col_offs[ci] + ri < col_slices[ci].len()
                                            // by the resolution check above.
                                            unsafe {
                                                buf[buf_row + ci] = *col_slices[ci]
                                                    .get_unchecked(col_offs[ci] + ri);
                                            }
                                        }
                                    }
                                } else {
                                    for ri in 0..tile_rows {
                                        let abs_row = abs_row_start + ri;
                                        let buf_row = ri * SUB;
                                        for ci in 0..tile_cols {
                                            let chunks = &column_chunks[col_start + sc + ci];
                                            let cur = &mut cursors[sc + ci];
                                            while abs_row >= cur.end {
                                                cur.idx += 1;
                                                cur.off = cur.end;
                                                cur.end = cur.off + chunks[cur.idx].len();
                                            }
                                            // SAFETY: cursor advanced only while
                                            // row < total length so cur.idx is in
                                            // bounds; offset < chunks[cur.idx].len()
                                            // by the chunk-end invariant.
                                            unsafe {
                                                buf[buf_row + ci] = *chunks[cur.idx]
                                                    .get_unchecked(abs_row - cur.off);
                                            }
                                        }
                                    }
                                }

                                for ri in 0..tile_rows {
                                    let abs_row = abs_row_start + ri;
                                    let dst_off = abs_row * num_cols + col_start + sc;
                                    let src = &buf[ri * SUB..ri * SUB + tile_cols];
                                    // SAFETY: dst_off + tile_cols <= height * num_cols;
                                    // disjoint blocks own disjoint output rows.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            src.as_ptr(),
                                            (ptr as *mut N::Native).add(dst_off),
                                            tile_cols,
                                        );
                                    }
                                }
                            }
                        }
                    }
                };
                if parallel {
                    POOL.install(|| (0..num_blocks).into_par_iter().for_each(writer));
                } else {
                    (0..num_blocks).for_each(writer);
                }
            },
        }

        // SAFETY:
        // we have written all data, so we can now safely set length
        unsafe {
            membuf.set_len(shape.0 * shape.1);
        }
        // Depending on the desired order, we can either return the array buffer as-is or reverse
        // the axes.
        match ordering {
            IndexOrder::C => Ok(Array2::from_shape_vec((shape.0, shape.1), membuf).unwrap()),
            IndexOrder::Fortran => {
                let ndarr = Array2::from_shape_vec((shape.1, shape.0), membuf).unwrap();
                Ok(ndarr.reversed_axes())
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ndarray_from_ca() -> PolarsResult<()> {
        let ca = Float64Chunked::new(PlSmallStr::EMPTY, &[1.0, 2.0, 3.0]);
        let ndarr = ca.to_ndarray()?;
        assert_eq!(ndarr, ArrayView1::from(&[1.0, 2.0, 3.0]));

        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            PlSmallStr::EMPTY,
            10,
            10,
            DataType::Float64,
        );
        builder.append_opt_slice(Some(&[1.0, 2.0, 3.0]));
        builder.append_opt_slice(Some(&[2.0, 4.0, 5.0]));
        builder.append_opt_slice(Some(&[6.0, 7.0, 8.0]));
        let list = builder.finish();

        let ndarr = list.to_ndarray::<Float64Type>()?;
        let expected = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        assert_eq!(ndarr, expected);

        // test list array that is not square
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            PlSmallStr::EMPTY,
            10,
            10,
            DataType::Float64,
        );
        builder.append_opt_slice(Some(&[1.0, 2.0, 3.0]));
        builder.append_opt_slice(Some(&[2.0]));
        builder.append_opt_slice(Some(&[6.0, 7.0, 8.0]));
        let list = builder.finish();
        assert!(list.to_ndarray::<Float64Type>().is_err());
        Ok(())
    }

    #[test]
    fn test_ndarray_from_df_order_fortran() -> PolarsResult<()> {
        let df = df!["a"=> [1.0, 2.0, 3.0],
            "b" => [2.0, 3.0, 4.0]
        ]?;

        let ndarr = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let expected = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        assert!(!ndarr.is_standard_layout());
        assert_eq!(ndarr, expected);

        Ok(())
    }

    #[test]
    fn test_ndarray_from_df_order_c() -> PolarsResult<()> {
        let df = df!["a"=> [1.0, 2.0, 3.0],
            "b" => [2.0, 3.0, 4.0]
        ]?;

        let ndarr = df.to_ndarray::<Float64Type>(IndexOrder::C)?;
        let expected = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        assert!(ndarr.is_standard_layout());
        assert_eq!(ndarr, expected);

        Ok(())
    }
}
