/// Container of byte-widths for (partial) rows.
///
/// The `RowWidths` keeps track of the sum of all widths and allows to efficiently deal with a
/// constant row-width (i.e. with primitive types).
#[derive(Debug, Clone)]
pub(crate) enum RowWidths {
    Constant { num_rows: usize, width: usize },
    // @TODO: Maybe turn this into a Box<[usize]>
    Variable { widths: Vec<usize>, sum: usize },
}

impl Default for RowWidths {
    fn default() -> Self {
        Self::Constant {
            num_rows: 0,
            width: 0,
        }
    }
}

impl RowWidths {
    pub fn new(num_rows: usize) -> Self {
        Self::Constant { num_rows, width: 0 }
    }

    /// Push a constant width into the widths
    pub fn push_constant(&mut self, constant: usize) {
        match self {
            Self::Constant { width, .. } => *width += constant,
            Self::Variable { widths, sum } => {
                widths.iter_mut().for_each(|w| *w += constant);
                *sum += constant * widths.len();
            },
        }
    }
    /// Push an another [`RowWidths`] into the widths
    pub fn push(&mut self, other: &Self) {
        debug_assert_eq!(self.num_rows(), other.num_rows());

        match (std::mem::take(self), other) {
            (mut slf, RowWidths::Constant { width, num_rows: _ }) => {
                slf.push_constant(*width);
                *self = slf;
            },
            (RowWidths::Constant { num_rows, width }, RowWidths::Variable { widths, sum }) => {
                *self = RowWidths::Variable {
                    widths: widths.iter().map(|w| *w + width).collect(),
                    sum: num_rows * width + sum,
                };
            },
            (
                RowWidths::Variable { mut widths, sum },
                RowWidths::Variable {
                    widths: other_widths,
                    sum: other_sum,
                },
            ) => {
                widths
                    .iter_mut()
                    .zip(other_widths.iter())
                    .for_each(|(l, r)| *l += *r);
                *self = RowWidths::Variable {
                    widths,
                    sum: sum + other_sum,
                };
            },
        }
    }

    /// Create a [`RowWidths`] with the chunked sum with a certain `chunk_size`.
    pub fn collapse_chunks(&self, chunk_size: usize, output_num_rows: usize) -> RowWidths {
        if chunk_size == 0 {
            assert_eq!(self.num_rows(), 0);
            return RowWidths::new(output_num_rows);
        }

        assert_eq!(self.num_rows() % chunk_size, 0);
        assert_eq!(self.num_rows() / chunk_size, output_num_rows);
        match self {
            Self::Constant { num_rows, width } => Self::Constant {
                num_rows: num_rows / chunk_size,
                width: width * chunk_size,
            },
            Self::Variable { widths, sum } => Self::Variable {
                widths: widths
                    .chunks_exact(chunk_size)
                    .map(|chunk| chunk.iter().copied().sum())
                    .collect(),
                sum: *sum,
            },
        }
    }

    pub fn extend_with_offsets(&self, out: &mut Vec<usize>) {
        match self {
            RowWidths::Constant { num_rows, width } => {
                out.extend((0..*num_rows).map(|i| i * width));
            },
            RowWidths::Variable { widths, sum: _ } => {
                let mut next = 0;
                out.extend(widths.iter().map(|w| {
                    let current = next;
                    next += w;
                    current
                }));
            },
        }
    }

    pub fn num_rows(&self) -> usize {
        match self {
            Self::Constant { num_rows, .. } => *num_rows,
            Self::Variable { widths, .. } => widths.len(),
        }
    }

    pub fn push_iter(&mut self, mut iter: impl ExactSizeIterator<Item = usize>) {
        assert_eq!(self.num_rows(), iter.len());

        match self {
            RowWidths::Constant { num_rows, width } => {
                let Some(constant) = iter.by_ref().next() else {
                    return;
                };

                // If the iterator turns out to be constant anyway. We would like to keep that
                // benefit.
                match iter.by_ref().enumerate().find(|(_, v)| *v != constant) {
                    None => *width += constant,
                    Some((i, v)) => {
                        let mut push_sum = (i + 1) * constant + v;
                        let mut slf = Vec::with_capacity(*num_rows);

                        slf.resize(i + 1, *width + constant);
                        slf.push(*width + v);

                        slf.extend(iter.map(|v| {
                            push_sum += v;
                            v + *width
                        }));

                        *self = Self::Variable {
                            widths: slf,
                            sum: *num_rows * *width + push_sum,
                        };
                    },
                }
            },
            RowWidths::Variable { widths, sum } => {
                let mut push_sum = 0;
                iter.zip(widths).for_each(|(v, w)| {
                    push_sum += v;
                    *w += v;
                });
                *sum += push_sum;
            },
        }
    }

    pub fn get(&self, index: usize) -> usize {
        assert!(index < self.num_rows());
        match self {
            Self::Constant { width, .. } => *width,
            Self::Variable { widths, .. } => widths[index],
        }
    }

    pub fn sum(&self) -> usize {
        match self {
            Self::Constant { num_rows, width } => *num_rows * *width,
            Self::Variable { sum, .. } => *sum,
        }
    }
}
