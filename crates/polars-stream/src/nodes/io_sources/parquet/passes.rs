//! Which predicate columns to decode in each pass over a row group.

/// Above this share of rows kept by a pass, the next pass is merged into it.
const STAGED_MAX_KEPT_PERCENT: usize = 85;

pub(super) fn keeps_most_rows(kept: usize, total: usize) -> bool {
    kept * 100 > STAGED_MAX_KEPT_PERCENT * total
}

/// `passes` with `columns` moved to one pass at the front, so each is measured
/// on every row. An emptied pass goes, but a last pass for the rest stays.
pub(super) fn promote(passes: &[Vec<usize>], columns: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![columns.to_vec()];
    let last = passes.len() - 1;
    for (i, pass) in passes.iter().enumerate() {
        let pass: Vec<usize> = pass
            .iter()
            .copied()
            .filter(|c| !columns.contains(c))
            .collect();
        if !pass.is_empty() || (i == last && passes[last].is_empty()) {
            out.push(pass);
        }
    }
    out
}

/// The rows a predicate column, or a pass, was evaluated on and the rows it kept.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) struct Selectivity {
    pub(super) input_rows: usize,
    pub(super) kept_rows: usize,
}

impl Selectivity {
    /// The share of rows kept. No rows to measure counts as keeping every row.
    fn percent(self) -> usize {
        match self.input_rows {
            0 => 100,
            _ => self.kept_rows * 100 / self.input_rows,
        }
    }
}

/// The passes of the next row group from the rows each predicate column kept in the
/// last one: the most selective first, a column in a pass of its own while it rejects
/// enough rows, otherwise together with the next. The fields only the rest of the
/// predicate reads get a pass of their own when the passes before them reject enough
/// rows.
///
/// A column in a later pass is measured on the rows the passes before it kept. Over the
/// whole row group it keeps between `kept_rows` and `kept_rows + num_rows - input_rows`
/// rows, and moves before a column of an earlier pass only when that range lies below
/// the rows that column kept. The rows a pass keeps as a whole come from
/// `pass_selectivity`.
pub(super) fn plan_passes(
    passes: &[Vec<usize>],
    num_rows: usize,
    selectivity: &[Selectivity],
    pass_selectivity: &[Selectivity],
    has_rest_fields: bool,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = Vec::new();
    for pass in passes {
        let mut pass = pass.clone();
        pass.sort_by_key(|&c| selectivity[c].kept_rows);
        for c in pass {
            let Selectivity {
                input_rows,
                kept_rows,
            } = selectivity[c];
            let upper = kept_rows + num_rows - input_rows;
            let mut i = order.len();
            while i > 0 && upper < selectivity[order[i - 1]].kept_rows {
                i -= 1;
            }
            order.insert(i, c);
        }
    }

    // A column that keeps most rows shares its pass with the next one in the order.
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut pass = Vec::new();
    for c in order {
        pass.push(c);
        if !keeps_most_rows(selectivity[c].percent(), 100) {
            out.push(std::mem::take(&mut pass));
        }
    }
    if !pass.is_empty() {
        out.push(pass);
    }

    let last_kept = match out.last() {
        None => 100,
        Some(last) => {
            let same_columns =
                |p: &Vec<usize>| p.len() == last.len() && last.iter().all(|c| p.contains(c));
            match passes.iter().position(same_columns) {
                Some(i) => pass_selectivity[i].percent(),
                None => last
                    .iter()
                    .fold(100, |acc, &c| acc * selectivity[c].percent() / 100),
            }
        },
    };
    if has_rest_fields && (out.is_empty() || !keeps_most_rows(last_kept, 100)) {
        out.push(Vec::new());
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_promote() {
        use super::promote;
        assert_eq!(
            promote(&[vec![0, 1], vec![2], vec![]], &[2]),
            vec![vec![2], vec![0, 1], vec![]]
        );
        assert_eq!(
            promote(&[vec![0], vec![1, 2]], &[1]),
            vec![vec![1], vec![0], vec![2]]
        );
        assert_eq!(promote(&[vec![0, 1]], &[0, 1]), vec![vec![0, 1]]);
        assert_eq!(
            promote(&[vec![0], vec![1], vec![2]], &[1, 2]),
            vec![vec![1, 2], vec![0]]
        );
    }

    #[test]
    fn test_plan_passes() {
        use super::{Selectivity, plan_passes};

        fn sel(input_rows: usize, kept_rows: usize) -> Selectivity {
            Selectivity {
                input_rows,
                kept_rows,
            }
        }

        // Most selective first, each alone; the dense ones together with the rest.
        assert_eq!(
            plan_passes(
                &[vec![0, 1, 2, 3]],
                100,
                &[sel(100, 90), sel(100, 20), sel(100, 50), sel(100, 99)],
                &[sel(100, 10)],
                true
            ),
            vec![vec![1], vec![2], vec![0, 3]]
        );
        // Dense columns that reject enough together still get the rest apart.
        assert_eq!(
            plan_passes(
                &[vec![0, 1]],
                100,
                &[sel(100, 90), sel(100, 90)],
                &[sel(100, 80)],
                true
            ),
            vec![vec![0, 1], vec![]]
        );
        assert_eq!(
            plan_passes(
                &[vec![0, 1]],
                100,
                &[sel(100, 90), sel(100, 90)],
                &[sel(100, 80)],
                false
            ),
            vec![vec![0, 1]]
        );
        // Dense columns that reject the same rows are measured as a pass, not multiplied.
        assert_eq!(
            plan_passes(
                &[vec![0, 1], vec![]],
                100,
                &[sel(100, 90), sel(100, 90)],
                &[sel(100, 90), sel(90, 90)],
                true
            ),
            vec![vec![0, 1]]
        );
        // Everything staged: the rest alone.
        assert_eq!(
            plan_passes(
                &[vec![0], vec![1]],
                100,
                &[sel(100, 10), sel(10, 5)],
                &[sel(100, 10), sel(10, 5)],
                true
            ),
            vec![vec![0], vec![1], vec![]]
        );
        // A later column is measured on fewer rows: it stays behind unless it keeps
        // fewer rows than the earlier one however the rejected rows fall.
        assert_eq!(
            plan_passes(
                &[vec![0], vec![1], vec![]],
                100,
                &[sel(100, 40), sel(40, 1)],
                &[sel(100, 40), sel(40, 1), sel(1, 1)],
                true
            ),
            vec![vec![0], vec![1], vec![]]
        );
        assert_eq!(
            plan_passes(
                &[vec![0], vec![1], vec![]],
                100,
                &[sel(100, 80), sel(80, 4)],
                &[sel(100, 80), sel(80, 4), sel(4, 4)],
                true
            ),
            vec![vec![1], vec![0], vec![]]
        );
        // A dense column ahead of one that is not stays ahead and shares its pass, so
        // both are measured on the same rows next.
        assert_eq!(
            plan_passes(
                &[vec![0], vec![1], vec![]],
                100,
                &[sel(100, 86), sel(86, 73)],
                &[sel(100, 86), sel(86, 73), sel(73, 73)],
                true
            ),
            vec![vec![0, 1], vec![]]
        );
        // A pass is matched by its columns regardless of their order.
        assert_eq!(
            plan_passes(
                &[vec![1, 0], vec![]],
                100,
                &[sel(100, 90), sel(100, 91)],
                &[sel(100, 90), sel(90, 90)],
                true
            ),
            vec![vec![0, 1]]
        );
        // A column with no rows to measure keeps every row and stays in place.
        assert_eq!(
            plan_passes(
                &[vec![0], vec![1], vec![2]],
                100,
                &[sel(100, 0), sel(0, 0), sel(0, 0)],
                &[sel(100, 0), sel(0, 0), sel(0, 0)],
                false
            ),
            vec![vec![0], vec![1, 2]]
        );
        // No predicate columns: the rest is the only pass.
        assert_eq!(
            plan_passes(&[Vec::new()], 100, &[], &[sel(100, 100)], true),
            vec![Vec::<usize>::new()]
        );
    }
}
