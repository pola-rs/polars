use std::fmt::{Display, Formatter, UpperExp};

use polars_core::error::*;

use crate::logical_plan::visitor::{VisitRecursion, Visitor};
use crate::prelude::visitor::AexprNode;
use crate::prelude::*;

/// Hack UpperExpr trait to get a kind of formatting that doesn't traverse the nodes.
/// So we can format with {foo:E}
impl UpperExp for AExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AExpr::Explode(_) => "explode",
            AExpr::Alias(_, name) => return write!(f, "alias({})", name.as_ref()),
            AExpr::Column(name) => return write!(f, "col({})", name.as_ref()),
            AExpr::Literal(lv) => return write!(f, "lit({lv:?})"),
            AExpr::BinaryExpr { op, .. } => return write!(f, "binary: {}", op),
            AExpr::Cast {
                data_type, strict, ..
            } => {
                return if *strict {
                    write!(f, "strict cast({})", data_type)
                } else {
                    write!(f, "cast({})", data_type)
                }
            },
            AExpr::Sort { options, .. } => {
                return write!(
                    f,
                    "sort: {}{}{}",
                    options.descending as u8, options.nulls_last as u8, options.multithreaded as u8
                )
            },
            AExpr::Gather { .. } => "gather",
            AExpr::SortBy { descending, .. } => {
                write!(f, "sort_by:")?;
                for i in descending {
                    write!(f, "{}", *i as u8)?;
                }
                return Ok(());
            },
            AExpr::Filter { .. } => "filter",
            AExpr::Agg(a) => {
                let s: &str = a.into();
                return write!(f, "{}", s.to_lowercase());
            },
            AExpr::Ternary { .. } => "ternary",
            AExpr::AnonymousFunction { options, .. } => {
                return write!(f, "anonymous_function: {}", options.fmt_str)
            },
            AExpr::Function { function, .. } => return write!(f, "function: {function}"),
            AExpr::Window { .. } => "window",
            AExpr::Wildcard => "*",
            AExpr::Slice { .. } => "slice",
            AExpr::Len => "len",
            AExpr::Nth(v) => return write!(f, "nth({})", v),
        };

        write!(f, "{s}")
    }
}

pub(crate) struct TreeFmtVisitor {
    levels: Vec<Vec<String>>,
    depth: u32,
    width: u32,
}

impl TreeFmtVisitor {
    pub(crate) fn new() -> Self {
        Self {
            levels: vec![],
            depth: 0,
            width: 0,
        }
    }
}

impl Visitor for TreeFmtVisitor {
    type Node = AexprNode;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        let repr = format!("{:E}", ae);

        if self.levels.len() <= self.depth as usize {
            self.levels.push(vec![])
        }

        // the post-visit ensures the width of this node is known
        let row = self.levels.get_mut(self.depth as usize).unwrap();

        // set default values to ensure we format at the right width
        row.resize(self.width as usize + 1, "".to_string());
        row[self.width as usize] = repr;

        // we will enter depth first, we enter child so depth increases
        self.depth += 1;

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        // because we traverse depth first
        // every post-visit increases the width as we finished a depth-first branch
        self.width += 1;

        // we finished this branch so we decrease in depth, back the caller node
        self.depth -= 1;
        Ok(VisitRecursion::Continue)
    }
}

/// Calculates the number of digits in a `usize` number
/// Useful for the alignment of of `usize` values when they are displayed
fn digits(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        f64::log10(n as f64) as usize + 1
    }
}

/// Meta-info of a column in a populated `TreeFmtVisitor` required for the pretty-print of a tree
#[derive(Clone, Default, Debug)]
struct TreeViewColumn {
    offset: usize,
    width: usize,
    center: usize,
    /// A `TreeViewColumn` `is_empty` when it doesn't contain a single populated `TreeViewCell`
    is_empty: bool,
}

/// Meta-info of a cell in a populated `TreeFmtVisitor`
#[derive(Clone, Default, Debug)]
struct TreeViewCell<'a> {
    text: Option<&'a str>,
    /// A `Vec` of indices of of `TreeViewColumn`-s stored elsewhere in another `Vec`
    /// For a cell on a row `i` these indices point to the columns that contain child-cells on a
    /// row `i + 1` (if the latter exists)
    /// NOTE: might warrant a rethink should this code become used broader
    children_columns: Vec<usize>,
}

/// The complete intermediate representation of a `TreeFmtVisitor` that can be drawn on a `Canvas`
/// down the line
#[derive(Default, Debug)]
struct TreeView<'a> {
    n_rows: usize,
    n_rows_width: usize,
    matrix: Vec<Vec<TreeViewCell<'a>>>,
    /// NOTE: `TreeViewCell`'s `children_columns` field contains indices pointing at the elements
    /// of this `Vec`
    columns: Vec<TreeViewColumn>,
}

// NOTE: the code below this line is full of hardcoded integer offsets which may not be a big
// problem as long as it remains the private implementation of the pretty-print
/// The conversion from a reference to `levels` field of a `TreeFmtVisitor`
impl<'a> From<&'a [Vec<String>]> for TreeView<'a> {
    #[allow(clippy::needless_range_loop)]
    fn from(value: &'a [Vec<String>]) -> Self {
        let n_rows = value.len();
        let n_cols = value.iter().map(|row| row.len()).max().unwrap_or(0);
        if n_rows == 0 || n_cols == 0 {
            return TreeView::default();
        }
        // the character-width of the highest index of a row
        let n_rows_width = digits(n_rows - 1);

        let mut matrix = vec![vec![TreeViewCell::default(); n_cols]; n_rows];
        for i in 0..n_rows {
            for j in 0..n_cols {
                if j < value[i].len() && !value[i][j].is_empty() {
                    matrix[i][j].text = Some(value[i][j].as_str());
                    if i < n_rows - 1 {
                        if j < value[i + 1].len() && !value[i + 1][j].is_empty() {
                            matrix[i][j].children_columns.push(j);
                        }
                        for k in j + 1..n_cols {
                            if (k >= value[i].len() || value[i][k].is_empty())
                                && k < value[i + 1].len()
                            {
                                if !value[i + 1][k].is_empty() {
                                    matrix[i][j].children_columns.push(k);
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        let mut offset = n_rows_width + 3;
        let mut columns = vec![TreeViewColumn::default(); n_cols];
        // the two nested loops below are those `needless_range_loop`s
        // more readable this way to my taste
        for j in 0..n_cols {
            let mut width = 0;
            let mut is_empty = true;
            for i in 0..n_rows {
                if let Some(text) = matrix[i][j].text {
                    is_empty = false;
                    width = [text.len(), width].into_iter().max().unwrap();
                }
            }
            width += if width > 0 { 6 } else { 4 };
            columns[j].offset = offset;
            columns[j].width = width;
            columns[j].center = width / 2 + width % 2;
            columns[j].is_empty = is_empty;
            offset += width;
        }

        Self {
            n_rows,
            n_rows_width,
            matrix,
            columns,
        }
    }
}

/// The basic charset that's used for drawing lines and boxes on a `Canvas`
struct Glyphs {
    void: char,
    vertical_line: char,
    horizontal_line: char,
    top_left_corner: char,
    top_right_corner: char,
    bottom_left_corner: char,
    bottom_right_corner: char,
}

impl Default for Glyphs {
    fn default() -> Self {
        Self {
            void: ' ',
            vertical_line: '│',
            horizontal_line: '─',
            top_left_corner: '╭',
            top_right_corner: '╮',
            bottom_left_corner: '╰',
            bottom_right_corner: '╯',
        }
    }
}

/// A `Point` on a `Canvas`
#[derive(Clone, Copy)]
struct Point(usize, usize);

/// The orientation of a line on a `Canvas`
#[derive(Clone, Copy)]
enum Orientation {
    Vertical,
    Horizontal,
}

/// `Canvas`
struct Canvas {
    width: usize,
    height: usize,
    canvas: Vec<Vec<char>>,
    glyphs: Glyphs,
}

impl Canvas {
    fn new(width: usize, height: usize, glyphs: Glyphs) -> Self {
        Self {
            width,
            height,
            canvas: vec![vec![glyphs.void; width]; height],
            glyphs,
        }
    }

    /// Draws a single `symbol` on the `Canvas`
    /// NOTE: The `Point`s that lay outside of the `Canvas` are quietly ignored
    fn draw_symbol(&mut self, point: Point, symbol: char) {
        let Point(x, y) = point;
        if x < self.width && y < self.height {
            self.canvas[y][x] = symbol;
        }
    }

    /// Draws a line of `length` from an `origin` along the `orientation`
    fn draw_line(&mut self, origin: Point, orientation: Orientation, length: usize) {
        let Point(x, y) = origin;
        if let Orientation::Vertical = orientation {
            let mut down = 0;
            while down < length {
                self.draw_symbol(Point(x, y + down), self.glyphs.vertical_line);
                down += 1;
            }
        } else if let Orientation::Horizontal = orientation {
            let mut right = 0;
            while right < length {
                self.draw_symbol(Point(x + right, y), self.glyphs.horizontal_line);
                right += 1;
            }
        }
    }

    /// Draws a box of `width` and `height` with an `origin` being the top left corner
    fn draw_box(&mut self, origin: Point, width: usize, height: usize) {
        let Point(x, y) = origin;
        self.draw_symbol(origin, self.glyphs.top_left_corner);
        self.draw_symbol(Point(x + width - 1, y), self.glyphs.top_right_corner);
        self.draw_symbol(Point(x, y + height - 1), self.glyphs.bottom_left_corner);
        self.draw_symbol(
            Point(x + width - 1, y + height - 1),
            self.glyphs.bottom_right_corner,
        );
        self.draw_line(Point(x + 1, y), Orientation::Horizontal, width - 2);
        self.draw_line(
            Point(x + 1, y + height - 1),
            Orientation::Horizontal,
            width - 2,
        );
        self.draw_line(Point(x, y + 1), Orientation::Vertical, height - 2);
        self.draw_line(
            Point(x + width - 1, y + 1),
            Orientation::Vertical,
            height - 2,
        );
    }

    /// Draws a box of height 3 containing a center-aligned text
    fn draw_label_centered(&mut self, center: Point, text: &str) {
        let Point(x, y) = center;
        let half = text.len() / 2 + text.len() % 2;
        if x >= half + 2 || y >= 1 {
            self.draw_box(Point(x - half - 2, y - 1), text.len() + 4, 3);
            for (i, c) in text.chars().enumerate() {
                self.draw_symbol(Point(x - half + i, y), c);
            }
        }
    }

    /// Draws branched lines from a `Point` to multiple `Point`s below
    /// NOTE: the shape of these connections is very specific for this particular kind of the
    /// representation of a tree
    fn draw_connections(&mut self, from: Point, to: &[Point]) {
        let mut start_with_corner = true;
        let Point(mut x_from, y_from) = from;
        for Point(x, y) in to {
            if *x >= x_from && *y >= y_from - 1 {
                if *x == x_from {
                    // if the first connection goes straight below
                    self.draw_line(Point(x_from, y_from), Orientation::Vertical, *y - y_from);
                    x_from += 1;
                } else {
                    if start_with_corner {
                        // if the first or the second connection steers to the right
                        self.draw_symbol(Point(x_from, y_from), self.glyphs.bottom_left_corner);
                        start_with_corner = false;
                        x_from += 1;
                    }
                    let length = *x - x_from;
                    self.draw_line(Point(x_from, y_from), Orientation::Horizontal, length);
                    x_from += length;
                    self.draw_symbol(Point(x_from, y_from), self.glyphs.top_right_corner);
                    self.draw_line(
                        Point(x_from, y_from + 1),
                        Orientation::Vertical,
                        *y - y_from - 1,
                    );
                    x_from += 1;
                }
            }
        }
    }
}

/// The actual drawing happens in the conversion of the intermediate `TreeView` into `Canvas`
impl From<TreeView<'_>> for Canvas {
    fn from(value: TreeView<'_>) -> Self {
        let width = value.n_rows_width + 3 + value.columns.iter().map(|c| c.width).sum::<usize>();
        let height = 3 + 3 * value.n_rows + 3 * (value.n_rows - 1);
        let mut canvas = Canvas::new(width, height, Glyphs::default());

        // Axles
        let (x, y) = (value.n_rows_width + 2, 1);
        canvas.draw_symbol(Point(x, y), '┌');
        canvas.draw_line(Point(x + 1, y), Orientation::Horizontal, width - x);
        canvas.draw_line(Point(x, y + 1), Orientation::Vertical, height - y);

        // Row and column indices
        let mut y_offset = 4;
        for i in 0..value.n_rows {
            // the prefix `Vec` of spaces compensates for the row indices that are shorter than the
            // highest index, effectively, row indices are right-aligned
            for (j, c) in vec![' '; value.n_rows_width - digits(i)]
                .into_iter()
                .chain(format!("{i}").chars())
                .enumerate()
            {
                canvas.draw_symbol(Point(j + 1, y_offset), c);
            }
            y_offset += 6;
        }
        let mut j = 0;
        for col in &value.columns {
            if !col.is_empty {
                // NOTE: the empty columns (i.e. such that don't contain a single populated cell)
                // don't obtain an index
                // the column indices are centered
                let j_width = digits(j);
                let start = col.offset + col.center - (j_width / 2 + j_width % 2);
                for (k, c) in format!("{j}").chars().enumerate() {
                    canvas.draw_symbol(Point(start + k, 0), c);
                }
                j += 1;
            }
        }

        // Non-empty cells (nodes) and their connections (edges)
        let mut y_offset = 3;
        for row in &value.matrix {
            for (j, cell) in row.iter().enumerate() {
                if let Some(text) = cell.text {
                    let x_offset = value.columns[j].offset + value.columns[j].center;
                    let children_points = cell
                        .children_columns
                        .iter()
                        .map(|k| {
                            Point(
                                value.columns[*k].offset + value.columns[*k].center - 1,
                                y_offset + 6,
                            )
                        })
                        .collect::<Vec<_>>();
                    canvas.draw_label_centered(Point(x_offset, y_offset + 1), text);
                    canvas.draw_connections(Point(x_offset - 1, y_offset + 3), &children_points);
                }
            }
            y_offset += 6;
        }

        canvas
    }
}

impl Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.canvas {
            writeln!(f, "{}", row.iter().collect::<String>().trim_end())?;
        }

        Ok(())
    }
}

impl Display for TreeFmtVisitor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tree_view: TreeView<'_> = self.levels.as_slice().into();
        let canvas: Canvas = tree_view.into();
        write!(f, "{canvas}")?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::logical_plan::visitor::TreeWalker;

    #[test]
    fn test_tree_fmt_visit() {
        let e = (col("foo") * lit(2) + lit(3) + lit(43)).sum();
        let mut arena = Default::default();
        let node = to_aexpr(e, &mut arena);

        let mut visitor = TreeFmtVisitor::new();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();
        let expected: &[&[&str]] = &[
            &["sum"],
            &["binary: +"],
            &["lit(43)", "binary: +"],
            &["", "lit(3)", "binary: *"],
            &["", "", "lit(2)", "col(foo)"],
        ];

        assert_eq!(visitor.levels, expected);
    }

    #[test]
    fn test_tree_format_levels() {
        let e = (col("a") + col("b")).pow(2) + col("c") * col("d");
        let mut arena = Default::default();
        let node = to_aexpr(e, &mut arena);

        let mut visitor = TreeFmtVisitor::new();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let expected_lines = vec![
            "           0            1                   2                3            4",
            "   ┌─────────────────────────────────────────────────────────────────────────────",
            "   │",
            "   │ ╭───────────╮",
            " 0 │ │ binary: + │",
            "   │ ╰───────────╯",
            "   │       │╰───────────────────────────────╮",
            "   │       │                                │",
            "   │       │                                │",
            "   │ ╭───────────╮                  ╭───────────────╮",
            " 1 │ │ binary: * │                  │ function: pow │",
            "   │ ╰───────────╯                  ╰───────────────╯",
            "   │       │╰───────────╮                   │╰───────────────╮",
            "   │       │            │                   │                │",
            "   │       │            │                   │                │",
            "   │   ╭────────╮   ╭────────╮          ╭────────╮     ╭───────────╮",
            " 2 │   │ col(d) │   │ col(c) │          │ lit(2) │     │ binary: + │",
            "   │   ╰────────╯   ╰────────╯          ╰────────╯     ╰───────────╯",
            "   │                                                         │╰───────────╮",
            "   │                                                         │            │",
            "   │                                                         │            │",
            "   │                                                     ╭────────╮   ╭────────╮",
            " 3 │                                                     │ col(b) │   │ col(a) │",
            "   │                                                     ╰────────╯   ╰────────╯",
        ];
        for (i, (line, expected_line)) in
            format!("{visitor}").lines().zip(expected_lines).enumerate()
        {
            assert_eq!(line, expected_line, "Difference at line {}", i + 1);
        }
    }

    #[cfg(feature = "range")]
    #[test]
    fn test_tree_format_levels_with_range() {
        let e = (col("a") + col("b")).pow(2)
            + int_range(
                Expr::Literal(LiteralValue::Int64(0)),
                Expr::Literal(LiteralValue::Int64(3)),
                1,
                polars_core::datatypes::DataType::Int64,
            );
        let mut arena = Default::default();
        let node = to_aexpr(e, &mut arena);

        let mut visitor = TreeFmtVisitor::new();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let expected_lines = vec![
            "                0                 1                   2                3            4",
            "   ┌───────────────────────────────────────────────────────────────────────────────────────",
            "   │",
            "   │      ╭───────────╮",
            " 0 │      │ binary: + │",
            "   │      ╰───────────╯",
            "   │            │╰────────────────────────────────────╮",
            "   │            │                                     │",
            "   │            │                                     │",
            "   │ ╭─────────────────────╮                  ╭───────────────╮",
            " 1 │ │ function: int_range │                  │ function: pow │",
            "   │ ╰─────────────────────╯                  ╰───────────────╯",
            "   │            │╰────────────────╮                   │╰───────────────╮",
            "   │            │                 │                   │                │",
            "   │            │                 │                   │                │",
            "   │        ╭────────╮        ╭────────╮          ╭────────╮     ╭───────────╮",
            " 2 │        │ lit(3) │        │ lit(0) │          │ lit(2) │     │ binary: + │",
            "   │        ╰────────╯        ╰────────╯          ╰────────╯     ╰───────────╯",
            "   │                                                                   │╰───────────╮",
            "   │                                                                   │            │",
            "   │                                                                   │            │",
            "   │                                                               ╭────────╮   ╭────────╮",
            " 3 │                                                               │ col(b) │   │ col(a) │",
            "   │                                                               ╰────────╯   ╰────────╯",
        ];
        for (i, (line, expected_line)) in
            format!("{visitor}").lines().zip(expected_lines).enumerate()
        {
            assert_eq!(line, expected_line, "Difference at line {}", i + 1);
        }
    }
}
