use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter, UpperExp};

use polars_core::error::*;
#[cfg(feature = "regex")]
use regex::Regex;

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

pub enum TreeFmtNode<'a> {
    Expression(Option<String>, &'a Expr),
    LogicalPlan(Option<String>, &'a LogicalPlan),
}

struct TreeFmtNodeData<'a>(String, Vec<TreeFmtNode<'a>>);

fn with_header(header: &Option<String>, text: &str) -> String {
    if let Some(header) = header {
        format!("{header}\n{text}")
    } else {
        text.to_string()
    }
}

#[cfg(feature = "regex")]
fn multiline_expression(expr: &str) -> Cow<'_, str> {
    let re = Regex::new(r"([\)\]])(\.[a-z0-9]+\()").unwrap();
    re.replace_all(expr, "$1\n  $2")
}

impl<'a> TreeFmtNode<'a> {
    pub fn root_logical_plan(lp: &'a LogicalPlan) -> Self {
        Self::LogicalPlan(None, lp)
    }

    pub fn traverse(&self, visitor: &mut TreeFmtVisitor) {
        let TreeFmtNodeData(title, child_nodes) = self.node_data();

        if visitor.levels.len() <= visitor.depth {
            visitor.levels.push(vec![]);
        }

        let row = visitor.levels.get_mut(visitor.depth).unwrap();
        row.resize(visitor.width + 1, "".to_string());

        row[visitor.width] = title;
        visitor.prev_depth = visitor.depth;
        visitor.depth += 1;

        for child in &child_nodes {
            child.traverse(visitor);
        }

        visitor.depth -= 1;
        visitor.width += if visitor.prev_depth == visitor.depth {
            1
        } else {
            0
        };
    }

    fn node_data(&self) -> TreeFmtNodeData<'_> {
        use LogicalPlan::*;
        use TreeFmtNode::{Expression as NE, LogicalPlan as NL};
        use {with_header as wh, TreeFmtNodeData as ND};

        match self {
            #[cfg(feature = "regex")]
            NE(h, expr) => ND(wh(h, &multiline_expression(&format!("{expr:?}"))), vec![]),
            #[cfg(not(feature = "regex"))]
            NE(h, expr) => ND(wh(h, &format!("{expr:?}")), vec![]),
            #[cfg(feature = "python")]
            NL(h, lp @ PythonScan { .. }) => ND(wh(h, &format!("{lp:?}",)), vec![]),
            NL(h, lp @ Scan { .. }) => ND(wh(h, &format!("{lp:?}",)), vec![]),
            NL(
                h,
                DataFrameScan {
                    schema,
                    projection,
                    selection,
                    ..
                },
            ) => ND(
                wh(
                    h,
                    &format!(
                        "DF {:?}\nPROJECT {}/{} COLUMNS",
                        schema.iter_names().take(4).collect::<Vec<_>>(),
                        if let Some(columns) = projection {
                            format!("{}", columns.len())
                        } else {
                            "*".to_string()
                        },
                        schema.len()
                    ),
                ),
                if let Some(expr) = selection {
                    vec![NE(Some("SELECTION:".to_string()), expr)]
                } else {
                    vec![]
                },
            ),
            NL(h, Union { inputs, options }) => ND(
                wh(
                    h,
                    &(if let Some(slice) = options.slice {
                        format!("SLICED UNION: {slice:?}")
                    } else {
                        "UNION".to_string()
                    }),
                ),
                inputs
                    .iter()
                    .enumerate()
                    .map(|(i, lp)| NL(Some(format!("PLAN {i}:")), lp))
                    .collect(),
            ),
            NL(h, HConcat { inputs, .. }) => ND(
                wh(h, "HCONCAT"),
                inputs
                    .iter()
                    .enumerate()
                    .map(|(i, lp)| NL(Some(format!("PLAN {i}:")), lp))
                    .collect(),
            ),
            NL(h, Cache { input, id, count }) => ND(
                wh(h, &format!("CACHE[id: {:x}, count: {}]", *id, *count)),
                vec![NL(None, input)],
            ),
            NL(h, Selection { input, predicate }) => ND(
                wh(h, "FILTER"),
                vec![
                    NE(Some("predicate:".to_string()), predicate),
                    NL(Some("FROM:".to_string()), input),
                ],
            ),
            NL(h, Projection { expr, input, .. }) => ND(
                wh(h, "SELECT"),
                expr.iter()
                    .map(|expr| NE(Some("expression:".to_string()), expr))
                    .chain([NL(Some("FROM:".to_string()), input)])
                    .collect(),
            ),
            NL(
                h,
                LogicalPlan::Sort {
                    input, by_column, ..
                },
            ) => ND(
                wh(h, "SORT BY"),
                by_column
                    .iter()
                    .map(|expr| NE(Some("expression:".to_string()), expr))
                    .chain([NL(None, input)])
                    .collect(),
            ),
            NL(
                h,
                Aggregate {
                    input, keys, aggs, ..
                },
            ) => ND(
                wh(h, "AGGREGATE"),
                aggs.iter()
                    .map(|expr| NE(Some("expression:".to_string()), expr))
                    .chain(
                        keys.iter()
                            .map(|expr| NE(Some("aggregate by:".to_string()), expr)),
                    )
                    .chain([NL(Some("FROM:".to_string()), input)])
                    .collect(),
            ),
            NL(
                h,
                Join {
                    input_left,
                    input_right,
                    left_on,
                    right_on,
                    options,
                    ..
                },
            ) => ND(
                wh(h, &format!("{} JOIN", options.args.how)),
                left_on
                    .iter()
                    .map(|expr| NE(Some("left on:".to_string()), expr))
                    .chain([NL(Some("LEFT PLAN:".to_string()), input_left)])
                    .chain(
                        right_on
                            .iter()
                            .map(|expr| NE(Some("right on:".to_string()), expr)),
                    )
                    .chain([NL(Some("RIGHT PLAN:".to_string()), input_right)])
                    .collect(),
            ),
            NL(h, HStack { input, exprs, .. }) => ND(
                wh(h, "WITH_COLUMNS"),
                exprs
                    .iter()
                    .map(|expr| NE(Some("expression:".to_string()), expr))
                    .chain([NL(None, input)])
                    .collect(),
            ),
            NL(h, Distinct { input, options }) => ND(
                wh(h, &format!("UNIQUE BY {:?}", options.subset)),
                vec![NL(None, input)],
            ),
            NL(h, LogicalPlan::Slice { input, offset, len }) => ND(
                wh(h, &format!("SLICE[offset: {offset}, len: {len}]")),
                vec![NL(None, input)],
            ),
            NL(h, MapFunction { input, function }) => {
                ND(wh(h, &format!("{function}")), vec![NL(None, input)])
            },
            NL(h, Error { input, err }) => ND(wh(h, &format!("{err:?}")), vec![NL(None, input)]),
            NL(h, ExtContext { input, .. }) => ND(wh(h, "EXTERNAL_CONTEXT"), vec![NL(None, input)]),
            NL(h, Sink { input, payload }) => ND(
                wh(
                    h,
                    match payload {
                        SinkType::Memory => "SINK (memory)",
                        SinkType::File { .. } => "SINK (file)",
                        #[cfg(feature = "cloud")]
                        SinkType::Cloud { .. } => "SINK (cloud)",
                    },
                ),
                vec![NL(None, input)],
            ),
        }
    }
}

#[derive(Default)]
pub(crate) struct TreeFmtVisitor {
    levels: Vec<Vec<String>>,
    prev_depth: usize,
    depth: usize,
    width: usize,
}

impl Visitor for TreeFmtVisitor {
    type Node = AexprNode;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        let repr = format!("{:E}", ae);

        if self.levels.len() <= self.depth {
            self.levels.push(vec![])
        }

        // the post-visit ensures the width of this node is known
        let row = self.levels.get_mut(self.depth).unwrap();

        // set default values to ensure we format at the right width
        row.resize(self.width + 1, "".to_string());
        row[self.width] = repr;

        // before entering a depth-first branch we preserve the depth to control the width increase
        // in the post-visit
        self.prev_depth = self.depth;

        // we will enter depth first, we enter child so depth increases
        self.depth += 1;

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        // we finished this branch so we decrease in depth, back the caller node
        self.depth -= 1;

        // because we traverse depth first
        // the width is increased once after one or more depth-first branches
        // this way we avoid empty columns in the resulting tree representation
        self.width += if self.prev_depth == self.depth { 1 } else { 0 };

        Ok(VisitRecursion::Continue)
    }
}

/// Calculates the number of digits in a `usize` number
/// Useful for the alignment of `usize` values when they are displayed
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
}

/// Meta-info of a column in a populated `TreeFmtVisitor` required for the pretty-print of a tree
#[derive(Clone, Default, Debug)]
struct TreeViewRow {
    offset: usize,
    height: usize,
    center: usize,
}

/// Meta-info of a cell in a populated `TreeFmtVisitor`
#[derive(Clone, Default, Debug)]
struct TreeViewCell<'a> {
    text: Vec<&'a str>,
    /// A `Vec` of indices of `TreeViewColumn`-s stored elsewhere in another `Vec`
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
    rows: Vec<TreeViewRow>,
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
                    matrix[i][j].text = value[i][j].split('\n').collect();
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

        let mut y_offset = 3;
        let mut rows = vec![TreeViewRow::default(); n_rows];
        for i in 0..n_rows {
            let mut height = 0;
            for j in 0..n_cols {
                height = [matrix[i][j].text.len(), height].into_iter().max().unwrap();
            }
            height += 2;
            rows[i].offset = y_offset;
            rows[i].height = height;
            rows[i].center = height / 2;
            y_offset += height + 3;
        }

        let mut x_offset = n_rows_width + 4;
        let mut columns = vec![TreeViewColumn::default(); n_cols];
        // the two nested loops below are those `needless_range_loop`s
        // more readable this way to my taste
        for j in 0..n_cols {
            let mut width = 0;
            for i in 0..n_rows {
                width = [
                    matrix[i][j].text.iter().map(|l| l.len()).max().unwrap_or(0),
                    width,
                ]
                .into_iter()
                .max()
                .unwrap();
            }
            width += 6;
            columns[j].offset = x_offset;
            columns[j].width = width;
            columns[j].center = width / 2 + width % 2;
            x_offset += width;
        }

        Self {
            n_rows,
            n_rows_width,
            matrix,
            columns,
            rows,
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
    tee_down: char,
    tee_up: char,
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
            tee_down: '┬',
            tee_up: '┴',
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

    /// Draws a box of height `2 + text.len()` containing a left-aligned text
    fn draw_label_centered(&mut self, center: Point, text: &[&str]) {
        if !text.is_empty() {
            let Point(x, y) = center;
            let text_width = text.iter().map(|l| l.len()).max().unwrap();
            let half_width = text_width / 2 + text_width % 2;
            let half_height = text.len() / 2;
            if x >= half_width + 2 && y > half_height {
                self.draw_box(
                    Point(x - half_width - 2, y - half_height - 1),
                    text_width + 4,
                    text.len() + 2,
                );
                for (i, line) in text.iter().enumerate() {
                    for (j, c) in line.chars().enumerate() {
                        self.draw_symbol(Point(x - half_width + j, y - half_height + i), c);
                    }
                }
            }
        }
    }

    /// Draws branched lines from a `Point` to multiple `Point`s below
    /// NOTE: the shape of these connections is very specific for this particular kind of the
    /// representation of a tree
    fn draw_connections(&mut self, from: Point, to: &[Point], branching_offset: usize) {
        let mut start_with_corner = true;
        let Point(mut x_from, mut y_from) = from;
        for (i, Point(x, y)) in to.iter().enumerate() {
            if *x >= x_from && *y >= y_from - 1 {
                self.draw_symbol(Point(*x, *y), self.glyphs.tee_up);
                if *x == x_from {
                    // if the first connection goes straight below
                    self.draw_symbol(Point(x_from, y_from - 1), self.glyphs.tee_down);
                    self.draw_line(Point(x_from, y_from), Orientation::Vertical, *y - y_from);
                    x_from += 1;
                } else {
                    if start_with_corner {
                        // if the first or the second connection steers to the right
                        self.draw_symbol(Point(x_from, y_from - 1), self.glyphs.tee_down);
                        self.draw_line(
                            Point(x_from, y_from),
                            Orientation::Vertical,
                            branching_offset,
                        );
                        y_from += branching_offset;
                        self.draw_symbol(Point(x_from, y_from), self.glyphs.bottom_left_corner);
                        start_with_corner = false;
                        x_from += 1;
                    }
                    let length = *x - x_from;
                    self.draw_line(Point(x_from, y_from), Orientation::Horizontal, length);
                    x_from += length;
                    if i == to.len() - 1 {
                        self.draw_symbol(Point(x_from, y_from), self.glyphs.top_right_corner);
                    } else {
                        self.draw_symbol(Point(x_from, y_from), self.glyphs.tee_down);
                    }
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
        let height =
            3 + value.rows.iter().map(|r| r.height).sum::<usize>() + 3 * (value.n_rows - 1);
        let mut canvas = Canvas::new(width, height, Glyphs::default());

        // Axles
        let (x, y) = (value.n_rows_width + 2, 1);
        canvas.draw_symbol(Point(x, y), '┌');
        canvas.draw_line(Point(x + 1, y), Orientation::Horizontal, width - x);
        canvas.draw_line(Point(x, y + 1), Orientation::Vertical, height - y);

        // Row and column indices
        for (i, row) in value.rows.iter().enumerate() {
            // the prefix `Vec` of spaces compensates for the row indices that are shorter than the
            // highest index, effectively, row indices are right-aligned
            for (j, c) in vec![' '; value.n_rows_width - digits(i)]
                .into_iter()
                .chain(format!("{i}").chars())
                .enumerate()
            {
                canvas.draw_symbol(Point(j + 1, row.offset + row.center), c);
            }
        }
        for (j, col) in value.columns.iter().enumerate() {
            let j_width = digits(j);
            let start = col.offset + col.center - (j_width / 2 + j_width % 2);
            for (k, c) in format!("{j}").chars().enumerate() {
                canvas.draw_symbol(Point(start + k, 0), c);
            }
        }

        // Non-empty cells (nodes) and their connections (edges)
        for (i, row) in value.matrix.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if !cell.text.is_empty() {
                    canvas.draw_label_centered(
                        Point(
                            value.columns[j].offset + value.columns[j].center,
                            value.rows[i].offset + value.rows[i].center,
                        ),
                        &cell.text,
                    );
                }
            }
        }

        fn even_odd(a: usize, b: usize) -> usize {
            if a % 2 == 0 && b % 2 == 1 {
                1
            } else {
                0
            }
        }

        for (i, row) in value.matrix.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if !cell.text.is_empty() && i < value.rows.len() - 1 {
                    let children_points = cell
                        .children_columns
                        .iter()
                        .map(|k| {
                            let child_total_padding =
                                value.rows[i + 1].height - value.matrix[i + 1][*k].text.len() - 2;
                            let even_cell_in_odd_row = even_odd(
                                value.matrix[i + 1][*k].text.len(),
                                value.rows[i + 1].height,
                            );
                            Point(
                                value.columns[*k].offset + value.columns[*k].center - 1,
                                value.rows[i + 1].offset
                                    + child_total_padding / 2
                                    + child_total_padding % 2
                                    - even_cell_in_odd_row,
                            )
                        })
                        .collect::<Vec<_>>();

                    let parent_total_padding =
                        value.rows[i].height - value.matrix[i][j].text.len() - 2;
                    let even_cell_in_odd_row =
                        even_odd(value.matrix[i][j].text.len(), value.rows[i].height);

                    canvas.draw_connections(
                        Point(
                            value.columns[j].offset + value.columns[j].center - 1,
                            value.rows[i].offset + value.rows[i].height
                                - parent_total_padding / 2
                                - even_cell_in_odd_row,
                        ),
                        &children_points,
                        parent_total_padding / 2 + 1 + even_cell_in_odd_row,
                    );
                }
            }
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
        Debug::fmt(self, f)
    }
}

impl Debug for TreeFmtVisitor {
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

        let mut visitor = TreeFmtVisitor::default();

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

        let mut visitor = TreeFmtVisitor::default();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let expected_lines = vec![
            "            0            1               2                3            4",
            "   ┌─────────────────────────────────────────────────────────────────────────",
            "   │",
            "   │  ╭───────────╮",
            " 0 │  │ binary: + │",
            "   │  ╰─────┬┬────╯",
            "   │        ││",
            "   │        │╰───────────────────────────╮",
            "   │        │                            │",
            "   │  ╭─────┴─────╮              ╭───────┴───────╮",
            " 1 │  │ binary: * │              │ function: pow │",
            "   │  ╰─────┬┬────╯              ╰───────┬┬──────╯",
            "   │        ││                           ││",
            "   │        │╰───────────╮               │╰───────────────╮",
            "   │        │            │               │                │",
            "   │    ╭───┴────╮   ╭───┴────╮      ╭───┴────╮     ╭─────┴─────╮",
            " 2 │    │ col(d) │   │ col(c) │      │ lit(2) │     │ binary: + │",
            "   │    ╰────────╯   ╰────────╯      ╰────────╯     ╰─────┬┬────╯",
            "   │                                                      ││",
            "   │                                                      │╰───────────╮",
            "   │                                                      │            │",
            "   │                                                  ╭───┴────╮   ╭───┴────╮",
            " 3 │                                                  │ col(b) │   │ col(a) │",
            "   │                                                  ╰────────╯   ╰────────╯",
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

        let mut visitor = TreeFmtVisitor::default();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        let expected_lines = vec![
            "                 0                 1               2                3            4",
            "   ┌───────────────────────────────────────────────────────────────────────────────────",
            "   │",
            "   │       ╭───────────╮",
            " 0 │       │ binary: + │",
            "   │       ╰─────┬┬────╯",
            "   │             ││",
            "   │             │╰────────────────────────────────╮",
            "   │             │                                 │",
            "   │  ╭──────────┴──────────╮              ╭───────┴───────╮",
            " 1 │  │ function: int_range │              │ function: pow │",
            "   │  ╰──────────┬┬─────────╯              ╰───────┬┬──────╯",
            "   │             ││                                ││",
            "   │             │╰────────────────╮               │╰───────────────╮",
            "   │             │                 │               │                │",
            "   │         ╭───┴────╮        ╭───┴────╮      ╭───┴────╮     ╭─────┴─────╮",
            " 2 │         │ lit(3) │        │ lit(0) │      │ lit(2) │     │ binary: + │",
            "   │         ╰────────╯        ╰────────╯      ╰────────╯     ╰─────┬┬────╯",
            "   │                                                                ││",
            "   │                                                                │╰───────────╮",
            "   │                                                                │            │",
            "   │                                                            ╭───┴────╮   ╭───┴────╮",
            " 3 │                                                            │ col(b) │   │ col(a) │",
            "   │                                                            ╰────────╯   ╰────────╯",
        ];
        for (i, (line, expected_line)) in
            format!("{visitor}").lines().zip(expected_lines).enumerate()
        {
            assert_eq!(line, expected_line, "Difference at line {}", i + 1);
        }
    }
}
