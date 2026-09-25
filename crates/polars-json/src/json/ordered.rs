//! Tape conversion that keeps the source order of `Map` objects.
//!
//! [`Value::Object`] switches to an unordered hash table above 32 members, which loses
//! entry order and duplicate keys. This conversion emits every object at a `Map` position as
//! `[{"key": k, "value": v}, ...]` in tape order instead.
use polars_arrow::datatypes::ArrowDataType;
use polars_error::*;
use polars_utils::aliases::PlHashMap;
use simd_json::borrowed::{Object, Value};
use simd_json::{Node, StaticNode, Tape};

/// The positions of the `Map`s in a target dtype, built once per read.
pub struct TapeGuide<'a>(Guide<'a>);

impl<'a> TapeGuide<'a> {
    pub fn new(dtype: &'a ArrowDataType) -> Self {
        Self(Guide::new(dtype))
    }
}

/// Subtrees without a `Map` are [`Guide::Default`].
enum Guide<'a> {
    Default,
    /// The guide of the map values.
    Map(Box<Guide<'a>>),
    List(Box<Guide<'a>>),
    /// Only the fields that contain a `Map`.
    Struct(PlHashMap<&'a str, Guide<'a>>),
}

impl<'a> Guide<'a> {
    fn new(dtype: &'a ArrowDataType) -> Self {
        match dtype.to_storage() {
            ArrowDataType::Map(field, _) => {
                let ArrowDataType::Struct(entry_fields) = field.dtype() else {
                    unreachable!("map entries are a struct")
                };
                Guide::Map(Box::new(Guide::new(entry_fields[1].dtype())))
            },
            ArrowDataType::LargeList(field)
            | ArrowDataType::List(field)
            | ArrowDataType::FixedSizeList(field, _) => match Guide::new(field.dtype()) {
                Guide::Default => Guide::Default,
                inner => Guide::List(Box::new(inner)),
            },
            ArrowDataType::Struct(fields) => {
                let fields: PlHashMap<_, _> = fields
                    .iter()
                    .filter_map(|f| match Guide::new(f.dtype()) {
                        Guide::Default => None,
                        guide => Some((f.name.as_str(), guide)),
                    })
                    .collect();
                if fields.is_empty() {
                    Guide::Default
                } else {
                    Guide::Struct(fields)
                }
            },
            _ => Guide::Default,
        }
    }
}

/// Convert `tape` to a value, keeping the source order of the `Map`s located by `guide`.
///
/// With `ignore_errors`, a non-object at a `Map` position becomes null.
pub fn tape_to_value<'i>(
    tape: &Tape<'i>,
    guide: &TapeGuide,
    ignore_errors: bool,
) -> PolarsResult<Value<'i>> {
    Converter::new(tape, ignore_errors).convert(&guide.0)
}

/// The same conversion as simd-json's `to_borrowed_value`.
pub fn tape_to_default_value<'i>(tape: &Tape<'i>) -> Value<'i> {
    Converter::new(tape, false).convert_default()
}

struct Converter<'t, 'i> {
    nodes: &'t [Node<'i>],
    idx: usize,
    ignore_errors: bool,
}

impl<'t, 'i> Converter<'t, 'i> {
    fn new(tape: &'t Tape<'i>, ignore_errors: bool) -> Self {
        Self {
            nodes: &tape.0,
            idx: 0,
            ignore_errors,
        }
    }

    fn next(&mut self) -> Node<'i> {
        let node = self.nodes[self.idx];
        self.idx += 1;
        node
    }

    fn key(&mut self) -> &'i str {
        match self.next() {
            Node::String(key) => key,
            _ => unreachable!("object keys are strings"),
        }
    }

    fn convert(&mut self, guide: &Guide) -> PolarsResult<Value<'i>> {
        if let Guide::Default = guide {
            return Ok(self.convert_default());
        }

        let node = self.next();
        match (guide, node) {
            (Guide::Map(value_guide), Node::Object { len, .. }) => {
                let mut entries = Vec::with_capacity(len);
                for _ in 0..len {
                    let key = self.key();
                    let value = self.convert(value_guide)?;
                    // Two members stay in halfbrown's insertion-ordered vec mode.
                    let mut entry = Object::with_capacity_and_hasher(2, Default::default());
                    entry.insert("key".into(), Value::String(key.into()));
                    entry.insert("value".into(), value);
                    entries.push(Value::Object(Box::new(entry)));
                }
                Ok(Value::Array(Box::new(entries)))
            },
            (Guide::Map(_), Node::Static(StaticNode::Null)) => Ok(Value::Static(StaticNode::Null)),
            (Guide::Map(_), node) => {
                polars_ensure!(
                    self.ignore_errors,
                    ComputeError: "expected a JSON object for a Map, got {:?}", node.value_type()
                );
                self.skip(node);
                Ok(Value::Static(StaticNode::Null))
            },
            (Guide::Struct(fields), Node::Object { len, .. }) => {
                let mut object = Object::with_capacity_and_hasher(len, Default::default());
                for _ in 0..len {
                    let key = self.key();
                    let value = self.convert(fields.get(key).unwrap_or(&Guide::Default))?;
                    // SAFETY: duplicate keys are memory-safe; this mirrors simd-json.
                    unsafe { object.insert_nocheck(key.into(), value) };
                }
                Ok(Value::Object(Box::new(object)))
            },
            (Guide::List(inner), Node::Array { len, .. }) => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.convert(inner)?);
                }
                Ok(Value::Array(Box::new(values)))
            },
            _ => {
                self.idx -= 1;
                Ok(self.convert_default())
            },
        }
    }

    /// Skip the rest of the subtree whose head `node` was just consumed.
    fn skip(&mut self, node: Node<'i>) {
        if let Node::Object { count, .. } | Node::Array { count, .. } = node {
            self.idx += count;
        }
    }

    fn convert_default(&mut self) -> Value<'i> {
        match self.next() {
            Node::Static(s) => Value::Static(s),
            Node::String(s) => Value::String(s.into()),
            Node::Array { len, .. } => {
                Value::Array(Box::new((0..len).map(|_| self.convert_default()).collect()))
            },
            Node::Object { len, .. } => {
                let mut object = Object::with_capacity_and_hasher(len, Default::default());
                for _ in 0..len {
                    let key = self.key();
                    let value = self.convert_default();
                    // SAFETY: duplicate keys are memory-safe; this mirrors simd-json.
                    unsafe { object.insert_nocheck(key.into(), value) };
                }
                Value::Object(Box::new(object))
            },
        }
    }
}
