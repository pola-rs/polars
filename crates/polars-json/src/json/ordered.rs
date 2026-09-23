//! Tape conversion that keeps the source order of `Map` objects.
//!
//! [`Value::Object`] switches to an unordered hash table above 32 members, which loses
//! entry order and duplicate keys. This conversion emits every object at a `Map` position as
//! `[{"key": k, "value": v}, ...]` in tape order instead.
use polars_arrow::datatypes::ArrowDataType;
use polars_error::*;
use simd_json::borrowed::{Object, Value};
use simd_json::{Node, StaticNode, Tape};

/// Convert `tape` to a value, using `guide` (the target dtype) to locate `Map`s.
///
/// With `ignore_errors`, a non-object at a `Map` position becomes null.
pub fn tape_to_value<'i>(
    tape: &Tape<'i>,
    guide: &ArrowDataType,
    ignore_errors: bool,
) -> PolarsResult<Value<'i>> {
    let mut converter = Converter {
        nodes: &tape.0,
        idx: 0,
        ignore_errors,
    };
    converter.convert(Some(guide))
}

/// Whether `dtype` contains a `Map`, i.e. whether a JSON reader needs [`tape_to_value`].
pub fn contains_map(dtype: &ArrowDataType) -> bool {
    match dtype.to_storage() {
        ArrowDataType::Map(_, _) => true,
        ArrowDataType::LargeList(field)
        | ArrowDataType::List(field)
        | ArrowDataType::FixedSizeList(field, _) => contains_map(field.dtype()),
        ArrowDataType::Struct(fields) => fields.iter().any(|f| contains_map(f.dtype())),
        _ => false,
    }
}

struct Converter<'t, 'i> {
    nodes: &'t [Node<'i>],
    idx: usize,
    ignore_errors: bool,
}

impl<'i> Converter<'_, 'i> {
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

    fn convert(&mut self, guide: Option<&ArrowDataType>) -> PolarsResult<Value<'i>> {
        let guide = guide
            .map(ArrowDataType::to_storage)
            .filter(|guide| contains_map(guide));
        let Some(guide) = guide else {
            return Ok(self.convert_default());
        };

        let node = self.next();
        match (guide, node) {
            (ArrowDataType::Map(field, _), Node::Object { len, .. }) => {
                let ArrowDataType::Struct(entry_fields) = field.dtype() else {
                    unreachable!("map entries are a struct")
                };
                let value_guide = entry_fields[1].dtype();
                let mut entries = Vec::with_capacity(len);
                for _ in 0..len {
                    let key = self.key();
                    let value = self.convert(Some(value_guide))?;
                    // Two members stay in halfbrown's insertion-ordered vec mode.
                    let mut entry = Object::with_capacity_and_hasher(2, Default::default());
                    entry.insert("key".into(), Value::String(key.into()));
                    entry.insert("value".into(), value);
                    entries.push(Value::Object(Box::new(entry)));
                }
                Ok(Value::Array(Box::new(entries)))
            },
            (ArrowDataType::Map(_, _), Node::Static(StaticNode::Null)) => {
                Ok(Value::Static(StaticNode::Null))
            },
            (ArrowDataType::Map(_, _), node) => {
                polars_ensure!(
                    self.ignore_errors,
                    ComputeError: "expected a JSON object for a Map, got {:?}", node.value_type()
                );
                self.skip(node);
                Ok(Value::Static(StaticNode::Null))
            },
            (ArrowDataType::Struct(fields), Node::Object { len, .. }) => {
                let mut object = Object::with_capacity_and_hasher(len, Default::default());
                for _ in 0..len {
                    let key = self.key();
                    let field_guide = fields.iter().find(|f| f.name == key).map(|f| f.dtype());
                    let value = self.convert(field_guide)?;
                    // SAFETY: duplicate keys are memory-safe; this mirrors simd-json.
                    unsafe { object.insert_nocheck(key.into(), value) };
                }
                Ok(Value::Object(Box::new(object)))
            },
            (
                ArrowDataType::LargeList(field)
                | ArrowDataType::List(field)
                | ArrowDataType::FixedSizeList(field, _),
                Node::Array { len, .. },
            ) => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.convert(Some(field.dtype()))?);
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

    /// The same conversion as simd-json's `to_borrowed_value`.
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
