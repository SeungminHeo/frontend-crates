// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A.X-K3 native tool calls: a port of the model team's `axk3_tool_parser.py`.
//!
//! ```text
//! <tool_call>NAME<|parameter|>KEY
//! VALUE<|/parameter|><|parameter|>KEY2
//! VALUE2<|/parameter|></tool_call>
//! ```
//!
//! * A key runs to the first newline after `<|parameter|>`; the value is
//!   everything after that newline up to `<|/parameter|>`, verbatim. String
//!   values are written raw; every other value is written as JSON.
//! * Values are decoded with the tool's JSON schema: a `string` parameter keeps
//!   the raw text; anything else is parsed as JSON. Undescribed parameters are
//!   parsed as JSON when that yields a non-string, else kept as text. In the
//!   arguments JSON, string values are escaped and every other value keeps the
//!   model's own spelling (`1.50` stays `1.50`).
//! * Malformed output: a key line that reaches a tag before its newline ends at
//!   that tag and is repaired from the schema (`top_k30` -> `top_k`, `30`) or
//!   dropped; a value missing `<|/parameter|>` runs to the next `</tool_call>`;
//!   a call without `</tool_call>` is never a tool call and its text is content.
//! * Several calls (`call_group_policy = "all"`): every complete, named call is
//!   executed; non-whitespace text between and after them is content.
//! * No call is looked for inside a `<think>` block seen outside a call.
//!
//! The batch functions here are the single source of truth. The unified parser
//! (`unified::ax_k3`) streams the text before the first call, sends each call
//! once its `</tool_call>` arrives ([`next_closed_call`]), and settles the rest
//! with [`resolve`] at end of stream.

use serde_json::{Map, Value};

use crate::tool_calling::scan::{
    BareRecoveryLatch, GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary,
    InvokeBoundaryFactory, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

pub(crate) const BLOCK_START: &str = "<tool_call>";
pub(crate) const BLOCK_END: &str = "</tool_call>";
pub(crate) const PARAM_START: &str = "<|parameter|>";
pub(crate) const PARAM_END: &str = "<|/parameter|>";
pub(crate) const THINK_START: &str = "<think>";
pub(crate) const THINK_END: &str = "</think>";

const TAGS_IN_CALL: [&str; 3] = [PARAM_START, PARAM_END, BLOCK_END];
const JSON_WS: [char; 4] = [' ', '\t', '\n', '\r'];

// ---------------------------------------------------------------------------
// Parsing (pure functions; the single source of truth)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Param {
    key: String,
    raw: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Call {
    /// `None` until terminated by a tag; `Some("")` is an empty name.
    name: Option<String>,
    params: Vec<Param>,
    closed: bool,
    /// Index of `<tool_call>`.
    start: usize,
    /// Index just past `</tool_call>` (when closed).
    end: usize,
}

fn first_tag(text: &str, pos: usize, tags: &[&str]) -> Option<usize> {
    tags.iter()
        .filter_map(|tag| text[pos..].find(tag).map(|at| pos + at))
        .min()
}

/// A `<|parameter|>` whose key line reached a tag before its newline.
/// `Some((key, value))` when the value is recoverable, else `None` (dropped).
fn repair_param(
    segment: &str,
    properties: Option<&Map<String, Value>>,
) -> Option<(String, String)> {
    let text = segment.trim_start();
    if text.trim().is_empty() {
        return None;
    }
    // The newline between a known key and its value was dropped: "top_k30".
    let best = properties?
        .keys()
        .filter(|key| !key.is_empty() && text.starts_with(key.as_str()))
        .max_by_key(|key| key.len())?;
    let value = &text[best.len()..];
    if value.trim().is_empty() {
        return None; // only a key: no value to execute with
    }
    let value = value.strip_prefix([' ', '\t']).unwrap_or(value);
    Some((best.clone(), value.to_string()))
}

/// Parse one call whose body starts at `text[body]` (just after `<tool_call>`).
/// `whole` is the reference's `final`: the text is complete, so a value missing
/// `<|/parameter|>` may run to `</tool_call>`. On a stream prefix (`!whole`) a
/// call closes only at a `</tool_call>` between parameters, and then parses the
/// same however the text continues.
fn parse_call(text: &str, body: usize, tools: &[Tool], whole: bool) -> (Call, usize) {
    let n = text.len();
    let mut call = Call {
        start: body - BLOCK_START.len(),
        ..Call::default()
    };
    let Some(name_end) = first_tag(text, body, &[PARAM_START, BLOCK_END]) else {
        return (call, n);
    };
    let name = text[body..name_end].trim().to_string();
    let properties = if name.is_empty() {
        None
    } else {
        tool_properties(tools, &name)
    };
    call.name = Some(name);
    let mut pos = name_end;

    loop {
        while pos < n && text[pos..].starts_with(JSON_WS) {
            pos += 1;
        }
        if pos >= n {
            return (call, n);
        }
        if text[pos..].starts_with(BLOCK_END) {
            call.closed = true;
            call.end = pos + BLOCK_END.len();
            let end = call.end;
            return (call, end);
        }
        if text[pos..].starts_with(PARAM_START) {
            let key_start = pos + PARAM_START.len();
            let newline = text[key_start..].find('\n').map(|at| key_start + at);
            let tag = first_tag(text, key_start, &TAGS_IN_CALL);
            if let Some(tag) = tag.filter(|tag| newline.is_none_or(|newline| *tag < newline)) {
                // The key line hit a tag: never let a key swallow it.
                let repaired = repair_param(&text[key_start..tag], properties);
                tracing::warn!(
                    why = "ax_k3_key_without_newline",
                    repaired = repaired.is_some(),
                    "A.X-K3 tool call parameter has no newline before a tag"
                );
                if let Some((key, raw)) = repaired {
                    call.params.push(Param { key, raw });
                }
                pos = if text[tag..].starts_with(PARAM_END) {
                    tag + PARAM_END.len()
                } else {
                    tag
                };
                continue;
            }
            let Some(newline) = newline else {
                return (call, n);
            };
            let key = text[key_start..newline].trim().to_string();
            let value_start = newline + 1;
            if let Some(value_end) = text[value_start..].find(PARAM_END) {
                let value_end = value_start + value_end;
                call.params.push(Param {
                    key,
                    raw: text[value_start..value_end].to_string(),
                });
                pos = value_end + PARAM_END.len();
                continue;
            }
            // Tolerate a missing <|/parameter|>: the value runs to the next
            // </tool_call>.
            if let Some(call_end) = text[value_start..].find(BLOCK_END).filter(|_| whole) {
                let call_end = value_start + call_end;
                call.params.push(Param {
                    key,
                    raw: text[value_start..call_end].to_string(),
                });
                call.closed = true;
                call.end = call_end + BLOCK_END.len();
                let end = call.end;
                return (call, end);
            }
            return (call, n);
        }
        // Stray text between tags: skip to the next tag.
        match first_tag(text, pos, &[PARAM_START, BLOCK_END]) {
            Some(next) => pos = next,
            None => return (call, n),
        }
    }
}

struct Scan {
    /// In order; ends with the unclosed call, if any.
    calls: Vec<Call>,
    /// `gaps[0]` before the first call, `gaps[k]` before call `k`.
    gaps: Vec<String>,
}

/// Split text into calls and the text around them. Tool calls are not looked
/// for inside a `<think>` block that starts outside a call.
fn scan(text: &str, tools: &[Tool]) -> Scan {
    let mut calls = Vec::new();
    let mut gaps = Vec::new();
    let (mut gap_start, mut pos) = (0, 0);
    while let Some(call_at) = next_call_start(text, pos) {
        gaps.push(text[gap_start..call_at].to_string());
        let (call, end) = parse_call(text, call_at + BLOCK_START.len(), tools, true);
        let closed = call.closed;
        calls.push(call);
        if !closed {
            return Scan { calls, gaps };
        }
        gap_start = end;
        pos = end;
    }
    gaps.push(text[gap_start..].to_string());
    Scan { calls, gaps }
}

/// The next `<tool_call>` at or after `pos`, skipping `<think>` blocks; `None`
/// also inside an unclosed `<think>` (the rest is not searched).
fn next_call_start(text: &str, mut pos: usize) -> Option<usize> {
    while pos < text.len() {
        let call_at = text[pos..].find(BLOCK_START).map(|at| pos + at);
        let think_at = text[pos..].find(THINK_START).map(|at| pos + at);
        match think_at.filter(|think| call_at.is_none_or(|call| *think < call)) {
            Some(think_at) => {
                let body = think_at + THINK_START.len();
                pos = body + text[body..].find(THINK_END)? + THINK_END.len();
            }
            None => return call_at,
        }
    }
    None
}

/// One executed call: its name and arguments JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedCall {
    pub name: String,
    pub arguments: String,
}

/// The reference `extract_tool_calls` under `call_group_policy = "all"`: the
/// executed calls and the content. Content is the text before the first call,
/// the non-whitespace text between and after closed calls, and an unclosed
/// call verbatim; with no executed call the whole text is content.
pub(crate) fn resolve(text: &str, tools: &[Tool]) -> (Vec<ResolvedCall>, String) {
    if !text.contains(BLOCK_START) {
        return (Vec::new(), text.to_string());
    }
    let scan = scan(text, tools);
    let closed = scan.calls.iter().filter(|call| call.closed).count();
    let calls: Vec<ResolvedCall> = scan
        .calls
        .iter()
        .filter(|call| call.closed)
        .filter_map(|call| executed(call, tools))
        .collect();
    if calls.is_empty() {
        return (Vec::new(), text.to_string());
    }
    // `gaps[k]` precedes call `k`; with every call closed the last gap is the
    // text after them, otherwise the last closed gap precedes the unclosed call.
    let mut content = scan.gaps[0].clone();
    for gap in scan.gaps.iter().skip(1).take(closed) {
        if !gap.trim().is_empty() {
            content.push_str(gap);
        }
    }
    if let Some(unclosed) = scan.calls.last().filter(|call| !call.closed) {
        content.push_str(&text[unclosed.start..]);
    }
    if content.trim().is_empty() {
        content.clear();
    }
    (calls, content)
}

/// A closed call as executed: `None` for an empty name.
fn executed(call: &Call, tools: &[Tool]) -> Option<ResolvedCall> {
    let name = call.name.as_deref().filter(|name| !name.is_empty())?;
    Some(ResolvedCall {
        name: name.to_string(),
        arguments: render_arguments(call, tool_properties(tools, name)),
    })
}

/// A call closed in a stream prefix: the text before it runs from the search
/// start to `start`, and the call ends at `end`.
pub(crate) struct ClosedCall {
    pub start: usize,
    pub end: usize,
    /// `None` for a call with an empty name (not executed).
    pub call: Option<ResolvedCall>,
}

/// The next call after `from` in a stream prefix, if its `</tool_call>` has
/// arrived. It is the call [`resolve`] finds there once the text is whole.
pub(crate) fn next_closed_call(text: &str, from: usize, tools: &[Tool]) -> Option<ClosedCall> {
    let start = next_call_start(text, from)?;
    let (call, end) = parse_call(text, start + BLOCK_START.len(), tools, false);
    call.closed.then(|| ClosedCall {
        start,
        end,
        call: executed(&call, tools),
    })
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

fn tool_properties<'a>(tools: &'a [Tool], name: &str) -> Option<&'a Map<String, Value>> {
    tools
        .iter()
        .find(|tool| tool.name == name)
        .and_then(|tool| tool.parameters.get("properties"))
        .and_then(Value::as_object)
}

/// Python `json.dumps(value, ensure_ascii=False)`: `", "` and `": "`
/// separators. Object keys keep their parsed order when serde_json's
/// `preserve_order` is enabled (it is in Dynamo builds).
fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(value, &mut out);
    out
}

fn write_py_json(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (at, item) in items.iter().enumerate() {
                if at > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (at, (key, item)) in entries.iter().enumerate() {
                if at > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).expect("strings serialize"));
                out.push_str(": ");
                write_py_json(item, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&serde_json::to_string(scalar).expect("scalars serialize")),
    }
}

/// Complete arguments: `{"k": v, ...}` with `json.dumps` separators. The first
/// occurrence of a duplicated key wins.
fn render_arguments(call: &Call, properties: Option<&Map<String, Value>>) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut pieces = Vec::with_capacity(call.params.len());
    for param in &call.params {
        if !seen.insert(param.key.as_str()) {
            continue;
        }
        let schema = properties.and_then(|properties| properties.get(&param.key));
        pieces.push(format!(
            "{}: {}",
            dumps(&Value::String(param.key.clone())),
            encode_value(&param.raw, schema)
        ));
    }
    format!("{{{}}}", pieces.join(", "))
}

type TypeSet = std::collections::BTreeSet<String>;

/// JSON types a property schema allows, or `None` when it does not say.
fn schema_types(schema: &Value) -> Option<TypeSet> {
    let object = schema.as_object()?;
    if !["type", "enum", "anyOf", "oneOf", "allOf"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        return None;
    }
    Some(extract_types_from_schema(schema))
}

/// vLLM `extract_types_from_schema`.
fn extract_types_from_schema(schema: &Value) -> TypeSet {
    let mut types = TypeSet::new();
    let Some(object) = schema.as_object() else {
        types.insert("string".into());
        return types;
    };
    match object.get("type") {
        Some(Value::String(kind)) => {
            types.insert(kind.clone());
        }
        Some(Value::Array(kinds)) => {
            types.extend(kinds.iter().filter_map(Value::as_str).map(str::to_string));
        }
        _ => {}
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        for value in values {
            types.insert(value_type(value).into());
        }
    }
    for union in ["anyOf", "oneOf", "allOf"] {
        for choice in object
            .get(union)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            types.extend(extract_types_from_schema(choice));
        }
    }
    if types.is_empty() {
        types.insert("string".into());
    }
    types
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn json_type_ok(value: &Value, types: &TypeSet) -> bool {
    let has = |kind: &str| types.contains(kind);
    match value {
        Value::Null => has("null"),
        Value::Bool(_) => has("boolean"),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            has("integer") || has("number")
        }
        Value::Number(number) => {
            has("number")
                || (has("integer") && number.as_f64().is_some_and(|float| float.fract() == 0.0))
        }
        Value::String(_) => has("string"),
        Value::Array(_) => has("array"),
        Value::Object(_) => has("object"),
    }
}

/// `(value, verbatim)`: `verbatim` is the model's own JSON text for the value
/// (padding included) when the value was taken from it as is.
fn decode_value_text<'a>(raw: &'a str, schema: Option<&Value>) -> (Value, Option<&'a str>) {
    let types = schema.and_then(schema_types);
    if types
        .as_ref()
        .is_some_and(|types| types.len() == 1 && types.contains("string"))
    {
        return (Value::String(raw.to_string()), None); // written verbatim
    }
    let text = raw.trim_matches(JSON_WS);
    let parsed = serde_json::from_str::<Value>(text).ok();
    let Some(types) = types else {
        // Not described by the schema.
        return match parsed {
            Some(value) if !value.is_string() => (value, Some(raw)),
            _ => (Value::String(raw.to_string()), None),
        };
    };
    if let Some(value) = parsed.filter(|value| !value.is_string() && json_type_ok(value, &types)) {
        return (value, Some(raw));
    }
    if types.contains("string") {
        return (Value::String(raw.to_string()), None);
    }
    (coerce_to_schema_type(text, &types), None)
}

/// JSON text of a parameter value inside the arguments object: a non-string
/// value keeps the model's own spelling instead of being re-serialized.
fn encode_value(raw: &str, schema: Option<&Value>) -> String {
    match decode_value_text(raw, schema) {
        (_, Some(verbatim)) => verbatim.to_string(),
        (value, None) => dumps(&value),
    }
}

fn normalized_type(kind: &str) -> &'static str {
    match kind.trim().to_ascii_lowercase().as_str() {
        "null" => "null",
        "string" | "str" | "text" | "varchar" | "char" | "enum" => "string",
        "integer" | "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short"
        | "unsigned" => "integer",
        "number" | "float" | "float32" | "float64" | "double" => "number",
        "boolean" | "bool" => "boolean",
        "object" | "dict" => "object",
        "array" | "arr" | "list" | "sequence" => "array",
        _ => "",
    }
}

/// vLLM `coerce_to_schema_type`: null > integer > number > boolean > object >
/// array > string, then a JSON parse, then the text itself.
fn coerce_to_schema_type(text: &str, types: &TypeSet) -> Value {
    let types: std::collections::HashSet<&str> =
        types.iter().map(|kind| normalized_type(kind)).collect();
    for candidate in [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ] {
        if !types.contains(candidate) {
            continue;
        }
        match candidate {
            "null" if text.eq_ignore_ascii_case("null") => return Value::Null,
            "integer" => {
                if let Ok(integer) = text.trim().parse::<i64>() {
                    return Value::from(integer);
                }
                if let Ok(integer) = text.trim().parse::<u64>() {
                    return Value::from(integer);
                }
            }
            "number" => {
                if let Some(float) = text.trim().parse::<f64>().ok().filter(|f| f.is_finite()) {
                    if float.fract() == 0.0 && float.abs() < 9_007_199_254_740_992.0 {
                        return Value::from(float as i64);
                    }
                    if let Some(number) = serde_json::Number::from_f64(float) {
                        return Value::Number(number);
                    }
                }
            }
            "boolean" => match text.trim().to_ascii_lowercase().as_str() {
                "true" | "1" => return Value::Bool(true),
                "false" | "0" => return Value::Bool(false),
                _ => {}
            },
            "object" | "array" => {
                if let Ok(value) = serde_json::from_str::<Value>(text) {
                    return value;
                }
            }
            "string" => return Value::String(text.to_string()),
            _ => {}
        }
    }
    serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

// ---------------------------------------------------------------------------
// Guided-decoding grammar: the shared guided reader locates native markup
// with this boundary. Native parsing itself goes through `resolve`.
// ---------------------------------------------------------------------------

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "ax_k3",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: BLOCK_START.to_string(),
        invoke_end: BLOCK_END.to_string(),
        orphan_markers: [BLOCK_END, PARAM_START, PARAM_END]
            .into_iter()
            .map(str::to_string)
            .collect(),
        holdback_markers: [BLOCK_START, BLOCK_END, PARAM_START, PARAM_END]
            .into_iter()
            .map(str::to_string)
            .collect(),
        bare_recovery_latch: BareRecoveryLatch::Clear,
        invoke_latch: InvokeLatch::IfEmitted,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(ax_k3_boundary)),
        preserve_special_tokens: true,
        ..Default::default()
    }
}

pub(crate) struct AxK3Emitter {
    tools: Vec<Tool>,
}

impl InvokeEmitter for AxK3Emitter {
    /// `invoke` is a call body plus its closer (the opener was consumed).
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let (calls, _) = resolve(&format!("{BLOCK_START}{invoke}"), &self.tools);
        Ok(calls.into_iter().next().map(|call| ToolCallDelta {
            tool_index,
            name: Some(call.name),
            arguments: call.arguments,
            complete: true,
        }))
    }
}

#[derive(Default)]
struct AxK3BoundaryProgress {
    cursor: usize,
    in_key: bool,
    in_value: bool,
}

impl AxK3BoundaryProgress {
    /// Offset just past the `</tool_call>` that closes the body, as
    /// `parse_call` reads it.
    fn end(&mut self, text: &str, flush: bool) -> Option<usize> {
        while self.cursor < text.len() {
            let rest = &text[self.cursor..];
            let partial = |marker: &str| rest.len() < marker.len() && marker.starts_with(rest);
            if self.in_value {
                if rest.starts_with(PARAM_END) {
                    self.in_value = false;
                    self.cursor += PARAM_END.len();
                    continue;
                }
                if !flush && partial(PARAM_END) {
                    return None;
                }
            } else if self.in_key {
                if rest.starts_with('\n') {
                    self.in_key = false;
                    self.in_value = true;
                } else if rest.starts_with(PARAM_END) {
                    self.in_key = false;
                    self.cursor += PARAM_END.len();
                    continue;
                } else if rest.starts_with(PARAM_START) || rest.starts_with(BLOCK_END) {
                    self.in_key = false;
                    continue;
                } else if !flush && TAGS_IN_CALL.iter().any(|tag| partial(tag)) {
                    return None;
                }
            } else {
                if rest.starts_with(PARAM_START) {
                    self.in_key = true;
                    self.cursor += PARAM_START.len();
                    continue;
                }
                if rest.starts_with(BLOCK_END) {
                    return Some(self.cursor + BLOCK_END.len());
                }
                if !flush && (partial(PARAM_START) || partial(BLOCK_END)) {
                    return None;
                }
            }
            let ch = rest.chars().next().expect("cursor is before text end");
            self.cursor += ch.len_utf8();
        }
        None
    }
}

#[derive(Default)]
struct AxK3Boundary {
    native: AxK3BoundaryProgress,
    guided: AxK3BoundaryProgress,
}

impl InvokeBoundary for AxK3Boundary {
    fn block_is_invoke(&self) -> bool {
        true
    }

    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_invoke_at(&self, text: &str) -> Option<(usize, usize)> {
        text.find(BLOCK_START).map(|at| (at, BLOCK_START.len()))
    }

    fn is_guided_invoke_marker(&self, marker: &str) -> bool {
        marker == BLOCK_START
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let body = candidate.strip_prefix(BLOCK_START)?;
        if body.trim_start().starts_with(['{', '[']) {
            return Some(GuidedInvokePrefix::Match(BLOCK_START.len()));
        }
        if !context.outside_reasoning {
            return Some(GuidedInvokePrefix::Strip(BLOCK_START.len()));
        }
        self.guided
            .end(body, false)
            .map(|end| GuidedInvokePrefix::Strip(BLOCK_START.len() + end))
            .or(Some(GuidedInvokePrefix::Pending))
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        self.native.end(candidate, flush)
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn owns_prose_invoke(&self, text: &str, at: usize) -> bool {
        let Some(body) = text[at..].strip_prefix(BLOCK_START) else {
            return false;
        };
        !body.split('<').next().unwrap_or(body).trim().is_empty()
    }

    fn holdback(&self, text: &str) -> usize {
        marker_prefix_suffix_len(text, [BLOCK_END, PARAM_START, PARAM_END])
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn ax_k3_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(AxK3Boundary::default())
}

/// The A.X-K3 marker grammar, used for guided-decoding wiring.
pub(crate) fn ax_k3_scanner(tools: &[Tool]) -> WrappedBlockScanner<AxK3Emitter> {
    WrappedBlockScanner::new(
        spec(),
        AxK3Emitter {
            tools: tools.to_vec(),
        },
    )
}

/// Tool-only projection of the unified A.X-K3 parser, starting in the visible
/// response (no reasoning channel).
pub struct AxK3ToolStreamParser {
    parser: Box<dyn crate::unified::UnifiedParser>,
}

impl AxK3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        let mut parser = crate::unified::ax_k3::ax_k3_unified(tools);
        parser
            .initialize_request(crate::unified::UnifiedParserInit {
                starting_state: crate::unified::UnifiedParserStartingState::Response,
                ..Default::default()
            })
            .expect("native response initialization is always valid");
        Self { parser }
    }

    fn project(output: crate::unified::UnifiedParserOutput) -> ToolParseResult {
        let mut result = ToolParseResult::default();
        for event in output.events {
            match event {
                crate::unified::UnifiedParserEvent::Text(text)
                | crate::unified::UnifiedParserEvent::Reasoning(text) => {
                    result.normal_text.push_str(&text)
                }
                crate::unified::UnifiedParserEvent::ToolCall(call) => result.calls.push(call),
                _ => {}
            }
        }
        result
    }
}

impl ToolParser for AxK3ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        let mut output = crate::unified::UnifiedParserOutput::default();
        self.parser.parse_into(chunk, &mut output)?;
        Ok(Self::project(output))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(Self::project(self.parser.finish()?))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Reference-parser behaviour; arguments are compared both as bytes
    //! (verbatim spelling) and as JSON.
    use super::*;
    use serde_json::json;

    pub(crate) fn tools() -> Vec<Tool> {
        let tool = |name: &str, parameters: Value| Tool {
            name: name.into(),
            description: None,
            parameters,
            strict: None,
        };
        vec![
            tool(
                "write_file",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "options": {"type": "object"}
                }, "required": ["path", "content"]}),
            ),
            tool(
                "typed",
                json!({"type": "object", "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "f": {"type": "number"},
                    "b": {"type": "boolean"},
                    "arr": {"type": "array", "items": {"type": "string"}},
                    "obj": {"type": "object"},
                    "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "untyped": {"description": "no type given"},
                    "top_k": {"type": "integer"},
                    "top": {"type": "string"}
                }}),
            ),
            tool("ping", json!({"type": "object", "properties": {}})),
        ]
    }

    pub(crate) fn call(name: &str, params: &[(&str, &str)]) -> String {
        let body: String = params
            .iter()
            .map(|(key, value)| format!("{PARAM_START}{key}\n{value}{PARAM_END}"))
            .collect();
        format!("{BLOCK_START}{name}{body}{BLOCK_END}")
    }

    fn run(text: &str) -> (Vec<(String, Value)>, String) {
        let (calls, content) = resolve(text, &tools());
        (
            calls
                .into_iter()
                .map(|call| {
                    (
                        call.name,
                        serde_json::from_str(&call.arguments).expect("JSON"),
                    )
                })
                .collect(),
            content,
        )
    }

    fn raw_arguments(text: &str) -> Vec<String> {
        resolve(text, &tools())
            .0
            .into_iter()
            .map(|call| call.arguments)
            .collect()
    }

    /// Differential check against the reference Python parser: `{text, calls,
    /// content}` cases produced by running `axk3_tool_parser.py` (`_scan` +
    /// `_resolve("all")` + `_render_arguments`) with the schemas in [`tools`];
    /// see `tests/data/ax_k3/gen_reference_cases.py`.
    /// Reference cases: the checked-in fixture, or a larger generated set at
    /// `AX_K3_REFERENCE_CASES`.
    pub(crate) fn reference_cases() -> Vec<Value> {
        let text = match std::env::var("AX_K3_REFERENCE_CASES") {
            Ok(path) => std::fs::read_to_string(path).expect("cases file"),
            Err(_) => include_str!("../../tests/data/ax_k3/reference_cases.json").to_string(),
        };
        serde_json::from_str(&text).expect("cases JSON")
    }

    #[test]
    fn fixture_schemas_match_the_test_tools() {
        let fixture: Vec<Value> =
            serde_json::from_str(include_str!("../../tests/data/ax_k3/tools.json"))
                .expect("tools JSON");
        let ours: Vec<Value> = tools()
            .into_iter()
            .map(|tool| json!({"name": tool.name, "parameters": tool.parameters}))
            .collect();
        assert_eq!(fixture, ours);
    }

    #[test]
    fn matches_reference_cases() {
        let cases = reference_cases();
        let mut failures = Vec::new();
        for case in &cases {
            let text = case["text"].as_str().expect("text");
            let (calls, content) = resolve(text, &tools());
            let got: Vec<Value> = calls
                .iter()
                .map(|call| json!({"name": call.name, "arguments": call.arguments}))
                .collect();
            if got != *case["calls"].as_array().expect("calls")
                || content != case["content"].as_str().expect("content")
            {
                failures.push(format!(
                    "{text:?}\n  want {} / {:?}\n  got  {} / {content:?}",
                    case["calls"],
                    case["content"],
                    Value::Array(got)
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} reference cases differ:\n{}",
            failures.len(),
            cases.len(),
            failures[..failures.len().min(8)].join("\n")
        );
    }

    #[test]
    fn no_call_is_content() {
        assert_eq!(run("안녕하세요"), (vec![], "안녕하세요".into()));
    }

    #[test]
    fn values_follow_schema_and_keep_their_spelling() {
        let text = call(
            "typed",
            &[
                ("s", "007"),
                ("i", "42"),
                ("f", "1.50"),
                ("b", "false"),
                ("arr", r#"["a", "b\"c"]"#),
                ("obj", r#"{"k": [1, null]}"#),
                ("maybe", "null"),
                ("untyped", "123"),
            ],
        );
        assert_eq!(
            raw_arguments(&text),
            vec![
                r#"{"s": "007", "i": 42, "f": 1.50, "b": false, "arr": ["a", "b\"c"], "obj": {"k": [1, null]}, "maybe": null, "untyped": 123}"#
            ]
        );
    }

    #[test]
    fn padding_is_kept_and_coercion_is_reserialized() {
        let text = call("typed", &[("i", " 7 "), ("b", "True")]);
        assert_eq!(raw_arguments(&text), vec![r#"{"i":  7 , "b": true}"#]);
        assert_eq!(run(&text).0[0].1, json!({"i": 7, "b": true}));
    }

    #[test]
    fn strings_unions_and_undescribed_keys() {
        let text = call(
            "typed",
            &[
                ("maybe", "plain text"),
                ("unknown", "not json"),
                ("unknown2", "[1, 2]"),
                ("unknown3", "\"quoted\""),
            ],
        );
        assert_eq!(
            run(&text).0[0].1,
            json!({"maybe": "plain text", "unknown": "not json", "unknown2": [1, 2], "unknown3": "\"quoted\""})
        );
    }

    #[test]
    fn string_values_are_verbatim_including_markers() {
        let tricky =
            "a \"quote\" \\ back\tslash\n<tool_call>x</tool_call> <think>y</think> 한글\n\n";
        let text = call(
            "write_file",
            &[("path", " spaced.py "), ("content", tricky)],
        );
        assert_eq!(
            run(&text),
            (
                vec![(
                    "write_file".into(),
                    json!({"path": " spaced.py ", "content": tricky})
                )],
                String::new()
            )
        );
    }

    #[test]
    fn every_call_is_executed_and_text_between_is_content() {
        let first = call("write_file", &[("path", "a.py"), ("content", "x")]);
        let ping = call("ping", &[]);
        let text = format!("먼저 {first} 잠깐, 다시. {ping} 끝.");
        assert_eq!(
            run(&text),
            (
                vec![
                    ("write_file".into(), json!({"path": "a.py", "content": "x"})),
                    ("ping".into(), json!({}))
                ],
                "먼저  잠깐, 다시.  끝.".into()
            )
        );
    }

    #[test]
    fn whitespace_between_calls_is_dropped() {
        let text = format!(
            "확인할게요.\n{}\n{}\n",
            call("ping", &[]),
            call("typed", &[("i", "1")])
        );
        let (calls, content) = run(&text);
        assert_eq!(calls.len(), 2);
        assert_eq!(content, "확인할게요.\n");
    }

    #[test]
    fn closed_calls_parse_the_same_once_the_text_is_whole() {
        // A value holding `</tool_call>` before its `<|/parameter|>` does not
        // close the call in a stream prefix.
        let prefix = "<tool_call>write_file<|parameter|>content\na</tool_call>";
        assert!(next_closed_call(prefix, 0, &tools()).is_none());
        let text = format!("{prefix}b<|/parameter|></tool_call>");
        let closed = next_closed_call(&text, 0, &tools()).expect("closed");
        assert_eq!((closed.start, closed.end), (0, text.len()));
        assert_eq!(
            closed.call.map(|call| call.arguments),
            Some(r#"{"content": "a</tool_call>b"}"#.into())
        );
        // An unclosed `<think>` hides what follows.
        assert!(next_closed_call(&format!("<think>{}", call("ping", &[])), 0, &tools()).is_none());
    }

    #[test]
    fn unclosed_call_text_is_content() {
        let open = "<tool_call>write_file<|parameter|>path\na.py<|/parameter|>";
        assert_eq!(run(open), (vec![], open.into()));
        let text = format!("{} 그리고{open}", call("ping", &[]));
        assert_eq!(
            run(&text),
            (vec![("ping".into(), json!({}))], format!(" 그리고{open}"))
        );
    }

    #[test]
    fn missing_parameter_end_runs_to_the_call_end() {
        let text = "<tool_call>write_file<|parameter|>path\na.py</tool_call>";
        assert_eq!(
            run(text).0,
            vec![("write_file".into(), json!({"path": "a.py"}))]
        );
    }

    #[test]
    fn key_without_newline_is_repaired_or_dropped() {
        let text = "<tool_call>typed<|parameter|>top_k30<|/parameter|><|parameter|>s\nok<|/parameter|></tool_call>";
        assert_eq!(run(text).0[0].1, json!({"top_k": 30, "s": "ok"}));
        let text = "<tool_call>typed<|parameter|>top_k<|/parameter|><|parameter|>mystery<|/parameter|></tool_call>";
        assert_eq!(run(text).0[0].1, json!({}));
        let text = "<tool_call>typed<|parameter|>top_k 5</tool_call>";
        assert_eq!(run(text).0[0].1, json!({"top_k": 5}));
    }

    #[test]
    fn calls_inside_a_think_block_are_not_parsed() {
        let text = format!("<think>{}</think>after", call("ping", &[]));
        assert_eq!(run(&text), (vec![], text.clone()));
    }

    #[test]
    fn first_duplicate_wins_and_empty_name_is_dropped() {
        let text = call("write_file", &[("path", "first"), ("path", "second")]);
        assert_eq!(raw_arguments(&text), vec![r#"{"path": "first"}"#]);
        assert_eq!(
            run("<tool_call></tool_call>"),
            (vec![], "<tool_call></tool_call>".into())
        );
    }

    #[test]
    fn legacy_projection_streams_to_the_same_result() {
        let text = format!("먼저 {} 끝", call("typed", &[("f", "1.50")]));
        let mut parser = AxK3ToolStreamParser::new(&tools());
        let mut normal = String::new();
        let mut calls = Vec::new();
        for ch in text.chars() {
            let result = parser.push(&ch.to_string()).expect("push");
            normal.push_str(&result.normal_text);
            calls.extend(result.calls);
        }
        let result = parser.finish().expect("finish");
        normal.push_str(&result.normal_text);
        calls.extend(result.calls);
        assert_eq!(normal, "먼저  끝");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"f": 1.50}"#);
    }
}
