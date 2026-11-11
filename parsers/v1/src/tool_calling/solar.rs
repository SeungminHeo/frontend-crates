// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Solar Open2's raw argument grammar. The shared jail owns streaming buffers.
//! Values follow the schema coercions in UpstageAI/vllm's Solar Open2 parser.
use super::{CalledFunction, ToolCallResponse, ToolCallType, ToolDefinition};
use serde_json::{Map, Value};

pub const CALL_START: &str = "<|tool_call:start|>";
pub const CALL_END: &str = "<|tool_call:end|>";
pub const ARG_START: &str = "<|tool_arg:start|>";
pub const ARG_VALUE: &str = "<|tool_arg:value|>";
pub const ARG_END: &str = "<|tool_arg:end|>";

/// No EOF recovery: a call is executable only after its explicit closing fence.
pub(super) fn complete_end(text: &str) -> Option<usize> {
    let start = text.rfind(CALL_START)?;
    text[start..]
        .find(CALL_END)
        .map(|end| start + end + CALL_END.len())
}

pub fn parse(
    text: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let Some(first) = text.find(CALL_START) else {
        return Ok((Vec::new(), Some(text.to_string())));
    };
    let mut calls = Vec::new();
    let mut remaining = &text[first..];
    while let Some(start) = remaining.find(CALL_START) {
        remaining = &remaining[start + CALL_START.len()..];
        let Some(end) = remaining.find(CALL_END) else {
            break;
        };
        let body = &remaining[..end];
        remaining = &remaining[end + CALL_END.len()..];
        let Some((name, mut args)) = body.split_once('\n') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || name.contains(CALL_START) {
            continue;
        }
        let properties = tools
            .and_then(|tools| tools.iter().find(|tool| tool.name == name))
            .and_then(|tool| tool.parameters.as_ref())
            .and_then(|schema| schema.get("properties"));
        let mut arguments = Map::new();
        let mut valid = true;
        while !args.trim().is_empty() {
            let Some(rest) = args.trim_start().strip_prefix(ARG_START) else {
                valid = false;
                break;
            };
            let Some((key, rest)) = rest.split_once(ARG_VALUE) else {
                valid = false;
                break;
            };
            let Some((value, rest)) = rest.split_once(ARG_END) else {
                valid = false;
                break;
            };
            if key.is_empty() || key.contains(ARG_START) {
                valid = false;
                break;
            }
            let ty = properties
                .and_then(|props| props.get(key))
                .and_then(|spec| spec.get("type"));
            arguments.insert(key.to_string(), coerce(value, ty));
            args = rest;
        }
        if valid {
            calls.push(ToolCallResponse {
                id: format!("call-{}", uuid::Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: name.to_string(),
                    arguments: serde_json::to_string(&arguments)?,
                },
            });
        }
    }
    // The template forbids prose after the first call, including between calls.
    Ok((calls, (first > 0).then(|| text[..first].to_string())))
}

fn coerce(raw: &str, schema_type: Option<&Value>) -> Value {
    let value = raw.trim();
    if value.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let ty = schema_type
        .and_then(|ty| match ty {
            Value::Array(types) => types
                .iter()
                .filter_map(Value::as_str)
                .find(|ty| *ty != "null"),
            _ => ty.as_str(),
        })
        .unwrap_or("string");
    let converted = match ty.trim().to_ascii_lowercase().as_str() {
        "integer" | "int" => value
            .parse::<i64>()
            .map(Value::from)
            .ok()
            .or_else(|| value.parse::<u64>().map(Value::from).ok()),
        "number" | "float" | "double" => value
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .and_then(|n| {
                if n.fract() == 0.0 {
                    // Avoid saturating casts and retain large, finite JSON numbers.
                    serde_json::from_str(&format!("{n:.0}")).ok()
                } else {
                    serde_json::Number::from_f64(n).map(Value::Number)
                }
            }),
        "boolean" | "bool" => match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Some(Value::Bool(true)),
            "false" | "0" | "no" => Some(Value::Bool(false)),
            _ => None,
        },
        "array" | "list" | "object" | "dict" => serde_json::from_str(value).ok(),
        "null" | "none" => Some(Value::Null),
        _ => None,
    };
    converted.unwrap_or_else(|| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn solar_schema_coercion() {
        for (raw, ty, expected) in [
            ("42", json!("string"), json!("42")),
            (" 7 ", json!(["null", "integer"]), json!(7)),
            ("2.0", json!("number"), json!(2)),
            ("1.25", json!("number"), json!(1.25)),
            ("yes", json!("boolean"), json!(true)),
            ("NO", json!("bool"), json!(false)),
            ("[1,2]", json!("array"), json!([1, 2])),
            ("{\"a\":1}", json!("object"), json!({"a":1})),
            ("null", json!("string"), Value::Null),
            ("many", json!("integer"), json!("many")),
            ("NaN", json!("number"), json!("NaN")),
            ("Infinity", json!("number"), json!("Infinity")),
            ("  서울\n ", Value::Null, json!("  서울\n ")),
        ] {
            assert_eq!(coerce(raw, Some(&ty)), expected, "{raw} {ty}");
        }
    }

    #[test]
    fn solar_complete_parallel_and_malformed_calls() {
        let call = "<|tool_call:start|>search\n<|tool_arg:start|>limit<|tool_arg:value|>7<|tool_arg:end|>\n<|tool_call:end|>";
        let tools = [ToolDefinition {
            name: "search".into(),
            parameters: Some(json!({"properties":{"limit":{"type":"integer"}}})),
            strict: None,
        }];
        let (calls, text) = parse(
            &format!("Sure. {call}\n<|tool_call:start|>now\n<|tool_call:end|>discard"),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(text.as_deref(), Some("Sure. "));
        assert_eq!(calls.len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"limit":7})
        );
        assert_eq!(calls[1].function.arguments, "{}");
        assert!(
            parse(
                "<|tool_call:start|>f\n<|tool_arg:start|>x<|tool_arg:value|>cut",
                None
            )
            .unwrap()
            .0
            .is_empty()
        );
        assert!(
            parse("<|tool_call:start|>f\ngarbage<|tool_call:end|>", None)
                .unwrap()
                .0
                .is_empty()
        );
        assert_eq!(
            parse("plain 서울", None).unwrap().1.as_deref(),
            Some("plain 서울")
        );
    }

    #[test]
    fn solar_argument_reasoning_markers_remain_literal() {
        use crate::{ReasoningParser, ReasoningParserType};
        let call = "<|tool_call:start|>f\n<|tool_arg:start|>text<|tool_arg:value|>literal <|think:start|>x<|think:end|><|tool_arg:end|>\n<|tool_call:end|>";
        let raw = format!("plan <|think:end|>{call}");
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("solar_open2");
        let mut content = String::new();
        let mut reasoning = String::new();
        for character in raw.chars() {
            let result = parser.parse_reasoning_streaming_incremental(&character.to_string(), &[]);
            content.push_str(&result.normal_text);
            reasoning.push_str(&result.reasoning_text);
        }
        let result = parser.finish_reasoning_stream();
        content.push_str(&result.normal_text);
        reasoning.push_str(&result.reasoning_text);
        assert_eq!(reasoning, "plan ");
        assert_eq!(content, call);
        let calls = parse(&content, None).unwrap().0;
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"text":"literal <|think:start|>x<|think:end|>"})
        );
    }

    #[test]
    fn solar_reasoning_every_split_and_tool_escape() {
        use crate::{ReasoningParser, ReasoningParserType};
        for (raw, reasoning, content) in [
            ("생각<|think:end|>답", "생각", "답"),
            ("<|think:start|>생각<|think:end|>답", "생각", "답"),
            // A complete call written while thinking moves to content; the rest
            // of the thought is stripped as in the reference.
            (
                "plan <|tool_call:start|>f\n<|tool_call:end|>",
                "plan",
                "<|tool_call:start|>f\n<|tool_call:end|>",
            ),
        ] {
            for split in raw
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(raw.len()))
            {
                let mut parser = ReasoningParserType::get_reasoning_parser_from_name("solar_open2");
                let mut r = String::new();
                let mut c = String::new();
                for chunk in [&raw[..split], &raw[split..]] {
                    let out = parser.parse_reasoning_streaming_incremental(chunk, &[]);
                    r.push_str(&out.reasoning_text);
                    c.push_str(&out.normal_text);
                }
                let out = parser.finish_reasoning_stream();
                r.push_str(&out.reasoning_text);
                c.push_str(&out.normal_text);
                assert_eq!(
                    (r.as_str(), c.as_str()),
                    (reasoning, content),
                    "split {split}: {raw}"
                );
            }
        }
    }
}
