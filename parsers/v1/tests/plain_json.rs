// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::{
    config::{JsonParserConfig, ParserConfig},
    detect_and_parse_tool_call,
    jail::{Annotated, JailedStream},
    json::{JsonParserType, try_tool_call_parse_basic_json},
    parsers::{detect_and_parse_tool_call_with_recovery, get_tool_parser_map},
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, CreateChatCompletionStreamResponse, FinishReason,
};
use futures::StreamExt;

const ORDINARY_JSON: &[&str] = &[
    "[5, 0, 8]",
    r#"["a","b"]"#,
    "[]",
    r#"[{"a":1},{"b":2}]"#,
    "결과: [5, 0, 8]",
    " \n[5, 0, 8]\t ",
    "[[1], [], [null, true]]",
    r#"{"a":5}"#,
    "[속보] 뉴스",
    "[네이버] 인증번호 [123456]",
];

const HERMES_UNMARKED: &[&str] = &[
    r#"{"name":"Alice"}"#,
    r#"[{"name":"Alice"}]"#,
    r#"{"name":"lookup","arguments":{}}"#,
    r#"[{"name":"lookup","arguments":{}}]"#,
    r#"결과: {"name":"Alice","parameters":{}}"#,
];

fn basic_json_parsers() -> Vec<&'static str> {
    let mut names: Vec<_> = get_tool_parser_map()
        .iter()
        .filter_map(|(&name, config)| {
            matches!(
                &config.parser_config,
                ParserConfig::Json(config) if matches!(config.parser_type, JsonParserType::Basic)
            )
            .then_some(name)
        })
        .collect();
    names.sort_unstable();
    names
}

#[tokio::test]
async fn ordinary_json_survives_batch_and_finalize() {
    let mut failures = Vec::new();
    for parser in basic_json_parsers() {
        for &text in ORDINARY_JSON
            .iter()
            .chain(HERMES_UNMARKED.iter().filter(|_| parser == "hermes"))
        {
            for recover in [false, true] {
                let (calls, content) = if recover {
                    detect_and_parse_tool_call_with_recovery(text, Some(parser), None).await
                } else {
                    detect_and_parse_tool_call(text, Some(parser), None).await
                }
                .unwrap();
                if !calls.is_empty() || content.as_deref() != Some(text) {
                    failures.push(format!(
                        "{parser} recover={recover} input={text:?}: calls={}, content={content:?}",
                        calls.len()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn chunk(
    text: &str,
    finish: Option<FinishReason>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(serde_json::from_value(serde_json::json!({
            "id": "plain-json", "object": "chat.completion.chunk", "created": 0, "model": "test",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]
        })).unwrap()),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

#[tokio::test]
async fn ordinary_json_survives_stream_chunk_boundaries() {
    for parser in basic_json_parsers() {
        for &text in ORDINARY_JSON
            .iter()
            .chain(HERMES_UNMARKED.iter().filter(|_| parser == "hermes"))
        {
            for split in (0..=text.len()).filter(|&at| text.is_char_boundary(at)) {
                let chunks = [
                    chunk(&text[..split], None),
                    chunk(&text[split..], Some(FinishReason::Stop)),
                ];
                let results: Vec<_> = JailedStream::builder()
                    .tool_call_parser(parser)
                    .build()
                    .apply_with_finish_reason(futures::stream::iter(chunks))
                    .collect()
                    .await;
                let mut content = String::new();
                let mut finish_reasons = Vec::new();
                for result in results {
                    assert!(result.error.is_none());
                    if let Some(data) = result.data {
                        for choice in data.choices {
                            assert!(choice.delta.tool_calls.as_ref().is_none_or(Vec::is_empty));
                            if let Some(ChatCompletionMessageContent::Text(text)) =
                                choice.delta.content
                            {
                                content.push_str(&text);
                            }
                            finish_reasons.extend(choice.finish_reason);
                        }
                    }
                }
                assert_eq!(content, text, "{parser} split={split}");
                assert_eq!(
                    finish_reasons,
                    [FinishReason::Stop],
                    "{parser} split={split}"
                );
            }
        }
    }
}

#[tokio::test]
async fn real_calls_and_explicit_empty_wrappers_keep_their_meaning() {
    let guided = JsonParserConfig {
        bare_json_mode: true,
        ..Default::default()
    };
    let (calls, content) = try_tool_call_parse_basic_json("[]", &guided, None).unwrap();
    assert!(calls.is_empty());
    assert_eq!(content.as_deref(), Some(""));

    for parser in basic_json_parsers()
        .into_iter()
        .filter(|name| *name != "hermes")
    {
        let payload = r#"[{"name":"lookup","arguments":{"b": 2, "a":1}}]"#;
        let (calls, content) =
            detect_and_parse_tool_call_with_recovery(payload, Some(parser), None)
                .await
                .unwrap();
        assert_eq!(calls.len(), 1, "{parser}");
        assert_eq!(calls[0].function.arguments, r#"{"b": 2, "a":1}"#);
        assert_eq!(content.as_deref(), Some(""));
    }
    for parser in ["hermes", "qwen25"] {
        for payload in ["[]", r#"[{"a":1}]"#] {
            let text = format!("prefix <tool_call>{payload}</tool_call>");
            let (calls, content) =
                detect_and_parse_tool_call_with_recovery(&text, Some(parser), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty());
            assert_eq!(content.as_deref(), Some("prefix"));
        }
    }
}

#[tokio::test]
async fn hermes_requires_markers_but_guided_json_still_works() {
    for text in [
        r#"{"name":"lookup","arguments":{}}"#,
        r#"[{"name":"lookup","arguments":{}}]"#,
        r#"{"name":"lookup"}"#,
    ] {
        for recover in [false, true] {
            let (calls, content) = if recover {
                detect_and_parse_tool_call_with_recovery(text, Some("hermes"), None).await
            } else {
                detect_and_parse_tool_call(text, Some("hermes"), None).await
            }
            .unwrap();
            assert!(calls.is_empty());
            assert_eq!(content.as_deref(), Some(text));
        }
    }
    let guided = JsonParserConfig {
        require_start_token: true,
        bare_json_mode: true,
        ..Default::default()
    };
    let (calls, _) =
        try_tool_call_parse_basic_json(r#"{"name":"lookup","arguments":{}}"#, &guided, None)
            .unwrap();
    assert_eq!(calls.len(), 1);
}
