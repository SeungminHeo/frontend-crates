// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai completion types and defines
// inference-serving extensions.

use std::collections::HashMap;
use std::pin::Pin;

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::OpenAIError;

use super::{ChatCompletionStreamOptions, CompletionUsage, Logprobs, Prompt, Stop};

/// OpenAI completion finish reasons with the engine repetition extension.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::completion::CompletionFinishReason))]
pub enum CompletionFinishReason {
    Stop,
    Length,
    ContentFilter,
    Repetition,
}

impl From<async_openai::types::chat::CompletionFinishReason> for CompletionFinishReason {
    fn from(reason: async_openai::types::chat::CompletionFinishReason) -> Self {
        use async_openai::types::chat::CompletionFinishReason as Upstream;
        match reason {
            Upstream::Stop => Self::Stop,
            Upstream::Length => Self::Length,
            Upstream::ContentFilter => Self::ContentFilter,
        }
    }
}

/// Completion choice retaining backend-specific finish reasons.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::completion::Choice))]
pub struct Choice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "protocol-schema", schema(value_type = Option<crate::schema::Logprobs>))]
    pub logprobs: Option<Logprobs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<CompletionFinishReason>,
}

/// OpenAI-compatible completion response with Dynamo's extended choices.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize, Default)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::completion::CreateCompletionResponse))]
pub struct CreateCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u32,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[cfg_attr(feature = "protocol-schema", schema(value_type = Option<crate::schema::CompletionUsage>))]
    pub usage: Option<CompletionUsage>,
}

/// Custom deserializer for the echo parameter that only accepts booleans.
/// Rejects integers and strings with clear error messages.
fn deserialize_echo_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StrictBoolVisitor;

    impl<'de> serde::de::Visitor<'de> for StrictBoolVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("echo parameter to be a boolean (true or false) or null")
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(BoolOnlyVisitor)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }
    }

    struct BoolOnlyVisitor;

    impl<'de> serde::de::Visitor<'de> for BoolOnlyVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("echo parameter to be a boolean (true or false) or null")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(value))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Err(E::invalid_type(
                serde::de::Unexpected::Str(value),
                &"echo parameter to be a boolean (true or false) or null",
            ))
        }
    }

    deserializer.deserialize_option(StrictBoolVisitor)
}

/// Completion request with inference-serving extensions.
///
/// Extends upstream `CreateCompletionRequest` with:
/// - `prompt_embeds`: base64-encoded PyTorch tensor for pre-computed embeddings
/// - `echo`: strict bool validation (rejects integers/strings)
/// - `stream_options`: uses our extended `ChatCompletionStreamOptions` (with `continuous_usage_stats`)
#[derive(Clone, Serialize, Deserialize, Default, Debug, Builder, PartialEq)]
#[builder(name = "CreateCompletionRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::completion::CreateCompletionRequest))]
pub struct CreateCompletionRequest {
    pub model: String,
    #[cfg_attr(feature = "protocol-schema", schema(value_type = crate::schema::Prompt))]
    pub prompt: Prompt,
    /// Base64-encoded PyTorch tensor containing pre-computed embeddings.
    /// At least one of prompt or prompt_embeds is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_embeds: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u8>,
    /// Echo back the prompt in addition to the completion.
    /// Strict bool validation -- rejects integers and strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_echo_bool")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Parsed server side events stream until an \[DONE\] is received from server.
pub type CompletionResponseStream =
    Pin<Box<dyn Stream<Item = Result<CreateCompletionResponse, OpenAIError>> + Send>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_round_trips_chat_and_text_responses() {
        let chat =
            serde_json::json!({"index":0,"delta":{"content":"hello"},"finish_reason":"repetition"});
        let choice: crate::types::ChatChoiceStream = serde_json::from_value(chat.clone()).unwrap();
        assert_eq!(
            choice.finish_reason,
            Some(crate::types::FinishReason::Repetition)
        );
        assert_eq!(
            serde_json::to_value(choice).unwrap()["finish_reason"],
            "repetition"
        );
        let text = serde_json::json!({"id":"test","object":"text_completion","created":0,"model":"test",
            "choices":[{"index":0,"text":"hello","finish_reason":"repetition"}],"usage":null,"system_fingerprint":null});
        let response: CreateCompletionResponse = serde_json::from_value(text.clone()).unwrap();
        assert_eq!(
            response.choices[0].finish_reason,
            Some(CompletionFinishReason::Repetition)
        );
        assert_eq!(serde_json::to_value(response).unwrap(), text);
    }

    #[test]
    fn echo_rejects_integer() {
        let json = r#"{"model": "test_model", "prompt": "test", "echo": 1}"#;
        let result: Result<CreateCompletionRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid type"));
        assert!(err_msg.contains("integer"));
        assert!(err_msg.contains("echo parameter"));
    }

    #[test]
    fn echo_rejects_string() {
        let json = r#"{"model": "test_model", "prompt": "test", "echo": "null"}"#;
        let result: Result<CreateCompletionRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid type"));
        assert!(err_msg.contains("string"));
        assert!(err_msg.contains("echo parameter"));
    }

    #[test]
    fn completion_choice_serializes_openai_shape() {
        use crate::types::{Choice, CompletionFinishReason};

        let choice = Choice {
            text: "hello".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: Some(CompletionFinishReason::Stop),
        };

        let value = serde_json::to_value(choice).expect("serialize choice");

        assert_eq!(value["finish_reason"], "stop");
        assert_eq!(value["text"], "hello");
    }

    #[test]
    fn stop_accepts_token_id_array() {
        let json = r#"{"model": "test_model", "prompt": [1, 2, 3], "stop": [32, 34]}"#;
        let request: CreateCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.stop, Some(Stop::TokenIdArray(vec![32, 34])));
    }

    #[test]
    fn stop_accepts_string_and_string_array() {
        let one_stop = r#"{"model": "test_model", "prompt": "hello", "stop": " The"}"#;
        let request: CreateCompletionRequest = serde_json::from_str(one_stop).unwrap();

        assert_eq!(request.stop, Some(Stop::String(" The".to_string())));

        let many_stops = r#"{"model": "test_model", "prompt": "hello", "stop": ["A", "B"]}"#;
        let request: CreateCompletionRequest = serde_json::from_str(many_stops).unwrap();

        assert_eq!(
            request.stop,
            Some(Stop::StringArray(vec!["A".to_string(), "B".to_string()]))
        );
    }

    #[test]
    fn stop_token_id_display_string_remains_string_stop() {
        let json = r#"{"model": "test_model", "prompt": [1, 2, 3], "stop": "token_id:576"}"#;
        let request: CreateCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.stop, Some(Stop::String("token_id:576".to_string())));

        let json = r#"{"model": "test_model", "prompt": [1, 2, 3], "stop": ["token_id:576"]}"#;
        let request: CreateCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(
            request.stop,
            Some(Stop::StringArray(vec!["token_id:576".to_string()]))
        );
    }

    #[test]
    fn builder_accepts_upstream_stop_configuration() {
        let upstream_stop = async_openai::types::chat::StopConfiguration::String("END".to_string());

        let request = CreateCompletionRequestArgs::default()
            .model("test_model")
            .prompt(Prompt::String("hello".to_string()))
            .stop(upstream_stop)
            .build()
            .unwrap();

        assert_eq!(request.stop, Some(Stop::String("END".to_string())));
    }

    #[test]
    fn stop_rejects_single_token_id() {
        let json = r#"{"model": "test_model", "prompt": [1, 2, 3], "stop": 576}"#;
        let result: Result<CreateCompletionRequest, _> = serde_json::from_str(json);

        assert!(result.is_err());
    }
}
