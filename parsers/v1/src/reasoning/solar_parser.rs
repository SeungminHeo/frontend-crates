// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Solar Open2 reasoning: a port of UpstageAI/vllm `v0.22.0-solar-open2`
//! `vllm/reasoning/solar_open2_reasoning_parser.py`.
//!
//! * The chat template prefills `<|think:start|>`, so generation starts inside
//!   the thought. Only `<|think:end|>` ends it; everything after it is content.
//!   Text the model writes while thinking (`<tool_call>`, `<|tool_call:start|>`,
//!   `<|think:start|>`) stays reasoning.
//! * Complete tool-call blocks written inside the thought (matching the Solar
//!   tool grammar) move to content, after the real content, because the tool
//!   parser keeps only the text before the first call as content. The
//!   reference does this for non-streaming output only; Dynamo builds
//!   non-streaming responses from the stream, so the streaming path holds the
//!   thought from the first `<|tool_call:start|>` and appends the blocks at the
//!   end of the stream.
//!
//! Deviations from the reference, both for the Dynamo stream:
//! * Output without `<|think:end|>` (cut off while thinking) is reasoning in
//!   both paths, as the reference streams it; its non-streaming path returns
//!   it as content. Complete call blocks in it still move to content.
//! * When blocks move, the reference strips the remaining reasoning at both
//!   ends. The stream strips the end; leading whitespace already streamed
//!   before non-whitespace reasoning stays.
//!
//! A leading `<|think:start|>` (template not prefilled) is dropped.

use std::sync::LazyLock;

use regex::Regex;

use super::{ParserResult, ReasoningParser};
use crate::tool_calling::solar::{ARG_END, ARG_START, CALL_END, CALL_START};

pub const THINK_START: &str = "<|think:start|>";
pub const THINK_END: &str = "<|think:end|>";

/// The reference `embedded_tool_call_regex`: blocks the Solar tool parser can
/// parse (`SolarOpen2ToolParser.tool_call_pattern` without groups).
static EMBEDDED_CALL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?s){}.+?\n(?:{}.*?{}\n?)*{}",
        regex::escape(CALL_START),
        regex::escape(ARG_START),
        regex::escape(ARG_END),
        regex::escape(CALL_END),
    ))
    .expect("valid Solar embedded call regex")
});

/// Remove complete call blocks from `reasoning`: the rest, and the blocks.
fn take_embedded_calls(reasoning: &str) -> (String, Vec<String>) {
    if !reasoning.contains(CALL_START) {
        return (reasoning.to_string(), Vec::new());
    }
    let blocks: Vec<String> = EMBEDDED_CALL
        .find_iter(reasoning)
        .map(|block| block.as_str().to_string())
        .collect();
    if blocks.is_empty() {
        return (reasoning.to_string(), blocks);
    }
    (
        EMBEDDED_CALL.replace_all(reasoning, "").into_owned(),
        blocks,
    )
}

/// The reference content join: real content first, then the moved blocks.
fn append_blocks(content: &mut String, content_present: bool, blocks: &[String]) {
    if blocks.is_empty() {
        return;
    }
    if content_present {
        content.push('\n');
    }
    content.push_str(&blocks.join("\n"));
}

/// Bytes at the end of `text` that could still become one of `sentinels`
/// (anchored on the last `<`; every Solar sentinel starts with `<|`).
fn holdback(text: &str, sentinels: &[&str]) -> usize {
    let Some(last_lt) = text.rfind('<') else {
        return 0;
    };
    let tail = &text[last_lt..];
    if sentinels
        .iter()
        .any(|sentinel| tail.len() < sentinel.len() && sentinel.starts_with(tail))
    {
        tail.len()
    } else {
        0
    }
}

#[derive(Debug, Clone)]
pub struct SolarOpen2ReasoningParser {
    /// The template opened the thought (default); `false` when it was closed.
    in_reasoning: bool,
    /// Nothing generated yet: a leading `<|think:start|>` is dropped.
    at_start: bool,
    /// Unsettled bytes: a possible sentinel prefix.
    buffer: String,
    /// Trailing whitespace of streamed reasoning, sent with the next text.
    pending_ws: String,
    /// Non-whitespace reasoning has been streamed.
    reasoning_sent: bool,
    /// The thought from its first `<|tool_call:start|>`, held until
    /// `<|think:end|>` or the end of the stream decides which blocks move.
    held: Option<String>,
    /// Moved call blocks, appended to content at the end of the stream.
    moved: Vec<String>,
    /// Non-empty content has been streamed.
    content_sent: bool,
}

impl Default for SolarOpen2ReasoningParser {
    fn default() -> Self {
        Self {
            in_reasoning: true,
            at_start: true,
            buffer: String::new(),
            pending_ws: String::new(),
            reasoning_sent: false,
            held: None,
            moved: Vec::new(),
            content_sent: false,
        }
    }
}

impl SolarOpen2ReasoningParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stream thought text, holding it from the first `<|tool_call:start|>`.
    fn reasoning(&mut self, text: &str, out: &mut ParserResult) {
        if let Some(held) = &mut self.held {
            held.push_str(text);
            return;
        }
        match text.find(CALL_START) {
            Some(at) => {
                self.stream_reasoning(&text[..at], out);
                self.held = Some(text[at..].to_string());
            }
            None => self.stream_reasoning(text, out),
        }
    }

    fn stream_reasoning(&mut self, text: &str, out: &mut ParserResult) {
        let body = text.trim_end().len();
        if body > 0 {
            out.reasoning_text
                .push_str(&std::mem::take(&mut self.pending_ws));
            out.reasoning_text.push_str(&text[..body]);
            self.reasoning_sent = true;
        }
        self.pending_ws.push_str(&text[body..]);
    }

    /// The thought is over (`<|think:end|>` or end of stream): settle the held
    /// text and the trailing whitespace.
    fn close_reasoning(&mut self, out: &mut ParserResult) {
        let pending = std::mem::take(&mut self.pending_ws);
        let Some(held) = self.held.take() else {
            out.reasoning_text.push_str(&pending);
            return;
        };
        let (rest, blocks) = take_embedded_calls(&held);
        if blocks.is_empty() {
            out.reasoning_text.push_str(&pending);
            out.reasoning_text.push_str(&held);
            return;
        }
        let tail = format!("{pending}{rest}");
        let tail = if self.reasoning_sent {
            tail.trim_end()
        } else {
            tail.trim()
        };
        out.reasoning_text.push_str(tail);
        self.moved.extend(blocks);
    }

    fn content(&mut self, text: &str, out: &mut ParserResult) {
        if !text.is_empty() {
            out.normal_text.push_str(text);
            self.content_sent = true;
        }
    }
}

impl ReasoningParser for SolarOpen2ReasoningParser {
    /// The reference `extract_reasoning` (see the module notes for the
    /// cut-off thought).
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        if !self.in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }
        let text = text.strip_prefix(THINK_START).unwrap_or(text);
        let (reasoning, content) = match text.find(THINK_END) {
            Some(end) => (&text[..end], &text[end + THINK_END.len()..]),
            None => (text, ""),
        };
        let (rest, blocks) = take_embedded_calls(reasoning);
        if blocks.is_empty() {
            return ParserResult {
                normal_text: content.to_string(),
                reasoning_text: reasoning.to_string(),
            };
        }
        let mut normal_text = content.to_string();
        append_blocks(&mut normal_text, !content.is_empty(), &blocks);
        ParserResult {
            normal_text,
            reasoning_text: rest.trim().to_string(),
        }
    }

    /// The reference `extract_reasoning_streaming`, plus moving complete call
    /// blocks out of the thought.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        let mut out = ParserResult::default();
        if !self.in_reasoning {
            self.content(text, &mut out);
            return out;
        }
        self.buffer.push_str(text);
        if self.at_start {
            if self.buffer.len() < THINK_START.len() && THINK_START.starts_with(&self.buffer) {
                return out; // a generated opener may still be arriving
            }
            if self.buffer.starts_with(THINK_START) {
                self.buffer.drain(..THINK_START.len());
            }
            self.at_start = false;
        }
        if let Some(end) = self.buffer.find(THINK_END) {
            let after = self.buffer.split_off(end + THINK_END.len());
            self.buffer.truncate(end);
            let before = std::mem::take(&mut self.buffer);
            self.reasoning(&before, &mut out);
            self.close_reasoning(&mut out);
            self.in_reasoning = false;
            self.content(&after, &mut out);
            return out;
        }
        let keep = holdback(&self.buffer, &[THINK_END, CALL_START]);
        let settled: String = self.buffer.drain(..self.buffer.len() - keep).collect();
        self.reasoning(&settled, &mut out);
        out
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        let mut out = ParserResult::default();
        if self.in_reasoning {
            let rest = std::mem::take(&mut self.buffer);
            self.at_start = false;
            self.reasoning(&rest, &mut out);
            self.close_reasoning(&mut out);
            self.in_reasoning = false;
        } else {
            let rest = std::mem::take(&mut self.buffer);
            self.content(&rest, &mut out);
        }
        let moved = std::mem::take(&mut self.moved);
        append_blocks(&mut out.normal_text, self.content_sent, &moved);
        out
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.in_reasoning = in_reasoning;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: &str = "<|tool_call:start|>get_weather\n<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n<|tool_call:end|>";
    const BLOCK_2: &str = "<|tool_call:start|>get_time\n<|tool_arg:start|>tz<|tool_arg:value|>KST<|tool_arg:end|>\n<|tool_call:end|>";

    fn batch(text: &str) -> (String, String) {
        let result = SolarOpen2ReasoningParser::new().detect_and_parse_reasoning(text, &[]);
        (result.reasoning_text, result.normal_text)
    }

    /// Differential check against the reference: `{text, reasoning, content}`
    /// from its non-streaming `extract_reasoning`, generated by
    /// `tests/data/solar_open2/gen_reasoning_cases.py`. Every case ends its
    /// thought, so the cut-off deviation does not apply; outputs that open
    /// with `<|think:start|>` (dropped here) are skipped. The stream must give
    /// the batch result, except leading whitespace already streamed before
    /// blocks moved.
    #[test]
    fn matches_reference_cases() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../tests/data/solar_open2/reasoning_cases.json"
        ))
        .expect("cases JSON");
        let mut failures = Vec::new();
        let mut checked = 0;
        for case in &cases {
            let text = case["text"].as_str().expect("text");
            if text.starts_with(THINK_START) {
                continue;
            }
            checked += 1;
            let want = pair(
                case["reasoning"].as_str().expect("reasoning"),
                case["content"].as_str().expect("content"),
            );
            let got = batch(text);
            if got != want {
                failures.push(format!("batch {text:?}\n  want {want:?}\n  got  {got:?}"));
                continue;
            }
            let chars: Vec<String> = text.chars().map(String::from).collect();
            let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
            let streamed = stream(&chars);
            let leading_ws_only = streamed.1 == want.1
                && streamed.0.trim_start() == want.0
                && text.starts_with(char::is_whitespace);
            if streamed != want && !leading_ws_only {
                failures.push(format!(
                    "stream {text:?}\n  want {want:?}\n  got  {streamed:?}"
                ));
            }
        }
        assert!(checked > 1000, "only {checked} cases checked");
        assert!(
            failures.is_empty(),
            "{} of {checked} reference cases differ:\n{}",
            failures.len(),
            failures[..failures.len().min(6)].join("\n")
        );
    }

    fn stream(chunks: &[&str]) -> (String, String) {
        let mut parser = SolarOpen2ReasoningParser::new();
        let (mut reasoning, mut content) = (String::new(), String::new());
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            content.push_str(&result.normal_text);
        }
        let result = parser.finish_reasoning_stream();
        reasoning.push_str(&result.reasoning_text);
        content.push_str(&result.normal_text);
        (reasoning, content)
    }

    /// Batch, then streaming per character, in chunks, and at every split.
    fn parse(text: &str) -> (String, String) {
        let whole = batch(text);
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(stream(&chars), whole, "per character {text:?}");
        for size in [2, 3, 5, 7, 100] {
            let bytes: Vec<char> = text.chars().collect();
            let chunks: Vec<String> = bytes.chunks(size).map(|c| c.iter().collect()).collect();
            let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
            assert_eq!(stream(&chunks), whole, "chunks of {size} {text:?}");
        }
        for at in (0..=text.len()).filter(|at| text.is_char_boundary(*at)) {
            assert_eq!(
                stream(&[&text[..at], &text[at..]]),
                whole,
                "split {at} {text:?}"
            );
        }
        whole
    }

    fn pair(reasoning: &str, content: &str) -> (String, String) {
        (reasoning.to_string(), content.to_string())
    }

    // UpstageAI/vllm tests/reasoning/test_solar_open2_reasoning_parser.py

    #[test]
    fn reference_split_cases() {
        let cases = [
            (
                "I need to analyze<|think:end|>The answer is 42",
                pair("I need to analyze", "The answer is 42"),
            ),
            ("My reasoning<|think:end|>", pair("My reasoning", "")),
            ("<|think:end|>Direct answer", pair("", "Direct answer")),
            (
                "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer",
                pair("Step 1: analyze\nStep 2: solve\n", "Final answer"),
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(parse(input), expected, "{input:?}");
        }
    }

    #[test]
    fn reference_sentinel_split_across_chunks() {
        assert_eq!(
            stream(&["reasoning body ", "<|think:", "end|>after"]),
            pair("reasoning body ", "after")
        );
    }

    #[test]
    fn reference_embedded_calls_move_after_the_content() {
        let cases = [
            (
                format!("I should check the weather.\n{BLOCK}\n<|think:end|>"),
                pair("I should check the weather.", BLOCK),
            ),
            (
                format!("thinking\n{BLOCK}\n<|think:end|>The answer is 42"),
                pair("thinking", &format!("The answer is 42\n{BLOCK}")),
            ),
            (
                format!("plan calls\n{BLOCK}\n<|think:end|>{BLOCK_2}"),
                pair("plan calls", &format!("{BLOCK_2}\n{BLOCK}")),
            ),
            (
                format!("first\n{BLOCK}\nthen\n{BLOCK_2}\n<|think:end|>"),
                pair("first\n\nthen", &format!("{BLOCK}\n{BLOCK_2}")),
            ),
            (
                "no args needed\n<|tool_call:start|>refresh\n<|tool_call:end|>\n<|think:end|>done"
                    .to_string(),
                pair(
                    "no args needed",
                    "done\n<|tool_call:start|>refresh\n<|tool_call:end|>",
                ),
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(parse(&input), expected, "{input:?}");
        }
    }

    #[test]
    fn reference_mentions_and_abandoned_calls_stay_reasoning() {
        let thinking = "Should I emit <|tool_call:start|> here? No.\n";
        assert_eq!(
            parse(&format!("{thinking}<|think:end|>Answer")),
            pair(thinking, "Answer")
        );
        let thinking =
            "let me call\n<|tool_call:start|>foo\n<|tool_arg:start|>a<|tool_arg:value|>1\n";
        assert_eq!(
            parse(&format!("{thinking}<|think:end|>after")),
            pair(thinking, "after")
        );
    }

    /// The reported bug: XML-style text while thinking ended the thought.
    #[test]
    fn literal_xml_tool_call_text_stays_reasoning() {
        let thinking = "I could answer with <tool_call>{\"name\": \"x\"}</tool_call>, but Solar uses its own markers. Thinking more.";
        assert_eq!(
            parse(&format!("{thinking}<|think:end|>Final answer.")),
            pair(thinking, "Final answer.")
        );
    }

    /// Deviation: a thought cut off before `<|think:end|>` is reasoning (the
    /// reference streams it so; its batch path returns content).
    #[test]
    fn cut_off_thought_is_reasoning_and_complete_calls_still_move() {
        assert_eq!(parse("Still thinking..."), pair("Still thinking...", ""));
        assert_eq!(
            parse(&format!("checking weather\n{BLOCK}")),
            pair("checking weather", BLOCK)
        );
    }

    #[test]
    fn markers_after_the_thought_are_content() {
        let content = "answer <|think:start|>x<|think:end|> <tool_call>";
        assert_eq!(
            parse(&format!("r<|think:end|>{content}")),
            pair("r", content)
        );
    }

    #[test]
    fn leading_generated_opener_is_dropped() {
        assert_eq!(parse("<|think:start|>r<|think:end|>a"), pair("r", "a"));
    }

    #[test]
    fn closed_thought_prompt_is_all_content() {
        let mut parser = SolarOpen2ReasoningParser::new();
        parser.set_in_reasoning(false);
        let result = parser.parse_reasoning_streaming_incremental("a<|think:end|>b", &[]);
        assert_eq!(result.normal_text, "a<|think:end|>b");
        assert!(result.reasoning_text.is_empty());
    }

    #[test]
    fn registry_builds_the_solar_parser() {
        let mut parser =
            crate::reasoning::ReasoningParserType::get_reasoning_parser_from_name("solar_open2");
        let result = parser.detect_and_parse_reasoning("a <tool_call> b<|think:end|>c", &[]);
        assert_eq!(result.reasoning_text, "a <tool_call> b");
        assert_eq!(result.normal_text, "c");
    }
}
