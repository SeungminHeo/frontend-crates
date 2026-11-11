// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified A.X-K3 parser: the reference `deepseek_r1` reasoning split followed
//! by the reference `axk3` tool parser, in one state machine.
//!
//! ```text
//! reasoning:  [<think>] … </think>        (the chat template opens <think>)
//! tool call:  <tool_call>NAME<|parameter|>KEY\nVALUE<|/parameter|></tool_call>
//! everything else: visible content
//! ```
//!
//! * The stream starts inside reasoning (the template opened `<think>`); only
//!   the first `</think>` ends it, so tool markup inside the thought stays
//!   reasoning text (DeepSeek-R1). `Response` starts in the answer instead.
//! * Content streams live up to the first `<tool_call>`. After it, under the
//!   reference `call_group_policy = "all"`, each call is sent whole (name and
//!   complete arguments) as soon as its `</tool_call>` arrives, with the
//!   non-whitespace text before it. Text after the last closed call waits for
//!   the next call or the end of stream, where the rest is settled by
//!   `tool_calling::ax_k3::resolve` as the reference non-streaming parser
//!   settles the whole answer (an unclosed call is content).
//! * A `<think>` block in the answer is content, and no call is looked for in it.
//!
//! Guided JSON (`tool_choice` required/named) is handled by the shared
//! [`GuidedRouted`] reader; it locates native markup with the A.X-K3 grammar.

use crate::tool_calling::ax_k3::{
    BLOCK_END, BLOCK_START, ResolvedCall, THINK_END, THINK_START, ax_k3_scanner, next_closed_call,
    resolve,
};
use crate::tool_calling::scan::{InvokeBoundaryFactory, ReasoningSpec, WrappedBlockScanner};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedGrammar, GuidedReasoning, GuidedRouted, NativeUnified, UnifiedParser,
    UnifiedParserOutput, UnifiedParserStartingState,
};
use anyhow::Result;

const REASONING: ReasoningSpec = ReasoningSpec {
    start: THINK_START,
    end: THINK_END,
    forced_start: true,
    preserve_special_tokens: true,
    start_label: None,
};

pub(crate) fn ax_k3_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(AxK3Native::new(tools)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Inside the prompt-opened thought; `at_start` accepts a generated opener.
    Reasoning { at_start: bool },
    /// Visible answer before any tool call.
    Content,
    /// A `<think>` block inside the answer: content, never a tool call.
    ContentThink,
    /// From the first `<tool_call>`: calls are sent as they close.
    Calls,
}

/// Progress through the text from the first `<tool_call>` (`buffer[0]`).
#[derive(Debug, Default)]
struct CallsProgress {
    /// Everything before this offset has been sent or held in `pending`.
    settled: usize,
    /// `buffer` length when it was last searched for `</tool_call>`.
    searched: usize,
    /// Non-whitespace text between calls, held until a call is executed: with
    /// none executed the whole text is content, calls included.
    pending: String,
    /// Content sent from `buffer`, and calls sent.
    sent_content: usize,
    sent_calls: usize,
}

struct AxK3Native {
    tools: Vec<Tool>,
    /// Grammar description for the shared guided reader only.
    grammar: WrappedBlockScanner<crate::tool_calling::ax_k3::AxK3Emitter>,
    starting_state: UnifiedParserStartingState,
    mode: Mode,
    /// Unconsumed input (marker prefixes held back at chunk boundaries).
    buffer: String,
    calls: CallsProgress,
    next_tool_index: usize,
}

impl AxK3Native {
    fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            grammar: ax_k3_scanner(tools).with_reasoning(REASONING),
            starting_state: UnifiedParserStartingState::None,
            mode: Mode::Reasoning { at_start: true },
            buffer: String::new(),
            calls: CallsProgress::default(),
            next_tool_index: 0,
        }
    }

    fn initial_mode(starting_state: UnifiedParserStartingState) -> Mode {
        match starting_state {
            UnifiedParserStartingState::Response => Mode::Content,
            UnifiedParserStartingState::None | UnifiedParserStartingState::Reasoning => {
                Mode::Reasoning { at_start: true }
            }
        }
    }

    /// Bytes at the end of `text` that could start one of `markers`.
    fn partial_suffix(text: &str, markers: &[&str]) -> usize {
        markers
            .iter()
            .flat_map(|marker| (1..marker.len()).map(move |len| (*marker, len)))
            .filter(|(marker, len)| text.ends_with(&marker[..*len]))
            .map(|(_, len)| len)
            .max()
            .unwrap_or(0)
    }

    /// Remove and return `buffer[..len]`.
    fn take(&mut self, len: usize) -> String {
        let rest = self.buffer.split_off(len);
        std::mem::replace(&mut self.buffer, rest)
    }

    /// Emit everything but a possible marker prefix (all of it when `flush`).
    fn take_settled(&mut self, markers: &[&str], flush: bool) -> String {
        let keep = if flush {
            0
        } else {
            Self::partial_suffix(&self.buffer, markers)
        };
        self.take(self.buffer.len() - keep)
    }

    fn drain(&mut self, output: &mut UnifiedParserOutput, flush: bool) {
        loop {
            match self.mode {
                Mode::Reasoning { at_start } => {
                    if at_start {
                        if !flush
                            && self.buffer.len() < THINK_START.len()
                            && THINK_START.starts_with(self.buffer.as_str())
                        {
                            return; // a generated opener may still be arriving
                        }
                        if self.buffer.starts_with(THINK_START) {
                            self.take(THINK_START.len());
                        }
                        self.mode = Mode::Reasoning { at_start: false };
                    }
                    if let Some(end) = self.buffer.find(THINK_END) {
                        output.push_reasoning(self.take(end));
                        self.take(THINK_END.len());
                        self.mode = Mode::Content;
                        continue;
                    }
                    output.push_reasoning(self.take_settled(&[THINK_END], flush));
                    return;
                }
                Mode::Content => {
                    let call = self.buffer.find(BLOCK_START);
                    let think = self.buffer.find(THINK_START);
                    match (call, think) {
                        (Some(call), think) if think.is_none_or(|think| call < think) => {
                            output.push_text(self.take(call));
                            self.calls = CallsProgress::default();
                            self.mode = Mode::Calls;
                            continue;
                        }
                        (_, Some(think)) => {
                            output.push_text(self.take(think + THINK_START.len()));
                            self.mode = Mode::ContentThink;
                            continue;
                        }
                        _ => {
                            output.push_text(self.take_settled(&[BLOCK_START, THINK_START], flush));
                            return;
                        }
                    }
                }
                Mode::ContentThink => {
                    if let Some(end) = self.buffer.find(THINK_END) {
                        output.push_text(self.take(end + THINK_END.len()));
                        self.mode = Mode::Content;
                        continue;
                    }
                    output.push_text(self.take_settled(&[THINK_END], flush));
                    return;
                }
                Mode::Calls => {
                    if flush {
                        self.finish_calls(output);
                    } else {
                        self.send_closed_calls(output);
                    }
                    return;
                }
            }
        }
    }

    fn push_call(&mut self, output: &mut UnifiedParserOutput, call: ResolvedCall) {
        output.push_call(ToolCallDelta {
            tool_index: self.next_tool_index,
            name: Some(call.name),
            arguments: call.arguments,
            complete: true,
        });
        self.next_tool_index += 1;
        self.calls.sent_calls += 1;
    }

    /// Send every call whose `</tool_call>` has arrived. Only a new
    /// `</tool_call>` can close one, so the text is parsed only when one does.
    fn send_closed_calls(&mut self, output: &mut UnifiedParserOutput) {
        let mut from = self.calls.searched.saturating_sub(BLOCK_END.len() - 1);
        while !self.buffer.is_char_boundary(from) {
            from -= 1;
        }
        self.calls.searched = self.buffer.len();
        if !self.buffer[from..].contains(BLOCK_END) {
            return;
        }
        while let Some(closed) = next_closed_call(&self.buffer, self.calls.settled, &self.tools) {
            let gap = &self.buffer[self.calls.settled..closed.start];
            if !gap.trim().is_empty() {
                self.calls.pending.push_str(gap);
            }
            self.calls.settled = closed.end;
            if self.calls.sent_calls == 0 && closed.call.is_none() {
                continue; // nothing executed yet: keep holding
            }
            let pending = std::mem::take(&mut self.calls.pending);
            self.calls.sent_content += pending.len();
            output.push_text(pending);
            if let Some(call) = closed.call {
                self.push_call(output, call);
            }
        }
    }

    /// End of stream: what [`resolve`] gives for the whole text, less what was
    /// sent (a prefix of it).
    fn finish_calls(&mut self, output: &mut UnifiedParserOutput) {
        let held = std::mem::take(&mut self.buffer);
        let (calls, content) = resolve(&held, &self.tools);
        if calls.is_empty() {
            output.push_text(held);
            return;
        }
        debug_assert!(self.calls.sent_calls <= calls.len());
        output.push_text(
            content
                .get(self.calls.sent_content..)
                .unwrap_or_default()
                .to_string(),
        );
        let sent = self.calls.sent_calls;
        for call in calls.into_iter().skip(sent) {
            self.push_call(output, call);
        }
    }
}

impl NativeUnified for AxK3Native {
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn guided_reasoning(&self) -> Option<GuidedReasoning> {
        Some(GuidedReasoning::Pair(REASONING))
    }

    fn guided_grammar(&self) -> GuidedGrammar {
        GuidedGrammar {
            control_markers: self.grammar.control_markers().to_vec(),
            invoke_start: self.grammar.invoke_start().to_string(),
            invoke_end: self.grammar.invoke_end().to_string(),
            invoke_boundary_factory: self
                .grammar
                .invoke_boundary_factory()
                .and_then(InvokeBoundaryFactory::for_guided),
            guided_prefix_policy: None,
            guided_prefix_factory: None,
        }
    }

    fn apply_native_init(&mut self, starting_state: UnifiedParserStartingState) {
        self.reset_native();
        self.restore_native_state(starting_state);
    }

    fn restore_native_state(&mut self, starting_state: UnifiedParserStartingState) {
        self.starting_state = starting_state;
        self.mode = Self::initial_mode(starting_state);
    }

    fn push_native(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(delta);
        self.drain(output, false);
        Ok(())
    }

    fn finish_native(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        self.drain(output, true);
        Ok(())
    }

    fn reset_native(&mut self) -> String {
        self.mode = Self::initial_mode(self.starting_state);
        self.next_tool_index = 0;
        self.calls = CallsProgress::default();
        std::mem::take(&mut self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::ax_k3::tests::{call, tools};
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedToolOutputMode, assemble, create_unified_parser_for_family,
    };
    use serde_json::json;

    fn weather() -> String {
        call("typed", &[("s", "서울"), ("i", "3")])
    }

    fn weather_event() -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: "typed".into(),
            arguments: json!({"s": "서울", "i": 3}),
        }
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    fn run(init: &UnifiedParserInit, chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = create_unified_parser_for_family("ax_k3", &tools()).expect("registry");
        parser.initialize_request(init.clone()).expect("initialize");
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).expect("push"));
        }
        events.extend(parser.finish().expect("finish").events);
        assemble(&events)
    }

    /// Whole input, every two-way UTF-8 split, and per character must agree.
    fn parse(init: UnifiedParserInit, input: &str) -> Vec<UnifiedEvent> {
        let whole = run(&init, &[input]);
        for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                run(&init, &[&input[..at], &input[at..]]),
                whole,
                "split {at} of {input:?}"
            );
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(run(&init, &chars), whole, "per-character {input:?}");
        whole
    }

    fn native() -> UnifiedParserInit {
        UnifiedParserInit::default()
    }

    fn response() -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..UnifiedParserInit::default()
        }
    }

    /// Guided output as Dynamo starts it for this family: inside the
    /// template-opened thought. (parsers-v2 0.7.24's guided reader does not
    /// infer that thought from a bare `</think>` under `None`.)
    fn guided(named_tool: Option<&str>) -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Reasoning,
            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                named_tool: named_tool.map(str::to_string),
            },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
            ..UnifiedParserInit::default()
        }
    }

    /// Streaming differential against the reference parser (see
    /// `tool_calling::ax_k3::tests::matches_reference_cases`):
    /// each case is fed per character from the answer channel. Text before the
    /// first call streams live, so whitespace-only content the reference drops
    /// may remain.
    #[test]
    fn streaming_matches_reference_cases() {
        let cases = crate::tool_calling::ax_k3::tests::reference_cases();
        let mut failures = Vec::new();
        for case in &cases {
            let input = case["text"].as_str().expect("text");
            let mut parser = create_unified_parser_for_family("ax_k3", &tools()).expect("registry");
            parser.initialize_request(response()).expect("initialize");
            let mut output = UnifiedParserOutput::default();
            for ch in input.chars() {
                parser
                    .parse_into(&ch.to_string(), &mut output)
                    .expect("push");
            }
            output
                .events
                .extend(parser.finish().expect("finish").events);
            let mut content = String::new();
            let mut calls = Vec::new();
            for event in output.events {
                match event {
                    crate::unified::UnifiedParserEvent::Text(text) => content.push_str(&text),
                    crate::unified::UnifiedParserEvent::ToolCall(call) => calls.push(
                        json!({"name": call.name.expect("name"), "arguments": call.arguments}),
                    ),
                    other => panic!("unexpected event {other:?}"),
                }
            }
            let want = case["content"].as_str().expect("content");
            let content_ok = content == want || (want.is_empty() && content.trim().is_empty());
            if calls != *case["calls"].as_array().expect("calls") || !content_ok {
                failures.push(format!("{input:?}: got {calls:?} / {content:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases differ:\n{}",
            failures.len(),
            cases.len(),
            failures[..failures.len().min(8)].join("\n")
        );
    }

    #[test]
    fn registry_and_aliases_construct_the_family() {
        for name in ["ax_k3", "axk3", "ax-k3"] {
            let parser = create_unified_parser_for_family(name, &tools()).expect(name);
            assert!(parser.preserve_special_tokens(), "{name}");
        }
    }

    /// The reference reasoning parser's split cases (`deepseek_r1` / vLLM
    /// `test_axk3_reasoning_parser.py` `SPLIT_CASES`).
    #[test]
    fn reference_reasoning_split_cases() {
        let call = "<tool_call>get_weather<|parameter|>city\nTokyo<|/parameter|></tool_call>";
        let cases: Vec<(String, Vec<UnifiedEvent>)> = vec![
            (
                "Reasoning</think>The rest".into(),
                vec![reasoning("Reasoning"), text("The rest")],
            ),
            ("Reasoning</think>".into(), vec![reasoning("Reasoning")]),
            ("</think>content".into(), vec![text("content")]),
            (
                "multi\nline</think>rest\nthat".into(),
                vec![reasoning("multi\nline"), text("rest\nthat")],
            ),
            ("<think>R</think>C".into(), vec![reasoning("R"), text("C")]),
            // Only the first </think> splits.
            (
                "R</think>A</think>B".into(),
                vec![reasoning("R"), text("A</think>B")],
            ),
            // A tool tag cannot end reasoning without </think>.
            (
                format!("thinking{call}"),
                vec![reasoning(&format!("thinking{call}"))],
            ),
            (call.to_string(), vec![reasoning(call)]),
            (
                format!("x{call}y</think>final"),
                vec![reasoning(&format!("x{call}y")), text("final")],
            ),
            ("still thinking".into(), vec![reasoning("still thinking")]),
        ];
        for (input, expected) in cases {
            assert_eq!(parse(native(), &input), expected, "{input:?}");
        }
    }

    #[test]
    fn reasoning_content_and_call() {
        assert_eq!(
            parse(
                native(),
                &format!("날씨를 확인하자.</think>\n\n확인할게요. {}", weather())
            ),
            vec![
                reasoning("날씨를 확인하자."),
                text("\n\n확인할게요. "),
                weather_event()
            ]
        );
    }

    #[test]
    fn every_call_is_executed_with_the_text_between() {
        let input = format!(
            "r</think>먼저 {} 다시. {}\n끝.",
            call("ping", &[]),
            weather()
        );
        let events = parse(native(), &input);
        let calls: Vec<_> = events
            .iter()
            .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
            .cloned()
            .collect();
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                UnifiedEvent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            vec![
                UnifiedEvent::ToolCall {
                    name: "ping".into(),
                    arguments: json!({})
                },
                weather_event()
            ]
        );
        assert_eq!(content, "먼저  다시. \n끝.");
    }

    /// A call is sent when its `</tool_call>` arrives, not at end of stream;
    /// text after it waits for the next call.
    #[test]
    fn calls_are_sent_as_they_close() {
        let mut parser = create_unified_parser_for_family("ax_k3", &tools()).expect("registry");
        parser.initialize_request(response()).expect("initialize");
        let ping = call("ping", &[]);
        let (head, tail) = ping.split_at(ping.len() - 3);
        let names = |events: Vec<crate::unified::UnifiedParserEvent>| -> Vec<String> {
            events
                .into_iter()
                .map(|event| match event {
                    crate::unified::UnifiedParserEvent::Text(text) => format!("text:{text}"),
                    crate::unified::UnifiedParserEvent::ToolCall(call) => {
                        format!("call:{}", call.name.expect("name"))
                    }
                    other => panic!("unexpected event {other:?}"),
                })
                .collect()
        };
        assert_eq!(
            names(parser.push(&format!("a {head}")).expect("push")),
            vec!["text:a "]
        );
        assert_eq!(names(parser.push(tail).expect("push")), vec!["call:ping"]);
        assert!(names(parser.push(" 그리고 ").expect("push")).is_empty());
        assert_eq!(
            names(parser.push(&weather()).expect("push")),
            vec!["text: 그리고 ", "call:typed"]
        );
        assert!(names(parser.finish().expect("finish").events).is_empty());
    }

    /// Before any call is executed, a call with an empty name keeps the text
    /// held: with no executed call the whole answer is content.
    #[test]
    fn nameless_call_holds_the_text_until_a_call_is_executed() {
        let empty = "<tool_call></tool_call>";
        assert_eq!(
            parse(response(), &format!("a {empty} b")),
            vec![text(&format!("a {empty} b"))]
        );
        let events = parse(response(), &format!("a {empty} b {}", weather()));
        assert_eq!(events.last(), Some(&weather_event()));
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                UnifiedEvent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "a  b ");
    }

    #[test]
    fn unclosed_call_is_content_and_completed_calls_survive() {
        let open = "<tool_call>typed<|parameter|>s\n서";
        assert_eq!(
            parse(native(), &format!("r</think>ok {open}")),
            vec![reasoning("r"), text(&format!("ok {open}"))]
        );
        assert_eq!(
            parse(native(), &format!("r</think>{}{open}", weather())),
            vec![reasoning("r"), weather_event(), text(open)]
        );
    }

    #[test]
    fn think_block_in_the_answer_is_content() {
        let input = format!("r</think>a<think>{}</think>b", call("ping", &[]));
        assert_eq!(
            parse(native(), &input),
            vec![
                reasoning("r"),
                text(&format!("a<think>{}</think>b", call("ping", &[])))
            ]
        );
    }

    #[test]
    fn response_start_skips_reasoning() {
        assert_eq!(
            parse(response(), &format!("답변 {}", weather())),
            vec![text("답변 "), weather_event()]
        );
    }

    #[test]
    fn verbatim_argument_spelling_reaches_the_event() {
        let mut parser = create_unified_parser_for_family("ax_k3", &tools()).expect("registry");
        parser.initialize_request(response()).expect("initialize");
        let mut events = parser.push(&call("typed", &[("f", "1.50")])).expect("push");
        events.extend(parser.finish().expect("finish").events);
        let arguments: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                crate::unified::UnifiedParserEvent::ToolCall(call) => Some(call.arguments.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(arguments, vec![r#"{"f": 1.50}"#]);
    }

    #[test]
    fn guided_required_array_after_reasoning() {
        let input = r#"plan</think>[{"name":"typed","parameters":{"s":"서울","i":3}}]"#;
        assert_eq!(
            parse(guided(None), input),
            vec![reasoning("plan"), weather_event()]
        );
    }

    #[test]
    fn guided_named_object_with_and_without_reasoning() {
        assert_eq!(
            parse(guided(Some("typed")), r#"plan</think>{"s":"서울","i":3}"#),
            vec![reasoning("plan"), weather_event()]
        );
        let init = UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..guided(Some("typed"))
        };
        assert_eq!(parse(init, r#"{"s":"서울","i":3}"#), vec![weather_event()]);
    }

    #[test]
    fn reset_reuses_the_parser_for_a_new_stream() {
        let mut parser = create_unified_parser_for_family("ax_k3", &tools()).expect("registry");
        parser.initialize_request(native()).expect("initialize");
        let _ = parser.push("r</think>half <tool_call>ty").expect("push");
        parser.reset();
        parser.initialize_request(native()).expect("initialize");
        let mut events = parser
            .push(&format!("r</think>{}", weather()))
            .expect("push");
        events.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&events), vec![reasoning("r"), weather_event()]);
    }
}
