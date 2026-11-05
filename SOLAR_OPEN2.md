<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Solar Open2 downstream parser

This branch adds `solar_open2` to the legacy reasoning and tool parser registries
at `dynamo-parsers` 11.0.7, for current frontend-crates main. It is a git-pinned
patch, not a crates.io release. Tracking: SeungminHeo/dynamo#1.

Reasoning uses `<|think:start|>` and `<|think:end|>`, starts in reasoning mode,
and exits into a native tool call even when the model omits the think closer.
The caller must bypass reasoning for requests whose template closes the think
block in the prompt. Marker-looking text within tool argument values is literal.

Native tool calls use a newline after the function name and
`<|tool_arg:start|>key<|tool_arg:value|>value<|tool_arg:end|>` arguments.
Values are coerced from the supplied tool schema; failed conversions keep the
original string. Nullable type arrays select the first non-null type. A literal
`null` becomes JSON null, matching the reference parser. Calls require explicit
closing fences. Text after the first call is discarded by the batch parser.
The existing jail owns incremental buffers and UTF-8 marker splits.

Dynamo's Solar response policy holds executable calls until an actual backend
`stop`; a length limit, repetition, filter, cancellation, error or unterminated
stream drops them all.
That finish policy belongs to the consumer: a standalone parser cannot infer
backend termination from text alone.

The parser depends on the companion `dynamo-protocols` 10.0.0 workspace crate.
Consumers must patch both packages from this git repository at the same commit
to keep protocol types consistent across the renderer and both parser crates.

Validation:

```sh
cargo test -p dynamo-parsers --lib
cargo clippy -p dynamo-parsers --all-targets --no-deps -- -D warnings
```

Reference formats: [Upstage Solar Open2 model card](https://huggingface.co/upstage/Solar-Open2-250B)
and [Upstage vLLM tool parser](https://github.com/UpstageAI/vllm/blob/v0.22.0-solar-open2/vllm/tool_parsers/solar_open2_tool_parser.py).

The companion `dynamo-protocols` 10.0.0 patch preserves the engine finish reason
`repetition` in chat and text completion responses. Pin both crates to the same
commit; this does not add or configure repetition detection.

## Maintained branches

- `solar-open2-10.0.7`: parser 10.0.7 / protocols 8.2.0 for Dynamo 1.6.1.
- `solar-open2-10.2.2`: parser 10.2.2 / protocols 8.4.0 for Dynamo development main.
- `main`: forward ports on current upstream frontend-crates; consumer pins must match its package versions.

All three include the unmarked zero-call JSON-array preservation fix tracked in
SeungminHeo/dynamo#2. Native tagged calls and guided required/named calls retain
their existing semantics. Normal tool-shaped bare JSON remains accepted.
