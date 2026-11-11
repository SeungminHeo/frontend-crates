# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate: python3 -I gen_reasoning_cases.py <solar_open2_reasoning_parser.py> reasoning_cases.json
#   (UpstageAI/vllm tag v0.22.0-solar-open2; SOLAR_FUZZ sets the fuzz size)
"""Run the reference Solar Open2 reasoning parser's non-streaming
extract_reasoning (vLLM stubbed) over fixed and fuzzed outputs and write
{text, reasoning, content} cases for the Rust differential test."""
import json, os, random, re, sys, types

ref_path, out_path = sys.argv[1:3]


def module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


class ReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer


class DeltaMessage:
    def __init__(self, reasoning=None, content=None):
        self.reasoning, self.content = reasoning, content


sys.modules["regex"] = re  # the reference's patterns are plain `re` syntax
for name in ["vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.openai.engine"]:
    module(name)
module("vllm.entrypoints.openai.engine.protocol", DeltaMessage=DeltaMessage)
module("vllm.reasoning", ReasoningParser=ReasoningParser)

# Python 3.9 cannot evaluate `str | None` annotations; defer them.
ref = types.ModuleType("solar_ref")
ref.__file__ = ref_path
sys.modules["solar_ref"] = ref
with open(ref_path) as source:
    code = "from __future__ import annotations\n" + source.read()
exec(compile(code, ref_path, "exec"), ref.__dict__)
parser = ref.SolarOpen2ReasoningParser(None)


def reference(text):
    reasoning, content = parser.extract_reasoning(text, None)
    return reasoning or "", content or ""


BLOCK = ("<|tool_call:start|>get_weather\n<|tool_arg:start|>city<|tool_arg:value|>Seoul"
         "<|tool_arg:end|>\n<|tool_call:end|>")
fixed = [
    "I need to analyze<|think:end|>The answer is 42",
    "My reasoning<|think:end|>",
    "<|think:end|>Direct answer",
    "Step 1\nStep 2\n<|think:end|>Final",
    f"I should check.\n{BLOCK}\n<|think:end|>",
    f"thinking\n{BLOCK}\n<|think:end|>The answer is 42",
    f"first\n{BLOCK}\nthen\n{BLOCK}\n<|think:end|>",
    "Should I emit <|tool_call:start|> here? No.\n<|think:end|>Answer",
    "I could use <tool_call>{}</tool_call> here.<|think:end|>Answer",
]

pieces = [
    "think", "생각", " ", "\n", "\t", "a b", "<tool_call>", "</tool_call>",
    "<|think:start|>", "<|think:end|>", "<|tool_call:start|>", "<|tool_call:end|>",
    "<|tool_arg:start|>", "<|tool_arg:value|>", "<|tool_arg:end|>", "f\n", "k", "v",
    BLOCK, "<|", "<|tool", "|>",
]
rng = random.Random(1111)
fuzz = []
for _ in range(int(os.environ.get("SOLAR_FUZZ", "1500"))):
    text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 14)))
    if "<|think:end|>" not in text:
        text += "<|think:end|>" + "".join(rng.choice(pieces) for _ in range(rng.randint(0, 4)))
    fuzz.append(text)

cases = []
for text in fixed + fuzz:
    reasoning, content = reference(text)
    cases.append({"text": text, "reasoning": reasoning, "content": content})
with open(out_path, "w") as out:
    json.dump(cases, out, ensure_ascii=False, indent=0)
print(f"{len(cases)} cases")
