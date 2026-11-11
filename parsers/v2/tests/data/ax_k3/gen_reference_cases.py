# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate: python3 -I gen_reference_cases.py <axk3_tool_parser.py> stub_vllm_utils.py \
#     reference_cases.json tools.json   (AX_K3_FUZZ / AX_K3_CALLS set the fuzz sizes,
#     AX_K3_POLICY the call_group_policy, default "all")
"""Run the reference axk3_tool_parser.py (vLLM stubbed) over fixed and fuzzed
inputs and write {text, calls, content} cases for the Rust differential test."""
import importlib.util, json, os, random, sys, types

ref_path, utils_path, out_path = sys.argv[1:4]

def module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod

class _Any:
    def __init__(self, *a, **k): self.__dict__.update(k)
    def model_dump(self, **k): return dict(self.__dict__)

class ToolParser:
    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer; self.tools = tools
        self.prev_tool_call_arr = []; self.streamed_args_for_tool = []
        self.current_tool_id = -1; self.current_tool_name_sent = False
    @property
    def vocab(self): return {}

class ToolParserManager:
    @staticmethod
    def register_module(name):
        return lambda cls: cls

for name in ["vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.generate",
             "vllm.entrypoints.generate.base", "vllm.entrypoints.openai.responses", "vllm.tool_parsers"]:
    module(name)
module("vllm.entrypoints.chat_utils", make_tool_call_id=lambda: "id")
module("vllm.entrypoints.openai.chat_completion")
module("vllm.entrypoints.openai.chat_completion.protocol", ChatCompletionRequest=_Any)
module("vllm.entrypoints.generate.base.protocol", **{n: _Any for n in [
    "DeltaFunctionCall", "DeltaMessage", "DeltaToolCall", "ExtractedToolCallInformation",
    "FunctionCall", "ToolCall"]})
module("vllm.entrypoints.openai.responses.protocol", ResponsesRequest=_Any)
import logging
module("vllm.logger", init_logger=lambda n: logging.getLogger(n))
module("vllm.tokenizers", TokenizerLike=object)
module("vllm.tool_parsers.abstract_tool_parser", Tool=dict, ToolParser=ToolParser,
       ToolParserManager=ToolParserManager)
utils = module("vllm.tool_parsers.utils")
exec(open(utils_path).read(), utils.__dict__)
TOOLS = json.load(open(sys.argv[4]))
def find_tool_properties(tools, name):
    for tool in tools:
        if tool["name"] == name:
            return tool["parameters"].get("properties", {})
    return {}
utils.find_tool_properties = find_tool_properties

spec = importlib.util.spec_from_file_location("axk3_ref", ref_path)
ref = importlib.util.module_from_spec(spec)
sys.modules["axk3_ref"] = ref
spec.loader.exec_module(ref)
logging.disable(logging.CRITICAL)

def props_for(name):
    return find_tool_properties(TOOLS, name)

# Dynamo serves A.X-K3 with call_group_policy="all" (the reference default is
# "last"); AX_K3_POLICY overrides it.
POLICY = os.environ.get("AX_K3_POLICY", "all")

def reference(text):
    """`extract_tool_calls` under POLICY."""
    if ref.TOOL_CALL_START not in text:
        return [], text
    scan = ref._scan(text, final=True, props_for=props_for)
    executed, content = ref._resolve(text, scan, POLICY)
    if not executed:
        return [], text
    calls = [{"name": c.name, "arguments": ref._render_arguments(c, props_for(c.name))}
             for c in executed]
    return calls, content or ""

def call(name, params):
    return "<tool_call>" + name + "".join(f"<|parameter|>{k}\n{v}<|/parameter|>" for k, v in params) + "</tool_call>"

fixed = [
    "안녕하세요",
    call("typed", [("s", "007"), ("i", "42"), ("f", "1.50"), ("b", "false"), ("arr", '["a", "b\\"c"]'),
                   ("obj", '{"k": [1, null]}'), ("maybe", "null"), ("untyped", "123")]),
    call("typed", [("i", " 7 "), ("b", "True"), ("f", "2"), ("i", "9")]),
    "먼저 " + call("ping", []) + " 잠깐, 다시. " + call("typed", [("s", "x")]) + " 끝.",
    "\n" + call("ping", []) + "\n" + call("ping", []) + "\n",
    call("ping", []) + "<tool_call>typed<|parameter|>s\n서",
    "<tool_call>write_file<|parameter|>path\na.py</tool_call>",
    "<tool_call>typed<|parameter|>top_k30<|/parameter|><|parameter|>s\nok<|/parameter|></tool_call>",
    "<tool_call>typed<|parameter|>top_k<|/parameter|></tool_call>",
    "<tool_call>typed<|parameter|>top_k 5</tool_call>",
    "<think>" + call("ping", []) + "</think>after",
    "<think>" + call("ping", []) + " no close",
    "<tool_call></tool_call>",
    "a" + call("", []) + "b" + call("ping", []),
    call("typed", [("maybe", "plain text"), ("unknown", "not json"), ("unknown2", "[1, 2]"),
                   ("unknown3", '"quoted"'), ("u4", "1e3"), ("u5", "NaN"), ("i", "abc")]),
    call("write_file", [("content", 'a "quote" \\ back\tslash\n<tool_call>x</tool_call> 한글\n\n')]),
]

FRAGMENTS = ["<tool_call>", "</tool_call>", "<|parameter|>", "<|/parameter|>", "<think>", "</think>",
             "typed", "ping", "write_file", "s", "i", "f", "b", "top_k", "maybe", "obj", "arr", "zz",
             "\n", " ", "\t", "30", "1.50", "-2", "true", "True", "null", '{"a": 1}', "[1, 2]",
             "text", "한글", '"q"', "abc", "\r\n"]
rng = random.Random(1110)
fuzz = []
for _ in range(int(os.environ.get("AX_K3_FUZZ", "4000"))):
    pieces = [rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 24))]
    if rng.random() < 0.5:
        pieces.insert(0, "<tool_call>")
    fuzz.append("".join(pieces))
for _ in range(int(os.environ.get("AX_K3_CALLS", "1500"))):
    parts = []
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.4:
            parts.append(rng.choice(["", " ", "\n", "다시 ", "ok."]))
        name = rng.choice(["typed", "ping", "write_file", "nope"])
        keys = ["s", "i", "f", "b", "obj", "arr", "maybe", "untyped", "top_k", "content", "x"]
        params = [(rng.choice(keys), rng.choice(FRAGMENTS[18:] + ["42", "", "  3  "]))
                  for _ in range(rng.randint(0, 4))]
        text = call(name, params)
        if rng.random() < 0.15:
            text = text[: rng.randint(0, len(text))]
        parts.append(text)
    fuzz.append("".join(parts))

cases = []
for text in fixed + fuzz:
    calls, content = reference(text)
    cases.append({"text": text, "calls": calls, "content": content})
json.dump(cases, open(out_path, "w"), ensure_ascii=False, indent=0)
print(len(cases), "cases;", sum(1 for c in cases if c["calls"]), "with calls")
