# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copied from vllm/tool_parsers/utils.py (vLLM fork commit 866987e247) so the
# reference A.X-K3 parser can run without vLLM installed.
from __future__ import annotations
import json, math
from typing import Any


def _is_json_finite(obj: Any) -> bool:
    """Whether *obj* can be serialized to valid JSON.

    ``json.dumps(..., allow_nan=False)`` raises ``ValueError`` on any
    non-finite float (``inf``/``-inf``/``nan``) anywhere in the value, so this
    detects non-finite floats nested inside parsed lists/dicts too.
    """
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False




def extract_types_from_schema(schema: Any) -> list[str]:
    """Extract all possible type strings from a JSON Schema definition.

    Handles ``type`` (string or list), ``enum`` value inference, and
    recursive ``anyOf``/``oneOf``/``allOf``.  Returns ``["string"]``
    when no type information can be determined.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                types.update(extract_types_from_schema(choice))

    return list(types) if types else ["string"]




_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}

def coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
    """Best-effort coercion of a raw string value to a JSON Schema type.

    Tries each type in priority order (null > integer > number > boolean >
    object > array > string) and returns the first successful coercion.
    Falls back to the original string when no coercion succeeds.

    Args:
        value: The raw string value from the model output.
        schema_type: One or more JSON Schema type strings
            (e.g. ``"string"`` or ``["string", "null"]``).
    """
    if isinstance(schema_type, str):
        schema_type = [schema_type]

    normalized_types = {
        _TYPE_ALIASES.get(key, key) for t in schema_type for key in [t.strip().lower()]
    }

    # Priority: null > integer > number > boolean > object > array > string
    type_priority = [
        "null",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
        "string",
    ]

    for candidate_type in type_priority:
        if candidate_type not in normalized_types:
            continue

        if candidate_type == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate_type == "string":
            return value
        if candidate_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate_type == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                # inf/-inf/nan are not valid JSON numbers. Fall through so
                # the value is preserved as a string instead of crashing
                # (int(float("inf")) raises OverflowError) or emitting
                # invalid JSON (json.dumps(inf) -> "Infinity").
                continue
            return val if val != int(val) else int(val)
        if candidate_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed
            # Non-finite floats (e.g. "[1e999]" -> [inf]) cannot be
            # serialized back to valid JSON; preserve the raw string.
            continue

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    # Reject non-finite results (e.g. json.loads("1e999") -> inf, or nested
    # inf/nan inside a parsed list/dict) which json.dumps would render as
    # invalid JSON (Infinity/NaN). Preserve the raw string instead.
    if not _is_json_finite(parsed):
        return value
    return parsed




def partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of *tag* that matches a suffix of *text*.

    E.g. text ending in ``"<tool_"`` returns 6 when tag is ``"<tool_call>"``.
    Returns 0 when there is no overlap.
    """
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


