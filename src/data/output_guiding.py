import json
import re
import types
import typing
from typing import Any, List, Type, Optional, Union

import json5
from pydantic import BaseModel, ValidationError

from pydantic_core import PydanticUndefined

from src.data.schemas import create_dynamic_pydantic_schema


def resolve_schema_model(schema_arg: Any) -> Type[BaseModel]:
    """
    Resolves schema argument (schema dict from config.yaml, or Type[BaseModel]) to a Pydantic model class.
    """
    if isinstance(schema_arg, type) and issubclass(schema_arg, BaseModel):
        return schema_arg
    if isinstance(schema_arg, dict):
        return create_dynamic_pydantic_schema(schema_arg)
    
    # Fallback to dynamic schema creation
    return create_dynamic_pydantic_schema({})


def _unwrap_type(annotation: Any) -> Any:
    """
    Unwrap Optional, Union, and Annotated types to get the base underlying type (int, float, str, bool, list).
    """
    if annotation is None:
        return str

    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _unwrap_type(args[0])

    if hasattr(annotation, "__args__") and annotation.__args__:
        if origin is list or origin is List:
            return annotation

    return annotation


def _extract_json_candidates(raw_output: str) -> list[str]:
    """
    Extracts potential JSON strings from raw LLM output using multiple strategies:
    1. Post-think block (content after </think>, </reasoning>, or </thought>)
    2. Markdown code fences (```json ... ```)
    3. Balanced brace JSON objects/arrays extracted from text
    4. Substring bounded by first and last brace
    """
    candidates = []

    # Strategy 1: Post-thinking content (if thinking tags exist)
    think_match = re.search(r"</(?:think|reasoning|thought)>\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
    if think_match and think_match.group(1).strip():
        post_think = think_match.group(1).strip()
        candidates.append(post_think)

    # Strategy 2: Code blocks ```json ... ``` or ``` ... ```
    code_fence_matches = re.findall(r"```(?:json)?\s*({[\s\S]*?}|\[[\s\S]*?\])\s*```", raw_output, re.IGNORECASE)
    for block in reversed(code_fence_matches):  # Reversed to prioritize the last code block
        if block.strip() and block.strip() not in candidates:
            candidates.append(block.strip())

    # Strategy 3: Find all balanced JSON objects {...} or arrays [...] using bracket stack
    def find_balanced_blocks(text: str) -> list[str]:
        blocks = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] in ("{", "["):
                start = i
                open_char = text[i]
                close_char = "}" if open_char == "{" else "]"
                depth = 0
                in_string = False
                escape = False
                for j in range(i, n):
                    char = text[j]
                    if in_string:
                        if escape:
                            escape = False
                        elif char == "\\":
                            escape = True
                        elif char == '"':
                            in_string = False
                    else:
                        if char == '"':
                            in_string = True
                        elif char == open_char:
                            depth += 1
                        elif char == close_char:
                            depth -= 1
                            if depth == 0:
                                blocks.append(text[start : j + 1])
                                i = j
                                break
            i += 1
        return blocks

    balanced_blocks = find_balanced_blocks(raw_output)
    # Reverse balanced blocks so that the last JSON object (final answer) is tested first
    for block in reversed(balanced_blocks):
        if block not in candidates:
            candidates.append(block)

    # Strategy 4: Fallback range from first '{' to last '}' or '[' to ']'
    try:
        start_brace = raw_output.find("{")
        start_bracket = raw_output.find("[")
        if start_brace != -1 or start_bracket != -1:
            if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                start = start_brace
                end = raw_output.rindex("}") + 1
            else:
                start = start_bracket
                end = raw_output.rindex("]") + 1
            fallback_candidate = raw_output[start:end]
            if fallback_candidate not in candidates:
                candidates.append(fallback_candidate)
    except ValueError:
        pass

    # Strategy 5: Full output trimmed
    trimmed = raw_output.strip()
    if trimmed not in candidates:
        candidates.append(trimmed)

    return candidates


def extract_structured_output(
    sample: dict[str, Any],
    output_schema_model: Type[BaseModel],
    col_to_structure: str = "output_text",
) -> dict[str, Any]:
    """
    Extracts structured output from raw model output using a multi-stage lenient
    parsing strategy
    """
    raw_output = sample.get(col_to_structure)
    if not isinstance(raw_output, str) or not raw_output.strip():
        print("Warning: Missing or empty column to structure. Returning default.")
        return _get_default_values(output_schema_model)

    candidates = _extract_json_candidates(raw_output)

    # Attempt parsing each candidate (prioritizing post-think and final JSON candidates)
    for candidate in candidates:
        # Direct Pydantic parse
        try:
            validated_output = output_schema_model.model_validate_json(candidate)
            return validated_output.model_dump()
        except Exception:
            pass

        # Parse with lenient json5 library + Pydantic model validation
        try:
            data = json5.loads(candidate)
            if isinstance(data, dict):
                validated_output = output_schema_model.model_validate(data)
                return validated_output.model_dump()
        except Exception:
            pass

        # Attempt to repair truncated JSON
        try:
            repaired_json = _repair_truncated_json(candidate)
            validated_output = output_schema_model.model_validate_json(repaired_json)
            return validated_output.model_dump()
        except Exception:
            pass

        try:
            repaired_data = json5.loads(repaired_json)
            if isinstance(repaired_data, dict):
                validated_output = output_schema_model.model_validate(repaired_data)
                return validated_output.model_dump()
        except Exception:
            pass

    # Field-by-field regex extraction on post-think text or raw output
    print("Warning: All JSON block parsing methods failed. Attempting field-by-field regex extraction.")
    extracted_data = {}
    
    # Strip thinking block for regex searching if present
    search_text = raw_output
    think_match = re.search(r"</(?:think|reasoning|thought)>\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
    if think_match and think_match.group(1).strip():
        search_text = think_match.group(1).strip()

    try:
        for field_name, field_info in output_schema_model.model_fields.items():
            base_type = _unwrap_type(field_info.annotation)
            value = _extract_field_with_regex(search_text, field_name, base_type)
            if value is None and search_text != raw_output:
                value = _extract_field_with_regex(raw_output, field_name, base_type)
            if value is not None:
                extracted_data[field_name] = value
    except Exception as e:
        print(f"Warning: Exception during regex extraction: {e}")

    if not extracted_data:
        print("Error: Could not extract any fields with regex. Returning default values.")
        print("\n###########################")
        print(f"Raw output was: {raw_output}")
        print("###########################\n")
        return _get_default_values(output_schema_model)

    print(f"Success: Extracted partial data with regex: {list(extracted_data.keys())}")
    defaults = _get_default_values(output_schema_model)
    defaults.update(extracted_data)

    try:
        final_model = output_schema_model.model_validate(defaults)
        return final_model.model_dump()
    except Exception as e:
        print(f"Warning: Error during final output field validation: {e}")
        return defaults


def _repair_truncated_json(s: str) -> str:
    """
    Append missing brackets/braces to a potentially truncated JSON string
    """
    s = s.strip()
    closures = {'{': '}', '[': ']'}
    stack = []
    for char in s:
        if char in closures:
            stack.append(closures[char])
        elif stack and char == stack[-1]:
            stack.pop()
    
    # Append missing closing characters
    s += "".join(reversed(stack))

    return s


def _extract_field_with_regex(
    text: str,
    field_name: str,
    field_type: Type,
) -> Any | None:
    """
    Extract a single field value using a type-aware regex pattern.
    Uses re.findall / last match to prefer final answer over reasoning prompt text.
    """
    # Pattern for null
    pattern_null = rf'"{field_name}"\s*:\s*null'
    if re.search(pattern_null, text, re.IGNORECASE):
        return None
    
    # Pattern for string
    if field_type == str:
        pattern = rf'"{field_name}"\s*:\s*"((?:\\"|[^"])*)"'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].replace('\\"', '"')
        return None

    # Pattern for number (int/float)
    if field_type in (int, float):
        pattern = rf'"{field_name}"\s*:\s*(-?\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        try:
            val = field_type(matches[-1])
            if isinstance(val, int) and (val > 2**63 - 1 or val < -2**63):
                return -1
            return val
        except (ValueError, TypeError):
            return None

    # Pattern for boolean
    if field_type == bool:
        pattern = rf'"{field_name}"\s*:\s*(true|false)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower() == 'true'
        return None
        
    # Pattern for lists of strings or numbers
    if hasattr(field_type, "__origin__") and field_type.__origin__ in (list, List):
        pattern_str = rf'"{field_name}"\s*:\s*\[\s*((?:"(?:\\"|[^"])*"\s*,\s*)*"(?:\\"|[^"])*")\s*\]'
        matches = re.findall(pattern_str, text)
        if matches:
            return [s.strip().strip('"') for s in matches[-1].split(',') if s.strip()]

        pattern_num = rf'"{field_name}"\s*:\s*\[\s*((-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?)\s*\]'
        matches = re.findall(pattern_num, text)
        if matches:
            item_type = field_type.__args__[0] if getattr(field_type, "__args__", None) else int
            try:
                return [item_type(n.strip()) for n in matches[-1].split(',') if n.strip()]
            except (ValueError, TypeError):
                return None

    return None


def _get_default_values(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Build a dictionary of default values from a Pydantic model
    """
    defaults = {}
    for name, field in model.model_fields.items():
        if field.default_factory:
            defaults[name] = field.default_factory()
        elif field.default is not PydanticUndefined:
            defaults[name] = field.default
        else:
            field_type = field.annotation
            if hasattr(field_type, "__origin__") and field_type.__origin__ in (list, List):
                defaults[name] = []
            else:
                defaults[name] = None

    return defaults