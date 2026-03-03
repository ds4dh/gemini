import re
import json
import json5
from typing import Any, Type
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefinedType

from src.data.schemas import PatientInfoSchema  #, ...
SCHEMA_REGISTRY = {
    "PatientInfoSchema": PatientInfoSchema,
    # ...
}


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

    # Direct Pydantic parse
    try:
        validated_output = output_schema_model.model_validate_json(raw_output.strip())
        print("Success: Direct parsing successful.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass

    # Isolate JSON block (from markdown or first/last braces)
    json_candidate = raw_output
    match = re.search(r"```(?:json)?\s*({.*}|\[.*\])\s*```", raw_output, re.DOTALL)
    if match:
        json_candidate = match.group(1)
    else:
        try:
            start_brace = raw_output.find("{")
            start_bracket = raw_output.find("[")

            if start_brace == -1 and start_bracket == -1:
                raise ValueError("No JSON object or array found")

            if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                start = start_brace
                end_char = "}"
            else:
                start = start_bracket
                end_char = "]"

            end = raw_output.rindex(end_char) + 1
            json_candidate = raw_output[start:end]
        except ValueError:
            pass
    try:
        validated_output = output_schema_model.model_validate_json(json_candidate)
        print("Success: Isolated and parsed JSON block.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass

    # Parse with lenient json5 library
    try:
        data = json5.loads(json_candidate)
        validated_output = output_schema_model.model_validate(data)
        print("Success: Parsed with lenient json5 library.")
        return validated_output.model_dump()
    except (ValidationError, Exception):
        pass

    # Attempt to repair truncated JSON
    try:
        repaired_json = _repair_truncated_json(json_candidate)
        validated_output = output_schema_model.model_validate_json(repaired_json)
        print("Success: Repaired truncated JSON and parsed.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass

    # Field-by-field regex extraction
    print("Warning: All parsing methods failed. Attempting field-by-field regex extraction.")
    extracted_data = {}
    for field_name, field_info in output_schema_model.model_fields.items():
        value = _extract_field_with_regex(json_candidate, field_name, field_info.annotation)
        if value is not None:
            extracted_data[field_name] = value

    if not extracted_data:
        print("Error: Could not extract any fields with regex. Returning default values.")
        return _get_default_values(output_schema_model)

    print(f"Success: Extracted partial data with regex: {list(extracted_data.keys())}")
    defaults = _get_default_values(output_schema_model)
    defaults.update(extracted_data)

    try:
        final_model = output_schema_model.model_validate(defaults)
        return final_model.model_dump()
    except ValidationError:
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
    Extract a single field value using a type-aware regex pattern
    """
    # Pattern for null
    pattern_null = rf'"{field_name}"\s*:\s*null'
    if re.search(pattern_null, text):
        return None
    
    # Pattern for string
    if field_type == str:
        pattern = rf'"{field_name}"\s*:\s*"((?:\\"|[^"])*)"'
        match = re.search(pattern, text)
        return match.group(1).replace('\\"', '"') if match else None

    # Pattern for number (int/float)
    if field_type in (int, float):
        pattern = rf'"{field_name}"\s*:\s*(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            val = field_type(match.group(1))
            if isinstance(val, int) and (val > 2**63 - 1 or val < -2**63):
                return -1
            return val
        except (ValueError, TypeError):
            return None

    # Pattern for boolean
    if field_type == bool:
        pattern = rf'"{field_name}"\s*:\s*(true|false)'
        match = re.search(pattern, text)
        return match.group(1) == 'true' if match else None
        
    # Pattern for lists of strings or numbers
    if hasattr(field_type, "__origin__") and field_type.__origin__ == list:

        # Simple case: list of strings
        pattern_str = rf'"{field_name}"\s*:\s*\[\s*((?:"(?:\\"|[^"])*"\s*,\s*)*"(?:\\"|[^"])*")\s*\]'
        match = re.search(pattern_str, text)
        if match:
            return [s.strip().strip('"') for s in match.group(1).split(',') if s.strip()]

        # Simple case: list of numbers
        pattern_num = rf'"{field_name}"\s*:\s*\[\s*((-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?)\s*\]'
        match = re.search(pattern_num, text)
        if match:
            item_type = field_type.__args__[0]
            try:
                return [item_type(n.strip()) for n in match.group(1).split(',') if n.strip()]
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
        elif not isinstance(field.default, PydanticUndefinedType):
            defaults[name] = field.default
        else:
            # For required fields, use a sensible empty default based on type
            field_type = field.annotation
            if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                defaults[name] = []
            else:
                defaults[name] = None

    return defaults