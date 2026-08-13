from typing import Any, Type, Optional, List, Literal
from pydantic import BaseModel, ConfigDict, Field, create_model


def create_dynamic_pydantic_schema(schema_config: dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically builds a Pydantic BaseModel from a dictionary/YAML schema specification.
    Fields with `default: null` or `default: None` are treated as REQUIRED in the JSON payload
    (the key must be present), but NULLABLE in value (can be an explicitly typed value or `null`).
    """
    schema_name = schema_config.get("name", "DynamicClinicalExtractionSchema")
    fields_spec = schema_config.get("fields", {})

    if not fields_spec:
        # Default fallback field if no fields are defined
        fields_spec = {
            "mRS": {
                "type": "int",
                "description": "Modified Rankin Scale score (0 to 6, or null if unmentioned)",
                "ge": 0,
                "le": 6,
                "default": None,
            }
        }

    model_fields: dict[str, Any] = {}

    for field_name, field_def in fields_spec.items():
        if isinstance(field_def, str):
            field_def = {"type": field_def}

        raw_type = field_def.get("type", "str").lower()
        description = field_def.get("description", "")
        default_val = field_def.get("default", ...)

        # Extract validation bounds
        ge = field_def.get("ge")
        le = field_def.get("le")

        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description
        if ge is not None:
            field_kwargs["ge"] = ge
        if le is not None:
            field_kwargs["le"] = le

        # Resolve base type
        if raw_type in ("int", "integer"):
            base_type = int
        elif raw_type in ("float", "double", "number"):
            base_type = float
        elif raw_type in ("bool", "boolean"):
            base_type = bool
        elif raw_type in ("enum", "choice"):
            enum_values = field_def.get("enum_values", [])
            if not isinstance(enum_values, list) or not enum_values:
                raise ValueError(f"Enum field {field_name!r} must define a non-empty 'enum_values' list.")
            if any(not isinstance(v, (str, int, float, bool)) for v in enum_values):
                raise ValueError(f"Enum field {field_name!r} contains non-JSON-primitive value: {enum_values!r}")
            if default_val not in (..., None) and default_val not in enum_values:
                raise ValueError(f"Default value {default_val!r} for enum {field_name!r} not in {enum_values!r}.")

            base_type = Literal.__getitem__(tuple(enum_values))
        elif raw_type in ("list_str", "list[str]"):
            base_type = List[str]
        else:
            base_type = str

        # Handle nullability and required status
        # If default is None or ..., the field key is required, but value can be null
        is_nullable = default_val is None or default_val == ...
        py_type = Optional[base_type] if is_nullable else base_type

        if default_val == ... or default_val is None:
            # Ellipsis (...) in Field indicates a mandatory key in Pydantic validation and JSON schema
            field_tuple = (py_type, Field(..., **field_kwargs))
        elif hasattr(default_val, "__class__") and "Field" in default_val.__class__.__name__:
            field_tuple = (py_type, default_val)
        else:
            field_tuple = (py_type, Field(default=default_val, **field_kwargs))

        model_fields[field_name] = field_tuple

    dynamic_model = create_model(
        schema_name,
        __module__=__name__,
        __config__=ConfigDict(extra="forbid"),
        **model_fields,
    )
    globals()[schema_name] = dynamic_model
    return dynamic_model