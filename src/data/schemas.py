from pydantic import BaseModel, Field, create_model
from typing import Any, Type, Optional, List


def create_dynamic_pydantic_schema(schema_config: dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically builds a Pydantic BaseModel from a dictionary/YAML schema specification.

    Example schema_config:
    {
        "name": "ClinicalVariablesExtractionSchema",
        "fields": {
            "mRS": {
                "type": "int",
                "description": "Modified Rankin Scale score (0-6)",
                "ge": 0,
                "le": 6,
                "default": -1
            },
            "smoking_status": {
                "type": "enum",
                "description": "Patient smoking habit",
                "enum_values": ["Smoker", "Non-smoker", "Former-smoker", "Unknown"],
                "default": "Unknown"
            },
            "aneurysm_size_mm": {
                "type": "float",
                "description": "Max aneurysm diameter in mm",
                "default": None
            }
        }
    }
    """
    schema_name = schema_config.get("name", "DynamicClinicalExtractionSchema")
    fields_spec = schema_config.get("fields", {})

    if not fields_spec:
        # Default fallback field if no fields are defined
        fields_spec = {
            "mRS": {
                "type": "int",
                "description": "Modified Rankin Scale score (0 to 6, or -1 if unmentioned)",
                "ge": -1,
                "le": 6,
                "default": -1
            }
        }

    model_fields: dict[str, Any] = {}

    for field_name, field_def in fields_spec.items():
        if isinstance(field_def, str):
            field_def = {"type": field_def}

        raw_type = field_def.get("type", "str").lower()
        description = field_def.get("description", "")
        default_val = field_def.get("default", ...)
        ge = field_def.get("ge")
        le = field_def.get("le")

        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description
        if ge is not None:
            field_kwargs["ge"] = ge
        if le is not None:
            field_kwargs["le"] = le

        py_type: Any = str

        if raw_type in ("int", "integer"):
            py_type = int if default_val is not None and default_val != ... else Optional[int]
        elif raw_type in ("float", "double", "number"):
            py_type = float if default_val is not None and default_val != ... else Optional[float]
        elif raw_type in ("bool", "boolean"):
            py_type = bool if default_val is not None and default_val != ... else Optional[bool]
        elif raw_type in ("enum", "choice"):
            enum_values = field_def.get("enum_values", [])
            if enum_values and description:
                field_kwargs["description"] = f"{description} (Allowed values: {', '.join(enum_values)})"
            elif enum_values:
                field_kwargs["description"] = f"Allowed values: {', '.join(enum_values)}"
            py_type = str if default_val is not None and default_val != ... else Optional[str]
        elif raw_type in ("list_str", "list[str]"):
            py_type = List[str]
        else:
            py_type = str if default_val is not None and default_val != ... else Optional[str]

        # Build Field(...) definition
        if default_val == ...:
            field_tuple = (py_type, Field(..., **field_kwargs))
        elif hasattr(default_val, "__class__") and "Field" in default_val.__class__.__name__:
            field_tuple = (py_type, default_val)
        else:
            field_tuple = (py_type, Field(default=default_val, **field_kwargs))

        model_fields[field_name] = field_tuple

    dynamic_model = create_model(schema_name, __module__=__name__, **model_fields)
    globals()[schema_name] = dynamic_model
    return dynamic_model
