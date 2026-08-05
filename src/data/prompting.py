from typing import Union
from transformers import AutoTokenizer


def build_messages(
    sample: dict[str, str],
    cfg: dict,
) -> dict[str, Union[list, str]]:
    """
    Take in a data sample to build HuggingFace's style messages

    Args:
        sample: dictionary containing the input text, expected as sample["input_text"]
        cfg: configuration dictionary containing prompt templates and context data
    """
    # Get prompt info from configuration
    system_template: str = cfg["prompt_templates"]["system_template"]
    user_template: str = cfg["prompt_templates"]["user_template"]
    context_data: dict[str, str] = cfg["context_data"]

    # Update contextual data if thinking mode is not enabled and alternative prompt context is provided
    if cfg.get("enable_thinking") is False and "context_data_no_thinking" in cfg:
        context_data_no_thinking = cfg["context_data_no_thinking"]
        for field in context_data_no_thinking:
            context_data[field] = context_data_no_thinking[field]

    # Populate system and user prompt templates
    system_prompt_content = system_template.format(**context_data)
    user_prompt_content = user_template.format(input_text=sample["input_text"])
    
    # Create huggingface-formatted message stream
    messages = [
        {"role": "system", "content": system_prompt_content.strip()},
        {"role": "user", "content": user_prompt_content.strip()},
    ]

    return {"messages": messages}


def build_prompt(
    sample: dict[str, str],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
) -> dict[str, str]:
    """
    Apply the chat template to a single sample
    """
    sample["prompt"] = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    return sample