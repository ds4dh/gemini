from typing import Union, Any
from transformers import AutoTokenizer
from src.models.reasoning import ResolvedReasoning


def build_messages(
    sample: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Build Hugging Face chat messages for one extraction sample."""
    system_template: str = cfg["prompt_templates"]["system_template"]
    user_template: str = cfg["prompt_templates"]["user_template"]
    context_data: dict[str, str] = dict(cfg["context_data"])

    reasoning_enabled = (
        cfg.get("model", {})
        .get("reasoning", {})
        .get("enabled", "auto")
    )

    if reasoning_enabled is False and "context_data_nothinking" in cfg:
        context_data.update(cfg["context_data_nothinking"])

    system_prompt_content = system_template.format(**context_data)
    user_prompt_content = user_template.format(
        input_text=sample["input_text"]
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt_content.strip(),
            },
            {
                "role": "user",
                "content": user_prompt_content.strip(),
            },
        ]
    }



def build_prompt(
    sample: dict[str, Any],
    tokenizer: AutoTokenizer,
    reasoning: ResolvedReasoning,
    add_generation_prompt: bool = True,
) -> dict[str, str]:
    """Render one pre-built message list into a model-specific prompt."""
    prompt = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **reasoning.chat_template_kwargs,
    )

    return {"prompt": prompt}