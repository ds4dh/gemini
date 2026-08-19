from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

ReasoningEffort = Literal["auto", "off", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class ReasoningCapabilities:
    supports_thinking_toggle: bool = False
    supports_template_effort: bool = False
    supported_efforts: tuple[str, ...] = ()
    supports_preserve_thinking: bool = False
    think_end_marker: str | None = None


@dataclass(frozen=True)
class ResolvedReasoning:
    enable_thinking: bool
    reasoning_effort: str | None
    preserve_thinking: bool
    think_end_marker: str | None
    chat_template_kwargs: dict[str, Any]


def _chat_template_text(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), "")
    return str(template)


def detect_reasoning_capabilities(tokenizer) -> ReasoningCapabilities:
    template = _chat_template_text(tokenizer)
    effort_candidates = ("low", "medium", "high", "xhigh")
    supported_efforts = ()
    if "reasoning_effort" in template:
        supported_efforts = tuple(
            value for value in effort_candidates
            if f"'{value}'" in template or f'"{value}"' in template
        )
    return ReasoningCapabilities(
        supports_thinking_toggle="enable_thinking" in template,
        supports_template_effort=bool(supported_efforts),
        supported_efforts=supported_efforts,
        supports_preserve_thinking="preserve_thinking" in template,
        think_end_marker="</think>" if "</think>" in template else None,
    )


def normalize_reasoning_effort(
    requested: ReasoningEffort | None,
    supported_efforts: tuple[str, ...],
) -> str | None:
    if requested in (None, "auto", "off"):
        return None
    aliases = {
        "high": ("high", "xhigh"),
        "xhigh": ("xhigh", "high"),
    }
    for candidate in aliases.get(requested, (requested,)):
        if candidate in supported_efforts:
            return candidate
    return None


def resolve_reasoning(
    tokenizer,
    enabled: bool | str = "auto",
    effort: ReasoningEffort | None = "auto",
    preserve_thinking: bool = False,
) -> ResolvedReasoning:
    """Resolves reasoning structure given the tokenizer and run configuration"""
    capabilities = detect_reasoning_capabilities(tokenizer)
    enable_thinking = (
        capabilities.supports_thinking_toggle
        if enabled == "auto"
        else bool(enabled) and capabilities.supports_thinking_toggle
    )

    reasoning_effort = None
    if enable_thinking and capabilities.supports_template_effort:
        reasoning_effort = normalize_reasoning_effort(
            effort, capabilities.supported_efforts
        )

    chat_template_kwargs: dict[str, Any] = {}
    if capabilities.supports_thinking_toggle:
        chat_template_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if capabilities.supports_preserve_thinking:
        chat_template_kwargs["preserve_thinking"] = preserve_thinking

    return ResolvedReasoning(
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
        think_end_marker=(capabilities.think_end_marker if enable_thinking else None),
        chat_template_kwargs=chat_template_kwargs,
    )
