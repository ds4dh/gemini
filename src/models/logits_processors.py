import json
import math
from typing import Any
from enum import Enum, auto
from dataclasses import dataclass
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

try:
    import xgrammar as xgr
except ImportError as exc:
    raise ImportError(
        "ThinkingJSONAdapterProcessor requires xgrammar."
    ) from exc

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class GenerationPhase(Enum):
    THINKING = auto()
    FINAL_JSON = auto()


def _ends_with_token_ids(token_ids: list[int], suffix: list[int]) -> bool:
    return (
        len(token_ids) >= len(suffix)
        and token_ids[-len(suffix):] == suffix
    )


def _parse_bool_or_int(value: Any, default: bool = True) -> bool:
    """
    vLLM extra_args may serialize booleans as either bool or integer 0/1.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    raise TypeError(
        "Expected a bool or integer 0/1, "
        f"received {value!r} ({type(value).__name__})."
    )


@dataclass
class ThinkingJSONRequestProcessor:
    """
    Per-request stateful custom logits processor.

    AdapterLogitsProcessor supplies this callable with the actual current
    output_ids list at every decoding step.
    """
    matcher: Any
    vocab_size: int
    compressed_vocab_size: int
    eos_token_id: int
    think_end_token_ids: list[int]
    enable_thinking: bool
    max_thinking_tokens: int | None
    tokenizer: PreTrainedTokenizer
    verbose_level: int = 1

    processed_len: int = 0
    thinking_tokens_generated: int = 0
    phase: GenerationPhase = GenerationPhase.THINKING
    force_started_at: int | None = None
    _generated_output_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.enable_thinking:
            self.phase = GenerationPhase.FINAL_JSON
            self.matcher.reset()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Called by AdapterLogitsProcessor once per decoding step.

        output_ids contains all tokens already emitted for this request.
        logits contains the next-token distribution for this request.
        """
        self._generated_output_ids = list(output_ids)
        self._consume_new_output_tokens(output_ids)

        if self.phase == GenerationPhase.THINKING:
            forced_token_id = self._next_forced_think_end_token(output_ids)

            if forced_token_id is not None:
                self._force_single_token(logits, forced_token_id)

            return logits

        # XGrammar has already accepted EOS / its configured stop token.
        # Do not request another grammar mask after terminal state.
        if self.matcher.is_terminated():
            self._force_single_token(logits, self.eos_token_id)
            return logits

        self._apply_json_grammar_mask(logits)
        return logits

    def _consume_new_output_tokens(
        self,
        output_ids: list[int],
    ) -> None:
        """Advance reasoning and JSON state for newly emitted output tokens."""
        if len(output_ids) <= self.processed_len:
            return

        new_tokens = output_ids[self.processed_len:]

        for token_id in new_tokens:
            if token_id < 0:
                self.processed_len += 1
                continue

            self.processed_len += 1

            if self.phase == GenerationPhase.THINKING:
                self._consume_thinking_token(output_ids)
                continue

            accepted = self.matcher.accept_token(token_id)

            if accepted:
                continue

            decoded_token = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            if self.verbose_level > 0:
                print(
                    "Warning: token emitted after reasoning end was rejected by "
                    "the JSON grammar. Resetting grammar state so the next token "
                    f"must begin a fresh JSON object. token_id={token_id}, "
                    f"token={decoded_token!r}, "
                    f"decoded_position={self.processed_len}"
                )

            # The token already exists in the raw output, so it cannot be removed.
            # Resetting prevents a rejected token from poisoning the matcher state.
            # The next constrained step must start a new valid JSON object.
            self.matcher.reset()

    def _consume_thinking_token(
        self,
        output_ids: list[int],
    ) -> None:
        """Count thought tokens and enter JSON phase after the configured marker."""
        if (
            self.think_end_token_ids
            and _ends_with_token_ids(
                output_ids[:self.processed_len],
                self.think_end_token_ids,
            )
        ):
            if self.verbose_level > 1:
                print(
                    "Custom processor detected thought end; "
                    "activating JSON grammar."
                )

            self.phase = GenerationPhase.FINAL_JSON
            self.matcher.reset()
            return

        self.thinking_tokens_generated += 1


    def _next_forced_think_end_token(
        self,
        output_ids: list[int],
    ) -> int | None:
        """
        Return the next configured thought-end token to force.
        """
        if self.max_thinking_tokens is None:
            return None

        if self.force_started_at is None:
            if self.thinking_tokens_generated < self.max_thinking_tokens:
                return None

            self.force_started_at = len(output_ids)
            if self.verbose_level > 1:
                print(
                    "Custom processor: thinking budget reached "
                    f"({self.max_thinking_tokens}); forcing </think>."
                )

        emitted_forced_tokens = len(output_ids) - self.force_started_at

        if emitted_forced_tokens >= len(self.think_end_token_ids):
            return None

        return self.think_end_token_ids[emitted_forced_tokens]

    @staticmethod
    def _force_single_token(
        logits: torch.Tensor,
        token_id: int,
    ) -> None:
        """
        Make token_id the only possible next sampled token.
        """
        original_logit = logits[token_id].item()
        logits.fill_(float("-inf"))
        logits[token_id] = original_logit

    def _apply_json_grammar_mask(
        self,
        logits: torch.Tensor,
    ) -> None:
        """Apply XGrammar's official next-token mask for the current JSON state."""
        bitmask = xgr.allocate_token_bitmask(
            batch_size=1,
            vocab_size=self.vocab_size,
        )

        self.matcher.fill_next_token_bitmask(
            bitmask,
            index=0,
        )

        xgr.apply_token_bitmask_inplace(
            logits=logits.unsqueeze(0),
            bitmask=bitmask.to(logits.device),
            vocab_size=self.vocab_size,
        )

        if torch.isneginf(logits).all():
            generated_text = self.tokenizer.decode(
                self._generated_output_ids,
                skip_special_tokens=False,
            )

            raise RuntimeError(
                "XGrammar left no valid token after applying the JSON-schema "
                f"mask. phase={self.phase.name}; "
                f"processed_len={self.processed_len}; "
                f"generated_text={generated_text!r}"
            )


class ThinkingJSONAdapterProcessor(AdapterLogitsProcessor):
    """
    vLLM adapter for a custom combined thinking-budget and JSON-grammar
    request-level processor.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        verbose_level: int = 1,
    ):
        super().__init__(vllm_config, device, is_pin_memory)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.compressed_vocab_size = math.ceil(self.vocab_size / 32)

        self.eos_token_id = self.tokenizer.eos_token_id
        if not isinstance(self.eos_token_id, int):
            raise TypeError(
                "Expected tokenizer.eos_token_id to be a single integer; "
                f"received {self.eos_token_id!r}."
            )

        self.xgr_tokenizer = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
        )
        self.xgr_compiler = xgr.GrammarCompiler(self.xgr_tokenizer)

        self.verbose_level = verbose_level
        if self.verbose_level > 0:
            print("ThinkingJSONAdapterProcessor initialized.")

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> ThinkingJSONRequestProcessor | None:
        """Create isolated JSON-grammar and reasoning-budget state per request."""
        extra_args = params.extra_args or {}

        schema = extra_args.get("jsonschema")
        if schema is None:
            return None

        if isinstance(schema, dict):
            schema = json.dumps(schema)

        if not isinstance(schema, str):
            raise TypeError(
                "jsonschema must be a JSON string or dictionary, "
                f"received {type(schema).__name__}."
            )

        enable_thinking = _parse_bool_or_int(
            extra_args.get("enable_thinking", True),
            default=True,
        )
        max_thinking_tokens = extra_args.get("max_thinking_tokens")
        think_end_marker = extra_args.get("think_end_marker")
        think_end_token_ids: list[int] = []

        if enable_thinking:
            if not isinstance(think_end_marker, str) or not think_end_marker:
                raise ValueError(
                    "ThinkingJSONAdapterProcessor requires think_end_marker when "
                    "enable_thinking=True."
                )

            think_end_token_ids = self.tokenizer.encode(
                think_end_marker,
                add_special_tokens=False,
            )

            if not think_end_token_ids:
                raise ValueError(
                    f"Tokenizer cannot encode think_end_marker={think_end_marker!r}."
                )

        if max_thinking_tokens is not None:
            if (
                not isinstance(max_thinking_tokens, int)
                or isinstance(max_thinking_tokens, bool)
                or max_thinking_tokens < 0
            ):
                raise TypeError(
                    "max_thinking_tokens must be a non-negative integer or None."
                )

            if not enable_thinking:
                raise ValueError(
                    "max_thinking_tokens requires enable_thinking=True."
                )

        try:
            grammar = self.xgr_compiler.compile_json_schema(
                schema,
                indent=None,
                separators=None,
                strict_mode=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not compile JSON Schema with XGrammar: {exc}"
            ) from exc

        if self.verbose_level > 1:
            print(
                "Creating ThinkingJSONRequestProcessor: "
                f"enable_thinking={enable_thinking}, "
                f"max_thinking_tokens={max_thinking_tokens}, "
                f"think_end_marker={think_end_marker!r}, "
                f"think_end_token_count={len(think_end_token_ids)}"
            )

        return ThinkingJSONRequestProcessor(
            matcher=xgr.GrammarMatcher(grammar),
            vocab_size=self.vocab_size,
            compressed_vocab_size=self.compressed_vocab_size,
            eos_token_id=self.eos_token_id,
            think_end_token_ids=think_end_token_ids,
            enable_thinking=enable_thinking,
            max_thinking_tokens=max_thinking_tokens,
            tokenizer=self.tokenizer,
            verbose_level=self.verbose_level,
        )

    def is_argmax_invariant(self) -> bool:
        return False