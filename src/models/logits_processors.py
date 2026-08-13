import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

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


def _ends_with(token_ids: list[int], suffix: list[int]) -> bool:
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
    verbose_level: int = 1

    processed_len: int = 0
    thinking_tokens_generated: int = 0
    phase: GenerationPhase = GenerationPhase.THINKING
    force_started_at: int | None = None

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

    def _consume_new_output_tokens(self, output_ids: list[int]) -> None:
        """
        Advance thinking or JSON state using emitted tokens not previously seen.
        """
        if len(output_ids) <= self.processed_len:
            return

        new_tokens = output_ids[self.processed_len:]
        self.processed_len = len(output_ids)

        for token_id in new_tokens:
            if token_id < 0:
                continue

            if self.phase == GenerationPhase.THINKING:
                self._consume_thinking_token(output_ids, token_id)

            elif self.phase == GenerationPhase.FINAL_JSON:
                accepted = self.matcher.accept_token(token_id)

                if not accepted and self.verbose_level > 0:
                    print(
                        "Warning: generated token was rejected by the "
                        f"JSON grammar: token_id={token_id}, "
                        f"decoded_position={self.processed_len}"
                    )

    def _consume_thinking_token(
        self,
        output_ids: list[int],
        token_id: int,
    ) -> None:
        """
        Count reasoning tokens and detect a natural or forced </think>.
        """
        if _ends_with(output_ids[:self.processed_len], self.think_end_token_ids):
            if self.verbose_level > 1:
                print(
                    "Custom processor: detected </think>; "
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
        Return the next </think> token to force, or None when unconstrained.

        Supports both one-token and multi-token </think> representations.
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

    def _apply_json_grammar_mask(self, logits: torch.Tensor) -> None:
        """
        Generate and apply an XGrammar next-token mask for this request.
        """
        bitmask_np = np.full(
            (1, self.compressed_vocab_size),
            -1,
            dtype=np.int32,
        )

        self.matcher.fill_next_token_bitmask(
            bitmask_np,
            index=0,
        )

        logits_2d = logits.unsqueeze(0)

        xgr.apply_token_bitmask_inplace(
            logits=logits_2d,
            bitmask=torch.from_numpy(bitmask_np).to(logits.device),
            vocab_size=self.vocab_size,
        )

        if torch.isneginf(logits).all() and self.verbose_level > 0:
            print(
                "Warning: XGrammar masked every token for a request; "
                "permitting EOS only."
            )
            logits.fill_(float("-inf"))
            logits[self.eos_token_id] = 0.0


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

        self.think_end_token_ids = self.tokenizer.encode(
            "</think>",
            add_special_tokens=False,
        )
        if not self.think_end_token_ids:
            raise ValueError("Tokenizer cannot encode '</think>'.")

        self.xgr_tokenizer = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
            stop_token_ids=[self.eos_token_id],
        )
        self.xgr_compiler = xgr.GrammarCompiler(self.xgr_tokenizer)

        self.verbose_level = verbose_level
        if self.verbose_level > 0:
            print(
                "ThinkingJSONAdapterProcessor initialized: "
                f"vocab_size={self.vocab_size}, "
                f"think_end_token_ids={self.think_end_token_ids}"
            )

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> ThinkingJSONRequestProcessor | None:
        """
        Create one isolated request-level processor from vLLM request metadata.
        """
        extra_args = params.extra_args or {}

        schema = extra_args.get("json_schema")
        if schema is None:
            return None

        if isinstance(schema, dict):
            schema = json.dumps(schema)

        if not isinstance(schema, str):
            raise TypeError(
                "json_schema must be a JSON string or dict; "
                f"received {type(schema).__name__}."
            )

        enable_thinking = _parse_bool_or_int(
            extra_args.get("enable_thinking", True),
            default=True,
        )

        max_thinking_tokens = extra_args.get("max_thinking_tokens")

        if max_thinking_tokens is not None:
            if (
                not isinstance(max_thinking_tokens, int)
                or max_thinking_tokens < 0
            ):
                raise TypeError(
                    "max_thinking_tokens must be a non-negative integer "
                    f"or None; received {max_thinking_tokens!r}."
                )

            if not enable_thinking:
                raise ValueError(
                    "max_thinking_tokens was supplied while "
                    "enable_thinking=False."
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

        if self.verbose_level > 2:
            print(
                "Creating custom per-request processor: "
                f"enable_thinking={enable_thinking}, "
                f"max_thinking_tokens={max_thinking_tokens}"
            )

        return ThinkingJSONRequestProcessor(
            matcher=xgr.GrammarMatcher(grammar),
            vocab_size=self.vocab_size,
            compressed_vocab_size=self.compressed_vocab_size,
            eos_token_id=self.eos_token_id,
            think_end_token_ids=self.think_end_token_ids,
            enable_thinking=enable_thinking,
            max_thinking_tokens=max_thinking_tokens,
        )

    def is_argmax_invariant(self) -> bool:
        return False