import os
import math
import numpy as np
import torch
import xgrammar as xgr
from typing import Any
from enum import Enum, auto
from dataclasses import dataclass, field
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclass
class ThinkingBudgetRequestState:
    """
    A small helper class to store the state for each request.
    """
    max_thinking_tokens: int
    tokens_generated: int = 0
    stopped_thinking: bool = False
    think_end_sentence_ids: list[int] = field(default_factory=list)


class ThinkingBudgetProcessor(LogitsProcessor):
    """
    A vLLM processor that enforces a gradual budget on "thinking" tokens.
    Starting at 75% of the budget, it gradually increases the probability
    of ending the thinking process, reaching certainty at 100% budget.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        think_end_token = "</think>",
        think_end_sentence = "\nEnough thinking! Time for my final answer.\n"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer,
            # vllm_config.model_config.model,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        self.think_end_token_ids = self.tokenizer.encode(think_end_token, add_special_tokens=False)
        self.think_end_sentence_ids = self.tokenizer.encode(think_end_sentence, add_special_tokens=False)
        self.neg_inf = float("-inf")
        self.req_state: dict[int, ThinkingBudgetRequestState] = {}

    def update_state(
        self,
        batch_update: BatchUpdate | None,
    ) -> None:
        """
        Update the internal grammar state for all requests.
        """
        # Update request indexing only if there was a batch update
        if batch_update:

            # Remove old requests
            for index in batch_update.removed:
                self.req_state.pop(index, None)

            # Build a new grammar from the schema string, for each added request
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                if params and params.extra_args:
                    self._initialize_request_state(index, params.extra_args, prompt_tok_ids, output_tok_ids)

            # Align state indices from moved requests
            for adx, bdx, directionality in batch_update.moved:
                a_val = self.req_state.pop(adx, None)
                b_val = self.req_state.pop(bdx, None)
                if a_val is not None:
                    self.req_state[bdx] = a_val
                if directionality == MoveDirectionality.SWAP and b_val is not None:
                    self.req_state[adx] = b_val

        # Update request state for all active requests
        for index, state in self.req_state.items():
            self._update_request_state(index, state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:  # <--- MODIFIED (Logic fully rewritten)
        """
        Apply the gradual logits modification logic for the entire batch.
        """
        if not self.req_state:
            return logits

        for batch_idx, state in self.req_state.items():
            if state.stopped_thinking:
                continue

            # Already forcing the final sentence or token
            if state.think_end_sentence_ids:

                # Get the next token from the queue and force it
                next_token_id = state.think_end_sentence_ids.pop(0)
                logits[batch_idx, :] = self.neg_inf
                logits[batch_idx, next_token_id] = 0.0

                # If the queue is now empty, we are completely done
                if not state.think_end_sentence_ids:
                    state.stopped_thinking = True
                continue

            # Coming over budget and not yet forcing the sentence
            if state.tokens_generated > state.max_thinking_tokens:

                # Set up the queue of tokens to force: sentence + </think>
                state.think_end_sentence_ids = self.think_end_sentence_ids.copy()
                state.think_end_sentence_ids.extend(self.think_end_token_ids)

                # Force the first token in the queue
                next_token_id = state.think_end_sentence_ids.pop(0)
                logits[batch_idx, :] = self.neg_inf
                logits[batch_idx, next_token_id] = 0.0
                continue

            # Still thinking and under budget, increasing token count
            state.tokens_generated += 1

        return logits

    def is_argmax_invariant(self) -> bool:
        """
        Always false, since this processor changes biases specific tokens
        """
        return False

    def _initialize_request_state(
        self,
        index: int,
        init_params: dict[str, Any],
        prompt_tok_ids: list[int],
        output_tok_ids: list[int],
    ):
        """
        Logit-processor-specific function to initialize the state for a new request
        """
        max_thinking_tokens = init_params.get("max_thinking_tokens")
        if max_thinking_tokens is not None and isinstance(max_thinking_tokens, int):
            self.req_state[index] = ThinkingBudgetRequestState(max_thinking_tokens=max_thinking_tokens)

    @staticmethod
    def _update_request_state(
        index: int,
        state: ThinkingBudgetRequestState,
    ) -> None:
        """
        Logit-processor-specific function to update the state for an existing request
        """
        pass


class ParsingState(Enum):
    PRE_THINK = auto()
    POST_THINK = auto()
    ACTIVE_JSON = auto()


@dataclass
class JSONRequestState:
    """
    A helper class to store the JSON grammar state for each request.
    """
    grammar: xgr.CompiledGrammar
    matcher: xgr.GrammarMatcher
    output_tokens_ref: list[int]
    think_end_token_id: int
    post_think_token_ids: list[int]
    processed_len: int = 0
    parsing_state: ParsingState = ParsingState.PRE_THINK


class JSONParsingProcessor(LogitsProcessor):
    """
    A vLLM processor that enforces a JSON schema after thinking is complete.
    It waits for the end of reasoning and then activates, forcing all subsequent
    output to conform to the provided JSON schema.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ):
        # Load correct tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer,
            # vllm_config.model_config.model,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        # Create xgrammar tokenizer adapter
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.xgr_tokenizer = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        self.xgr_compiler = xgr.GrammarCompiler(self.xgr_tokenizer)

        # Crucial token ids, constants, and state initialization
        self.device = device
        self.compressed_vocab_size = math.ceil(self.vocab_size / 32)
        self.think_end_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.post_think_token_ids = [
            self.tokenizer.encode("\n", add_special_tokens=False)[0],
            self.tokenizer.encode("\n\n", add_special_tokens=False)[0],
        ]
        self.req_state: dict[int, JSONRequestState] = {}

    def update_state(
        self,
        batch_update: BatchUpdate | None,
    ) -> None:
        """
        Update the internal grammar state for all requests.
        """
        # Update request indexing only if there was a batch update
        if batch_update:

            # Remove old requests
            for index in batch_update.removed:
                self.req_state.pop(index, None)

            # Build a new grammar from the schema string, for each added request
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                if params and params.extra_args:
                    self._initialize_request_state(index, params.extra_args, prompt_tok_ids, output_tok_ids)

            # Align state indices from moved requests
            for adx, bdx, directionality in batch_update.moved:
                a_val = self.req_state.pop(adx, None)
                b_val = self.req_state.pop(bdx, None)
                if a_val is not None:
                    self.req_state[bdx] = a_val
                if directionality == MoveDirectionality.SWAP and b_val is not None:
                    self.req_state[adx] = b_val

        # Update request state for all active requests
        for index, state in self.req_state.items():
            self._update_request_state(index, state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the JSON grammar mask to the logits.
        """
        # No logit modification if there are no requests being tracked
        if not self.req_state:
            return logits

        # Find all active requests (those past the </think> token)
        batch_size = logits.shape[0]
        active_requests = [
            (batch_idx, state)
            for batch_idx, state in self.req_state.items()
            if state.parsing_state == ParsingState.ACTIVE_JSON and batch_idx < batch_size
        ]

        # If no requests are actively parsing JSON, do nothing.
        if not active_requests:
            return logits

        # Create a new 2D numpy bitmask for the batch (batch_size, compressed_vocab_size)
        step_bitmask_np = np.full(
            (batch_size, self.compressed_vocab_size),
            -1,  # this means "all tokens allowed" in int32 (all 1s)
            dtype=np.int32,
        )

        # Fill the bitmask only for the rows corresponding to active requests
        for batch_idx, state in active_requests:
            state.matcher.fill_next_token_bitmask(
                step_bitmask_np,  # the 2D array to fill
                index=batch_idx,  # the specific row to fill
            )

        # Rows for inactive requests (with -1 mask) are unaffected
        logits_original = logits.clone()
        bitmask_tensor = torch.tensor(step_bitmask_np, device=logits.device)
        xgr.apply_token_bitmask_inplace(
            logits=logits,
            bitmask=bitmask_tensor,
            vocab_size=self.vocab_size
        )

        # Ignore grammar for requests with all token ids rejected
        all_masked_rows = (logits == -float("inf")).all(dim=-1)
        if all_masked_rows.any():
            print(
                f"Warning: Grammar for request {batch_idx} resulted in all "
                f"tokens being masked. Broken grammar effect disabled."
            )
            return logits_original
            # logits[all_masked_rows, :] = -float("inf")
            # logits[all_masked_rows, self.tokenizer.eos_token_id] = 0.0

        return logits

    def is_argmax_invariant(self) -> bool:
        """
        Always false, as this processor zeros-out logits.
        """
        return False

    def _initialize_request_state(
        self,
        index: int,
        init_params: dict[str, Any],
        prompt_tok_ids: list[int],
        output_tok_ids: list[int],
    ) -> None:
        """
        Logit-processor-specific function to initialize the state for a new request
        """
        # Ensure the request has a JSON schema for output guiding
        schema_str = init_params.get("json_schema")
        if schema_str:
            try:

                # Use the compiler and json to create a grammar matcher
                compiled_grammar = self.xgr_compiler.compile_json_schema(
                    schema_str,
                    indent=None,
                    separators=None,
                    strict_mode=True,  # False,
                )
                matcher = xgr.GrammarMatcher(compiled_grammar)

                # Create a state entry for this request
                self.req_state[index] = JSONRequestState(
                    grammar=compiled_grammar,
                    matcher=matcher,
                    output_tokens_ref=output_tok_ids,
                    think_end_token_id=self.think_end_token_id,
                    post_think_token_ids=self.post_think_token_ids,
                )

            except Exception as e:
                raise ValueError(f"Could not build JSON grammar for req {index}. {e}")

        else:
            raise KeyError("No json_schema provided in vllm_xargs for output guiding.")

    @staticmethod
    def _update_request_state(
        index: int,
        state: JSONRequestState,
    ) -> None:
        """
        Logit-processor-specific function to update the state for an existing request
        """
        # Check if request state has changed since last update
        new_len = len(state.output_tokens_ref)
        if new_len == state.processed_len:
            return  # no new tokens were generated for this request

        # Process new tokens
        new_tokens = state.output_tokens_ref[state.processed_len:]            
        for token_id in new_tokens:

            # We are in "thinking" phase: check if we hit end-of-think token
            if state.parsing_state == ParsingState.PRE_THINK:
                if token_id == state.think_end_token_id:
                    state.parsing_state = ParsingState.POST_THINK  # no grammar activated yet

            # This case happens for the token_id immediately following </think>.
            elif state.parsing_state == ParsingState.POST_THINK:  # not "if"!

                # Activate JSON parsing now, regardless of what this token is
                state.parsing_state = ParsingState.ACTIVE_JSON
                state.matcher.reset()
                print(f"Request {index}: Activated JSON parsing after </think>.")

                # Logic lenient to reasoning-stop including a newline or not
                if token_id in state.post_think_token_ids:
                    print(f"Request {index}: Consumed optional newline.")
                else:
                    accepted = state.matcher.accept_token(token_id)
                    if not accepted:
                        print(f"Warning: Token {token_id} rejected by grammar for req {index}")

            # We are already actively parsing JSON
            elif state.parsing_state == ParsingState.ACTIVE_JSON:
                accepted = state.matcher.accept_token(token_id)
                if not accepted:
                    print(f"Warning: Token {token_id} rejected by grammar for req {index}")

        # Update the processed length
        state.processed_len = new_len
