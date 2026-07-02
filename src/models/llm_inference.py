import json
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import AutoModelForCausalLM
from datasets import Dataset
# from llama_cpp import Llama
from vllm import LLM, SamplingParams, RequestOutput
from openai import OpenAI, AsyncOpenAI, InternalServerError
from tqdm import tqdm
from tqdm.asyncio import tqdm as anext
from typing import Any, Type
from functools import partial
from pydantic import BaseModel
from vllm.sampling_params import StructuredOutputsParams
from src.data.prompting import build_prompt
from src.data.output_guiding import extract_structured_output, SCHEMA_REGISTRY


def _infer_vllm(
    model: LLM,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference with vLLM directly (faster than querying vLLM-serve)
    """
    # Define if structured output is used or not during inference
    structured_params = None
    if output_schema_model is not None:
        json_schema = output_schema_model.model_json_schema()
        structured_params = StructuredOutputsParams(json=json_schema)

    # Define thinking budget if required
    extra_args = {}
    if max_thinking_tokens is not None:
        extra_args["max_thinking_tokens"] = max_thinking_tokens

    # Build sampling parameters
    sampling_params = SamplingParams(
        n=n_inference_repeats,  # several completions per prompt
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        structured_outputs=structured_params,
        extra_args=extra_args if extra_args else None,
    )

    # Build prompts using model tokenizer by mapping dataset messages
    tokenizer_fn = partial(
        build_prompt,
        tokenizer=model.get_tokenizer(),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    dataset = dataset.map(tokenizer_fn, desc="Building prompts for vLLM")

    # Run vLLM model on all dataset's prompts (single, efficient batch call)
    outputs: list[RequestOutput] = model.generate(dataset["prompt"], sampling_params=sampling_params)
    
    # Each request_output in outputs corresponds to a prompt and contains n completions
    output_texts = [
        [completion.text.strip() for completion in request_output.outputs]
        for request_output in outputs
    ]

    return output_texts


def _extract_outputs_vllm(choices: list) -> list[str]:
    """
    Extract content from chat completion choices, falling back to reasoning content
    """
    outputs = []
    for choice in choices:
        content = getattr(choice.message, "content", None)
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if content is None: content = reasoning_content
        outputs.append((content or "").strip())

    return outputs


def _setup_inference_output(
    output_schema_model: Type[BaseModel] | None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
) -> tuple[dict | None, dict]:
    """
    Define how output is influenced during inference
    """
    # Structured output guiding
    # THIS WAS REMOVED FOR A LOGIT PROCESSOR!
    response_format = None
    # if output_schema_model is not None:
    #     response_format = {
    #         "type": "json_schema",
    #         "json_schema": {
    #             "name": output_schema_model.__name__,
    #             "schema": output_schema_model.model_json_schema(),
    #         },
    #     }

    # Arguments for logit processors
    extra_body = {}
    if enable_thinking == False:  # only for false! if True: should stay undefined
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    vllm_xargs = {}
    if max_thinking_tokens is not None:
        vllm_xargs["max_thinking_tokens"] = max_thinking_tokens
    if output_schema_model is not None:
        schema_dict = output_schema_model.model_json_schema()
        schema_str = json.dumps(schema_dict)
        safe_schema_str = schema_str.replace("'", "\\u0027")
        vllm_xargs["json_schema"] = safe_schema_str
    if vllm_xargs:
        extra_body["vllm_xargs"] = vllm_xargs

    return response_format, extra_body


def _infer_vllm_serve(
    model: OpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference by querying the vLLM server using the chat completion API
    """
    # Initialization
    client = model
    model_name = client.models.list().data[0].id
    response_format, extra_body = _setup_inference_output(output_schema_model, enable_thinking, max_thinking_tokens)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(max_attempt_number=5),
        retry=retry_if_exception_type((InternalServerError, ConnectionError)),
        reraise=True,
    )
    def generate_vllm_outputs(messages: list[dict[str, str]]):
        """Single-shot API call function"""
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n_inference_repeats,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            extra_body=extra_body,
        )
        return _extract_outputs_vllm(chat_completion.choices)

    # Query the server for each task separately
    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Querying vLLM server"):
        all_outputs.append(generate_vllm_outputs(messages))
        time.sleep(1)  # avoid overwhelming the server (useful?)

    return all_outputs


async def _infer_vllm_serve_async(
    model: AsyncOpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    max_concurrent_requests: int = 64,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference by querying the vLLM server asynchronously
    """
    # Re-initialize the client inside the current event loop to prevent 'Event loop is closed' errors
    client = AsyncOpenAI(base_url=str(model.base_url), api_key=model.api_key, timeout=model.timeout)  # client = model
    model_name = (await client.models.list()).data[0].id
    semaphore = asyncio.Semaphore(max_concurrent_requests)  # to avoid overload
    response_format, extra_body = _setup_inference_output(output_schema_model, enable_thinking, max_thinking_tokens)
    max_context_length = kwargs.get("max_context_length") or 20_000
    safe_prompt_length = max(1, max_context_length - max_new_tokens - 100)
    extra_body["truncate_prompt_tokens"] = safe_prompt_length

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((InternalServerError, ConnectionError)),
        reraise=True,
    )
    async def generate_vllm_outputs(messages: list[dict[str, str]]):
        """Single-shot async API call function, with semaphore control"""
        async with semaphore:
            chat_completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                n=n_inference_repeats,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
                extra_body=extra_body,
            )
            return _extract_outputs_vllm(chat_completion.choices)

    # Query the server for all tasks concurrently
    tasks = [generate_vllm_outputs(messages) for messages in dataset["messages"]]
    all_outputs = await anext.gather(*tasks, desc="Querying vLLM server (async)")

    return all_outputs


def _infer_llama_cpp(
    model,  # Llama,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    *args, **kwargs,
) -> list[list[str]]:
    """
    Runs inference with llama-cpp-python
    """
    # Build response format
    # TODO: CHECK IF THAT WORKS (NEVER TESTED SINCE I AM NOT USING LLAMA-CPP ANYMORE)
    _, response_format = _setup_inference_output(output_schema_model)

    # Run llama-cpp model on the dataset
    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Generating inferences (llama-cpp)"):
        prompt_outputs = []

        # Loop n_inference_repeats times for each message
        for _ in range(n_inference_repeats):
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )

            # Each response has one choice, so we get it at index 0
            content = response["choices"][0]["message"]["content"]
            prompt_outputs.append(content.strip())
            
        all_outputs.append(prompt_outputs)

    return all_outputs


def process_samples(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    inference_backend: str,
    n_inference_repeats: int,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    output_schema_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    *args, **kwargs,
) -> Dataset:
    """
    Run inference on dataset samples using vLLM, llama-cpp, or HuggingFace backends.
    """
    # Retrieve output schema if requested (to guide LLM inference)
    output_schema_model = None
    if output_schema_name is not None:
        output_schema_model = SCHEMA_REGISTRY.get(output_schema_name)
        if output_schema_model is None:
            raise ValueError(
                f"Schema '{output_schema_name}' not found in registry. "
                f"Available schemas are: {list(SCHEMA_REGISTRY.keys())}"
            )

    # Define arguments for model inference 
    infer_args = {
        "model": model,
        "dataset": dataset,
        "n_inference_repeats": n_inference_repeats,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "output_schema_model": output_schema_model,
        "enable_thinking": enable_thinking,
        "max_thinking_tokens": max_thinking_tokens,
        **kwargs,
    }

    # Run inference using the correct backend
    match inference_backend:
        case "vllm": output_texts = _infer_vllm(**infer_args)
        case "vllm-serve": output_texts = _infer_vllm_serve(**infer_args)
        case "llama-cpp": output_texts = _infer_llama_cpp(**infer_args)
        case "vllm-serve-async": output_texts = asyncio.run(_infer_vllm_serve_async(**infer_args))
        case _: raise ValueError(f"Unknown inference backend: {inference_backend}")

    # Extract data using the model and required inference backend
    transposed_output_texts = list(zip(*output_texts))
    for i, model_outputs in enumerate(transposed_output_texts):

        # Add raw output text, adding the model index
        column_name = f"output_text_{i:03d}"
        dataset = dataset.add_column(name=column_name, column=model_outputs)

        # Define function to add structured output, keeping the model index in the output column
        mapping_fn = partial(
            _map_and_structure_output,
            output_schema_model=output_schema_model,
            col_to_structure=column_name,
            inference_idx=i,
        )
        dataset = dataset.map(mapping_fn, desc="Extracting model predictions")

    print("All LLM outputs were parsed.")
    return dataset


def _map_and_structure_output(
    sample: dict[str, Any],
    output_schema_model: Type[BaseModel],
    col_to_structure: str,
    inference_idx: int,
) -> dict[str, Any]:
    """
    Extracts structured data from a single sample's text output and add the index
    of the inference that generated that output
    """
    structured_dict = extract_structured_output(
        sample=sample,
        output_schema_model=output_schema_model,
        col_to_structure=col_to_structure,
    )

    # Add the inference index to make column names unique
    return {f"{k}_{inference_idx:03d}": v for k, v in structured_dict.items()}
