import asyncio
import json
import math
import time
from functools import partial
from typing import Any, Type

from datasets import Dataset
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from openai import AsyncOpenAI, InternalServerError, OpenAI
from src.data.output_guiding import extract_structured_output, resolve_schema_model
from src.data.prompting import build_prompt


def _infer_vllm(
    model: LLM,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    use_output_guide: bool = False,
    *args: Any,
    **kwargs: Any,
) -> list[list[str]]:
    """Run direct vLLM inference with optional native JSON guidance."""
    if max_thinking_tokens is not None:
        raise ValueError(
            "max_thinking_tokens is supported only with "
            "inference_backend='vllm-serve' or 'vllm-serve-async'. "
            "Direct vLLM does not register ThinkingJSONAdapterProcessor."
        )

    structured_params = None
    if use_output_guide:
        if output_schema_model is None:
            raise ValueError(
                "use_output_guide=True requires a resolved Pydantic output schema."
            )
        structured_params = StructuredOutputsParams(
            json=output_schema_model.model_json_schema()
        )

    print(
        "Direct vLLM output-guidance configuration: "
        f"use_output_guide={use_output_guide}, "
        f"native_structured_output={structured_params is not None}"
    )

    sampling_params = SamplingParams(
        n=n_inference_repeats,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        structured_outputs=structured_params,
    )

    tokenizer_fn = partial(
        build_prompt,
        tokenizer=model.get_tokenizer(),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    dataset = dataset.map(tokenizer_fn, desc="Building prompts for vLLM")

    outputs: list[RequestOutput] = model.generate(
        dataset["prompt"],
        sampling_params=sampling_params,
    )
    return [
        [completion.text.strip() for completion in request_output.outputs]
        for request_output in outputs
    ]


def _extract_outputs_vllm(choices: list[Any]) -> list[str]:
    """Extract final content, falling back to reasoning content when needed."""
    outputs = []
    for choice in choices:
        content = getattr(choice.message, "content", None)
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        outputs.append((content if content is not None else reasoning_content or "").strip())
    return outputs


def _setup_inference_output(
    output_schema_model: Type[BaseModel] | None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    use_output_guide: bool = True,
) -> tuple[None, dict[str, Any]]:
    """Build server request options for ThinkingJSONAdapterProcessor.

    The server-side adapter receives request-specific JSON-schema and bounded-
    thinking settings through ``vllm_xargs``. The chat template receives the
    model-level thinking toggle through ``chat_template_kwargs``.
    """
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {
            "enable_thinking": bool(enable_thinking),
        }
    }
    vllm_xargs: dict[str, Any] = {}

    if max_thinking_tokens is not None:
        if not enable_thinking:
            raise ValueError("max_thinking_tokens requires enable_thinking=True.")
        vllm_xargs["max_thinking_tokens"] = max_thinking_tokens

    if use_output_guide:
        if output_schema_model is None:
            raise ValueError(
                "use_output_guide=True requires a resolved output schema."
            )
        vllm_xargs["json_schema"] = json.dumps(
            output_schema_model.model_json_schema()
        )
        vllm_xargs["enable_thinking"] = enable_thinking

    if vllm_xargs:
        extra_body["vllm_xargs"] = vllm_xargs

    return None, extra_body


def _server_extra_body(
    base_extra_body: dict[str, Any],
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    max_context_length: int | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Add vLLM-specific request sampling options without mutating shared input."""
    extra_body = dict(base_extra_body)
    extra_body.update(
        {
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
        }
    )

    if max_context_length is not None and max_new_tokens is not None:
        extra_body["truncate_prompt_tokens"] = max(
            1,
            max_context_length - max_new_tokens - 100,
        )

    return extra_body


def _infer_vllm_serve(
    model: OpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    use_output_guide: bool = False,
    *args: Any,
    **kwargs: Any,
) -> list[list[str]]:
    """Run synchronous inference through the vLLM chat-completions server."""
    client = model
    model_name = client.models.list().data[0].id
    response_format, base_extra_body = _setup_inference_output(
        output_schema_model=output_schema_model,
        enable_thinking=enable_thinking,
        max_thinking_tokens=max_thinking_tokens,
        use_output_guide=use_output_guide,
    )
    extra_body = _server_extra_body(
        base_extra_body=base_extra_body,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        max_context_length=kwargs.get("max_context_length"),
        max_new_tokens=max_new_tokens,
    )

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((InternalServerError, ConnectionError)),
        reraise=True,
    )
    def generate_vllm_outputs(messages: list[dict[str, str]]) -> list[str]:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=n_inference_repeats,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            response_format=response_format,
            extra_body=extra_body,
        )
        return _extract_outputs_vllm(chat_completion.choices)

    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Querying vLLM server"):
        all_outputs.append(generate_vllm_outputs(messages))
        time.sleep(1)

    return all_outputs


async def _infer_vllm_serve_async(
    model: AsyncOpenAI,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    use_output_guide: bool = False,
    max_concurrent_requests: int = 64,
    *args: Any,
    **kwargs: Any,
) -> list[list[str]]:
    """Run bounded-concurrency inference through the vLLM chat server."""
    client = AsyncOpenAI(
        base_url=str(model.base_url),
        api_key=model.api_key,
        timeout=model.timeout,
    )

    try:
        model_name = (await client.models.list()).data[0].id
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        response_format, base_extra_body = _setup_inference_output(
            output_schema_model=output_schema_model,
            enable_thinking=enable_thinking,
            max_thinking_tokens=max_thinking_tokens,
            use_output_guide=use_output_guide,
        )
        extra_body = _server_extra_body(
            base_extra_body=base_extra_body,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            max_context_length=kwargs.get("max_context_length"),
            max_new_tokens=max_new_tokens,
        )

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=16),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((InternalServerError, ConnectionError)),
            reraise=True,
        )
        async def generate_vllm_outputs(
            messages: list[dict[str, str]],
        ) -> list[str]:
            async with semaphore:
                chat_completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    n=n_inference_repeats,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                    extra_body=extra_body,
                )
            return _extract_outputs_vllm(chat_completion.choices)

        tasks = [
            generate_vllm_outputs(messages)
            for messages in dataset["messages"]
        ]
        return await tqdm_asyncio.gather(
            *tasks,
            desc="Querying vLLM server (async)",
        )
    finally:
        await client.close()


def _infer_llama_cpp(
    model: Any,
    dataset: Dataset,
    n_inference_repeats: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_schema_model: Type[BaseModel] | None = None,
    use_output_guide: bool = False,
    *args: Any,
    **kwargs: Any,
) -> list[list[str]]:
    """Run llama-cpp-python inference; this backend remains minimally supported."""
    response_format = (
        {"type": "json_object"}
        if use_output_guide and output_schema_model is not None
        else None
    )

    all_outputs = []
    for messages in tqdm(dataset["messages"], desc="Generating inferences (llama-cpp)"):
        prompt_outputs = []
        for _ in range(n_inference_repeats):
            request_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if response_format is not None:
                request_kwargs["response_format"] = response_format

            response = model.create_chat_completion(**request_kwargs)
            content = response["choices"][0]["message"]["content"]
            prompt_outputs.append((content or "").strip())
        all_outputs.append(prompt_outputs)

    return all_outputs


def _validate_outputs(
    output_texts: list[list[str]],
    expected_samples: int,
    expected_repeats: int,
) -> None:
    """Fail explicitly instead of silently truncating outputs through zip()."""
    if len(output_texts) != expected_samples:
        raise RuntimeError(
            f"Expected outputs for {expected_samples} samples, "
            f"but received {len(output_texts)}."
        )

    actual_counts = [len(outputs) for outputs in output_texts]
    if any(count != expected_repeats for count in actual_counts):
        raise RuntimeError(
            "Some samples returned an unexpected number of completions. "
            f"Expected {expected_repeats}; got {actual_counts}."
        )


def process_samples(
    model: Any,
    dataset: Dataset,
    inference_backend: str,
    n_inference_repeats: int,
    enable_thinking: bool = True,
    max_thinking_tokens: int | None = None,
    output_schema_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> Dataset:
    """Run inference and attach raw and schema-validated outputs to a dataset."""
    runtime_kwargs = dict(kwargs)
    use_output_guide = bool(runtime_kwargs.pop("use_output_guide", False))
    schema_arg = (
        runtime_kwargs.get("schema")
        or runtime_kwargs.get("schema_config")
        or output_schema_name
    )
    output_schema_model = resolve_schema_model(schema_arg)

    infer_args = {
        "model": model,
        "dataset": dataset,
        "n_inference_repeats": n_inference_repeats,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "output_schema_model": output_schema_model,
        "enable_thinking": enable_thinking,
        "max_thinking_tokens": max_thinking_tokens,
        "use_output_guide": use_output_guide,
        **runtime_kwargs,
    }

    match inference_backend:
        case "vllm":
            output_texts = _infer_vllm(**infer_args)
        case "vllm-serve":
            output_texts = _infer_vllm_serve(**infer_args)
        case "vllm-serve-async":
            output_texts = asyncio.run(_infer_vllm_serve_async(**infer_args))
        case "llama-cpp":
            output_texts = _infer_llama_cpp(**infer_args)
        case "mock":
            output_texts = _infer_mock(**infer_args)
        case _:
            raise ValueError(f"Unknown inference backend: {inference_backend}")

    _validate_outputs(
        output_texts=output_texts,
        expected_samples=len(dataset),
        expected_repeats=n_inference_repeats,
    )

    for inference_idx, model_outputs in enumerate(zip(*output_texts)):
        column_name = f"output_text_{inference_idx:03d}"
        dataset = dataset.add_column(name=column_name, column=model_outputs)
        mapping_fn = partial(
            _map_and_structure_output,
            output_schema_model=output_schema_model,
            col_to_structure=column_name,
            inference_idx=inference_idx,
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
    """Parse one raw model output and suffix structured fields by repeat index."""
    structured_dict = extract_structured_output(
        sample=sample,
        output_schema_model=output_schema_model,
        col_to_structure=col_to_structure,
    )
    return {f"{key}_{inference_idx:03d}": value for key, value in structured_dict.items()}


def _infer_mock(
    model: Any,
    dataset: Dataset,
    n_inference_repeats: int,
    output_schema_model: Type[BaseModel] | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[list[str]]:
    """Simulate schema-compatible outputs for offline pipeline verification."""
    all_outputs = []
    for sample in dataset:
        mock_dict: dict[str, Any] = {}
        if output_schema_model is not None:
            for field_name in output_schema_model.model_fields:
                ground_truth_key = f"ground_truth_{field_name}"
                ground_truth_value = sample.get(ground_truth_key, sample.get(field_name))

                if ground_truth_value is not None and not (
                    isinstance(ground_truth_value, float)
                    and math.isnan(ground_truth_value)
                ):
                    mock_dict[field_name] = ground_truth_value
                elif field_name.lower() == "mrs":
                    mock_dict[field_name] = 0
                elif "smoke" in field_name.lower() or "smoking" in field_name.lower():
                    mock_dict[field_name] = "Non-smoker"
                elif "aneurysm" in field_name.lower() or "size" in field_name.lower():
                    mock_dict[field_name] = None
                elif "hyper" in field_name.lower():
                    mock_dict[field_name] = "No"
                elif "age" in field_name.lower():
                    mock_dict[field_name] = 60
                elif "location" in field_name.lower() or "lesion" in field_name.lower():
                    mock_dict[field_name] = "None"
                else:
                    mock_dict[field_name] = "Sample"

        mock_json = json.dumps(mock_dict, indent=2)
        all_outputs.append([mock_json for _ in range(n_inference_repeats)])

    return all_outputs
