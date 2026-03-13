"""
Common helper functions shared by the targeted feature explainer notebooks
(Gemma, Llama, Qwen). Import from explainer_utils or explainer_utils.common.
"""

from __future__ import annotations

import csv
import os
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import orjson
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Default dataset cache base; notebooks can override via load_tokenized_data_with_cache(..., data_files=...).
DEFAULT_DATASET_CACHE_DIR = (
    "/scratch/jk87/sg9944/hugging_face_cache_data/monology___pile-uncopyrighted/"
    "default/0.0.0/3be90335b66f24456a5d6659d9c8d208c0357119/"
)
DEFAULT_PILE_DATA_FILES = {
    "train": [
        "/scratch/jk87/sg9944/hugging_face_cache_data/monology___pile-uncopyrighted/"
        "snapshots/3be90335b66f24456a5d6659d9c8d208c0357119/train/00.jsonl.zst"
    ],
    "test": (
        "/scratch/jk87/sg9944/hugging_face_cache_data/monology___pile-uncopyrighted/"
        "snapshots/3be90335b66f24456a5d6659d9c8d208c0357119/test.jsonl.zst"
    ),
}


def get_target_layers(
    features_path: str,
    layer_filter: list[int] | None = None,
) -> tuple[list[str], dict[str, list[int]]]:
    """
    Parse layer-featureid pairs from the features CSV,
    optionally filtering for specific layers.

    Args:
        features_path: Path to CSV with at least a "Source_Id" column (format "layerNum_featId").
        layer_filter: If provided, only include layers with these indices. Example: [5, 6, 7].

    Returns:
        (TARGET_HOOKPOINTS, TARGET_FEATURES)
    """
    TARGET_HOOKPOINTS: list[str] = []
    TARGET_FEATURES: dict[str, list[int]] = {}

    with open(features_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_id = row["Source_Id"]
            try:
                layer_num_str, feat_id = source_id.split("_")
                layer_num = int(layer_num_str)
                if (layer_filter is not None) and (layer_num not in layer_filter):
                    continue
                hookpoint = f"layers.{layer_num}"
                feat_id = int(feat_id)
                if hookpoint not in TARGET_FEATURES:
                    TARGET_FEATURES[hookpoint] = []
                    TARGET_HOOKPOINTS.append(hookpoint)
                TARGET_FEATURES[hookpoint].append(feat_id)
            except Exception as e:
                raise ValueError(f"Malformed source_id in row: {row}") from e
    return TARGET_HOOKPOINTS, TARGET_FEATURES


def load_artifacts(run_cfg, cache_dir: str | None = None):
    """
    Load model and sparse coders from run_cfg.
    Returns (hookpoints, hookpoint_to_sparse_encode, model, transcode).
    """
    from delphi.sparse_coders import load_hooks_sparse_coders

    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        torch_dtype=dtype,
        device_map="cuda",
        local_files_only=True,
        token=run_cfg.hf_token,
        cache_dir=cache_dir or "/g/data4/sz65/sg9944/hugging_face_cache/",
    )

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """Filter hookpoints to those not already present in results_path (unless overwrite)."""
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    in_results_path = [x.name for x in results_path.glob("*")]
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant


def load_tokenized_data_with_cache(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    cache_dir: str | None = None,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
    convert_to_tensor_chunk_size: int = 2**18,
    data_files: dict | None = None,
) -> Tensor:
    """Load and tokenize dataset; use pre-cached data_files when provided."""
    from datasets import load_dataset
    from sparsify.data import chunk_and_tokenize

    data_files = data_files or DEFAULT_PILE_DATA_FILES
    data = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=cache_dir,
    )
    data = data.shuffle(seed)
    tokens_ds = chunk_and_tokenize(
        data,
        tokenizer,
        max_seq_len=ctx_len,
        text_key=column_name,
    )["train"]

    tokens = tokens_ds["input_ids"]

    try:
        from datasets import Column

        if isinstance(tokens, Column):
            from datasets.table import table_iter

            tokens = torch.cat(
                [
                    torch.from_numpy(
                        np.stack(table_chunk["input_ids"].to_numpy(), axis=0)
                    )
                    for table_chunk in table_iter(
                        tokens.source._data, convert_to_tensor_chunk_size
                    )
                ]
            )
    except ImportError:
        assert len(tokens.shape) == 2

    return tokens


def populate_cache(
    run_cfg,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
    dataset_cache_dir: str | None = None,
) -> None:
    """Run latent cache and save splits to latents_path."""
    from delphi.latents import LatentCache

    dataset_cache_dir = dataset_cache_dir or DEFAULT_DATASET_CACHE_DIR
    latents_path.mkdir(parents=True, exist_ok=True)
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data_with_cache(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        dataset_cache_dir,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(
                flattened_tokens, torch.tensor([tokenizer.bos_token_id])
            )
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens)
                - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )
    cache.save_config(
        save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model
    )


def build_latent_dict(
    hookpoints: list[str],
    target_features: dict[str, list[int]],
    default_max_latents: int | None,
) -> dict[str, Tensor] | None:
    """
    Build a per-hookpoint latent dict from targeted feature IDs.
    For hookpoints in target_features, use those IDs; otherwise arange(default_max_latents).
    Returns None only if both target_features is empty and default_max_latents is None.
    """
    latent_dict = {}
    for hook in hookpoints:
        if hook in target_features:
            latent_dict[hook] = torch.tensor(
                target_features[hook], dtype=torch.long
            )
        elif default_max_latents is not None:
            latent_dict[hook] = torch.arange(default_max_latents)
    return latent_dict if latent_dict else None


@torch.no_grad()
def get_custom_text_activations(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    hookpoint_to_sparse_encode: dict[str, Callable],
    target_features: dict[str, list[int]],
    transcode: bool = False,
    max_length: int = 512,
) -> dict[str, list[dict]]:
    """
    Run model on custom texts and return targeted feature activations per token.

    Returns a dict keyed by hookpoint. Each value is a list (one per text) of dicts:
        {"text": str, "tokens": list[str], "feature_activations": {feature_id: list[float]}}
    """
    from delphi.latents.collect_activations import collect_activations

    hookpoints = list(hookpoint_to_sparse_encode.keys())
    results: dict[str, list[dict]] = {hp: [] for hp in hookpoints}

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = inputs["input_ids"].to(model.device)
        token_strs = [tokenizer.decode(tid) for tid in input_ids[0]]

        with collect_activations(model, hookpoints, transcode) as activations:
            model(input_ids)

        for hookpoint in hookpoints:
            latents = hookpoint_to_sparse_encode[hookpoint](
                activations[hookpoint]
            )
            latents_2d = latents[0]

            feat_ids = target_features.get(hookpoint, [])
            feat_acts = {}
            for fid in feat_ids:
                if fid < latents_2d.shape[1]:
                    feat_acts[fid] = latents_2d[:, fid].cpu().tolist()

            results[hookpoint].append(
                {
                    "text": text,
                    "tokens": token_strs,
                    "feature_activations": feat_acts,
                }
            )

    return results


def print_activation_summary(
    results: dict[str, list[dict]],
    top_k: int = 5,
    activation_threshold: float = 0.0,
) -> None:
    """Print a readable summary of which targeted features fired on which tokens."""
    for hookpoint, text_results in results.items():
        print(f"\n{'='*80}")
        print(f"Hookpoint: {hookpoint}")
        print(f"{'='*80}")

        for entry in text_results:
            print(f"\nText: {entry['text'][:100]}...")
            tokens = entry["tokens"]
            feat_acts = entry["feature_activations"]

            if not feat_acts:
                print("  (no targeted features in this hookpoint)")
                continue

            active_triples = []
            for fid, acts in feat_acts.items():
                for tok_idx, act_val in enumerate(acts):
                    if act_val > activation_threshold:
                        active_triples.append((fid, tok_idx, act_val))

            active_triples.sort(key=lambda x: x[2], reverse=True)

            if not active_triples:
                print("  No features activated above threshold.")
                continue

            print(
                f"  Top activations (threshold={activation_threshold}):"
            )
            for fid, tok_idx, act_val in active_triples[:top_k]:
                tok_str = tokens[tok_idx].replace("\n", "\\n")
                print(
                    f"    Feature {fid:>7d} | token {tok_idx:>3d} "
                    f"'{tok_str}' | activation {act_val:.4f}"
                )


def plot_activation_heatmap(
    results: dict[str, list[dict]],
    hookpoint: str,
    text_idx: int = 0,
    feature_ids: list[int] | None = None,
    max_features: int = 20,
    max_tokens: int = 40,
) -> None:
    """
    Plot a heatmap of feature activations across tokens for one text.
    feature_ids: if provided, only plot these; else auto-select top active features.
    """
    import plotly.graph_objects as go

    entry = results[hookpoint][text_idx]
    tokens = entry["tokens"][:max_tokens]
    feat_acts = entry["feature_activations"]

    if not feat_acts:
        print(f"No targeted features for {hookpoint}")
        return

    if feature_ids is not None:
        active_fids = [fid for fid in feature_ids if fid in feat_acts]
    else:
        active_fids = sorted(
            [
                fid
                for fid, acts in feat_acts.items()
                if any(a > 0 for a in acts[:max_tokens])
            ]
        )[:max_features]

    if not active_fids:
        print(
            f"No features activated for {hookpoint} on text {text_idx}"
        )
        return

    z = []
    y_labels = []
    for fid in active_fids:
        z.append(feat_acts[fid][:max_tokens])
        y_labels.append(f"F{fid}")

    token_labels = [t.replace("\n", "\\n") for t in tokens]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=token_labels,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Activation"),
        )
    )
    fig.update_layout(
        title=f"{hookpoint} — Feature Activations on Custom Text",
        xaxis_title="Token",
        yaxis_title="Feature ID",
        height=max(300, 40 * len(active_fids) + 100),
        width=max(600, 25 * len(tokens) + 100),
        xaxis=dict(tickangle=45),
    )
    fig.show()


def activations_to_dataframe(
    results: dict[str, list[dict]],
    years: list[str] | None = None,
):
    """
    Flatten activation results into a DataFrame.
    If years is provided, adds a 'year' column (one label per text index).
    """
    import pandas as pd

    rows = []
    for hookpoint, text_results in results.items():
        for text_idx, entry in enumerate(text_results):
            tokens = entry["tokens"]
            year = years[text_idx] if years is not None else None
            for fid, acts in entry["feature_activations"].items():
                for tok_idx, act_val in enumerate(acts):
                    if act_val > 0:
                        row = {
                            "hookpoint": hookpoint,
                            "text_idx": text_idx,
                            "text": entry["text"][:80],
                            "token_idx": tok_idx,
                            "token": tokens[tok_idx],
                            "feature_id": fid,
                            "activation": act_val,
                        }
                        if year is not None:
                            row["year"] = year
                        rows.append(row)
    return pd.DataFrame(rows)


async def process_cache(
    run_cfg,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_dict: dict[str, Tensor] | None = None,
) -> None:
    """Run the explain + score pipeline on cached latents.

    Args:
        latent_dict: per-hookpoint tensor of feature IDs to explain.
            Built by build_latent_dict(). None means explain everything.
    """
    from delphi.clients import Offline, OpenRouter
    from delphi.explainers import ContrastiveExplainer, DefaultExplainer, NoOpExplainer
    from delphi.explainers.explainer import ExplainerResult
    from delphi.latents import LatentDataset
    from delphi.pipeline import Pipe, Pipeline, process_wrapper
    from delphi.scorers import DetectionScorer, FuzzingScorer, OpenAISimulator

    explanations_path.mkdir(parents=True, exist_ok=True)

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=0.75,
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )
        llm_client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    if not run_cfg.explainer == "none":
        def explainer_postprocess(result):
            with open(
                explanations_path / f"{result.record.latent}.txt", "wb"
            ) as f:
                f.write(orjson.dumps(result.explanation))
            return result

        if run_cfg.constructor_cfg.non_activating_source == "FAISS":
            explainer = ContrastiveExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )
        else:
            explainer = DefaultExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )

        explainer_pipe = Pipe(
            process_wrapper(explainer, postprocess=explainer_postprocess)
        )
    else:
        def none_postprocessor(result):
            explanation_path = (
                explanations_path / f"{result.record.latent}.txt"
            )
            if not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file {explanation_path} does not exist. "
                    "Make sure to run an explainer pipeline first."
                )
            with open(explanation_path, "rb") as f:
                return ExplainerResult(
                    record=result.record,
                    explanation=orjson.loads(f.read()),
                )

        explainer_pipe = Pipe(
            process_wrapper(NoOpExplainer(), postprocess=none_postprocessor)
        )

    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorers = []
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)

        if scorer_name == "simulation":
            if isinstance(llm_client, Offline):
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=True
                )
            else:
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=False
                )
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
                fuzz_type=run_cfg.fuzz_type,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scorer_path),
        )
        scorers.append(wrapped_scorer)

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        Pipe(*scorers),
    )

    if (
        run_cfg.pipeline_num_proc > 1
        and run_cfg.explainer_provider == "openrouter"
    ):
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    await pipeline.run(run_cfg.pipeline_num_proc)
