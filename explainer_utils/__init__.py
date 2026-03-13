"""
Common functions for the targeted feature explainer notebooks (Gemma, Llama, Qwen).
"""

from explainer_utils.common import (
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_PILE_DATA_FILES,
    activations_to_dataframe,
    build_latent_dict,
    get_custom_text_activations,
    get_target_layers,
    load_artifacts,
    load_tokenized_data_with_cache,
    non_redundant_hookpoints,
    plot_activation_heatmap,
    populate_cache,
    print_activation_summary,
    process_cache,
)

__all__ = [
    "DEFAULT_DATASET_CACHE_DIR",
    "DEFAULT_PILE_DATA_FILES",
    "activations_to_dataframe",
    "build_latent_dict",
    "get_custom_text_activations",
    "get_target_layers",
    "load_artifacts",
    "load_tokenized_data_with_cache",
    "non_redundant_hookpoints",
    "plot_activation_heatmap",
    "populate_cache",
    "print_activation_summary",
    "process_cache",
]
