# explainer_utils

Shared helper functions for the targeted feature explainer notebooks (Gemma, Llama, Qwen).

## Usage

From the **repository root** (so that `explainer_utils` is on the path), or after adding the repo root to `sys.path`:

```python
from explainer_utils import (
    get_target_layers,
    load_artifacts,
    non_redundant_hookpoints,
    load_tokenized_data_with_cache,
    populate_cache,
    build_latent_dict,
    get_custom_text_activations,
    print_activation_summary,
    plot_activation_heatmap,
    activations_to_dataframe,
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_PILE_DATA_FILES,
)
```

## Model-specific configuration

- **get_target_layers**: pass `features_path` (path to the CSV with `Source_Id` column) and optional `layer_filter` (list of layer indices).
- **load_artifacts**: optional `cache_dir` for the Hugging Face cache.
- **load_tokenized_data_with_cache**: optional `data_files` for pre-cached Pile paths; defaults to `DEFAULT_PILE_DATA_FILES`.
- **activations_to_dataframe**: optional `years` list to add a `year` column (e.g. for Gemma year-events analysis).

## Notebook setup

If the notebook changes the working directory (e.g. to `lora-finetuning-saes`), run the following **before** `os.chdir()` so that `explainer_utils` remains importable:

```python
import sys
from pathlib import Path
_repo = Path.cwd()
if (_repo / "explainer_utils").exists():
    sys.path.insert(0, str(_repo))
```

Then run the rest of the notebook (including `os.chdir(...)`). The import cell can use `from explainer_utils import ...` as above.
