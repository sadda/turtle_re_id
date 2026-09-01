# Multi-body-part animal re-identification

Shared library for re-identification of animals from images, built around the [wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets) and [wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools) ecosystem, which supports merging predictions made on multiple body parts of the same individual (e.g. head and flippers) into a single identity match. The package provides the common building blocks — dataset loaders, detection/segmentation models, and similarity computation (feature extraction, matching pipelines, caching) — that are shared across projects. Individual research projects and dataset-specific pipelines live under [scripts/](scripts/), each in its own subfolder, and reuse this shared code.

## Installation

After cloning the repo, install it into a virtual environment either by

```bash
pip install .
```

when working with models (detectors, extracting features, computing similarity, ...) by

```bash
pip install -e ".[models]"
```

If DINOv3 is used for feature extraction, you will need to log in to HuggingFace with access to DINOv3 (`huggingface-cli login`).

## Usage

Dataset and output locations are configured in `animal_re_id/config/paths.json`, but can be configured manually. They are keyed by dataset name. See the individial projects in `scripts/` for examples on how to use the package.


## Example

For a full project built on this library, see [scripts/MultiBodyMerging](scripts/MultiBodyMerging), which computes and merges multi-body-part similarity across several real sea turtle datasets and evaluates the resulting matching accuracy.
