# MultiBodyMerging

Code for the paper on merging re-identification predictions made from multiple body parts of the same animal into a single match. The focus is on sea turtles using the `SeaTurtleID2022` and `TurtlewatchEgypt` datasets and merging the predictions from heads, bodies, front flippers and hind flippers. The former dataset can be installed directly via WildlifeDatasets, the latter is not publicly available as of now. The body parts segmentations are provided in the metadata for `SeaTurtleID2022` and must be obtained using [TurtleDetector](https://huggingface.co/BVRA/TurtleDetector) for `TurtlewatchEgypt`.

## Installation

See the [main README](../../README.md).

## Usage

1. `statistics.ipynb` — dataset-level statistics (encounters per year, known vs. new identities, number of segmentation masks per matching part per encounter).
2. `similarity_compute.py` — for each dataset and matching part, extracts features and computes/caches pairwise similarity grids (`SimilarityCache`, `SimilarityGrid`).
3. `similarity_analyse.ipynb` — loads the cached similarities, merges per-part predictions (`merge_data`), and evaluates matching accuracy as a function of the candidate shortlist size `B`.
4. `similarity_timing.py` — benchmarks detection, feature extraction, and matching time per pipeline/device, used for the paper's runtime figures/tables.

The dataset must be placed in the paths specified in the default paths `animal_re_id/config/paths.json` or the paths must be specified (see the main README).
