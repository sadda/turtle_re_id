import warnings

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.datasets import get_matching_dataset
from animal_re_id.similarity import SimilarityCache, SimilarityGrid, get_similarity_dataset_config

warnings.filterwarnings("ignore")


# dataset_name = "SeaTurtleID2022"
# dataset_name = "StripeSpotter"
dataset_name = "TurtlewatchEgypt"
# dataset_name = "TurtlesOfSMSRC"

B = 100
device = "cuda"

local_path = "config/paths.json"
paths = load_paths(dataset_name, local_path=local_path)
config = get_similarity_dataset_config(dataset_name, B=B)

matching_dataset = get_matching_dataset(dataset_name, paths, config.database_query_matching, img_load=config.img_load)

root_features = resolve_path(paths["root_features"])
root_similarity = resolve_path(paths["root_similarity"])
cache = SimilarityCache(root_similarity)
run_specs = config.run_specs()

for matching_part in config.matching_parts:
    cache_prefix = f"{root_features}/{matching_part}_{config.img_load}"

    grid = SimilarityGrid(
        matching_dataset=matching_dataset.for_matching_part(matching_part),
        run_specs=[run_spec for run_spec in run_specs if run_spec.matching_part == matching_part],
        cache=cache,
        cache_prefix=cache_prefix,
        device=device,
    )
    grid.run()
