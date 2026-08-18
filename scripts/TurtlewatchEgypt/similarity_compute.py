import ast
import os
import warnings

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.datasets import SeaTurtleID2022, TurtlesOfSMSRC, TurtlewatchEgypt_New
from animal_re_id.similarity import SimilarityCache, SimilarityGrid, get_similarity_dataset_config

warnings.filterwarnings("ignore")


dataset_name = "TurtlewatchEgypt"
# dataset_name = "SeaTurtleID2022"
# dataset_name = "TurtlesOfSMSRC"

B = 100
device = "cuda"

local_path = "config/paths.json"
paths = load_paths(dataset_name, local_path=local_path)

config = get_similarity_dataset_config(dataset_name, B=B)
img_load = config.img_load

# TODO: this should be part of the splitter/config
database_query_matching = True

if dataset_name == "TurtlewatchEgypt":
    matching_dataset_class = TurtlewatchEgypt_New
    dataset_kwargs = {"img_load": img_load, "remove_unknown": True}
elif dataset_name == "SeaTurtleID2022":
    matching_dataset_class = SeaTurtleID2022
    dataset_kwargs = {"img_load": img_load}
elif dataset_name == "TurtlesOfSMSRC":
    with open("scripts/SMSRC/hidden/matched_encounters.txt") as file:
        replace_identity = ast.literal_eval(file.read())

    database_query_matching = False
    matching_dataset_class = TurtlesOfSMSRC
    dataset_kwargs = {"img_load": img_load, "replace_identity": replace_identity}
else:
    raise ValueError("Dataset now known")

matching_dataset = matching_dataset_class.from_paths(paths, database_query_matching, dataset_kwargs=dataset_kwargs)

root_features = resolve_path(paths["root_features"])
root_similarity = resolve_path(paths["root_similarity"])
os.makedirs(root_similarity, exist_ok=True)

cache = SimilarityCache(root_similarity)
run_specs = config.run_specs()

for matching_part in config.matching_parts:
    cache_prefix = f"{root_features}/{matching_part}_{img_load}"

    grid = SimilarityGrid(
        matching_dataset=matching_dataset.for_matching_part(matching_part),
        run_specs=[run_spec for run_spec in run_specs if run_spec.matching_part == matching_part],
        cache=cache,
        cache_prefix=cache_prefix,
        device=device,
    )
    grid.run()
