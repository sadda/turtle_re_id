import ast
import time
import warnings

import pandas as pd
import torch

from animal_re_id.config import load_paths
from animal_re_id.datasets import SeaTurtleID2022, TurtlesOfSMSRC, TurtlewatchEgypt_New
from animal_re_id.models import turtle_detector
from animal_re_id.similarity import SinglePipelineSpec, WildfusionPipelineSpec, get_similarity_dataset_config

warnings.filterwarnings("ignore")


class Timer:
    def __init__(self, device: str) -> None:
        self.device = device
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start


def get_subset(matching_dataset, idx, transform):
    wildlife_dataset = matching_dataset.wildlife_dataset.get_subset(idx)
    wildlife_dataset.transform = transform
    return wildlife_dataset


class PreloadedDataset:
    def __init__(self, wildlife_dataset) -> None:
        self.metadata = wildlife_dataset.metadata
        self.col_label = wildlife_dataset.col_label
        self.items = [wildlife_dataset[i] for i in range(len(wildlife_dataset))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def get_matching_dataset(dataset_name, paths, config, img_load=None):
    img_load = img_load or config.img_load

    if dataset_name == "TurtlewatchEgypt":
        matching_dataset_class = TurtlewatchEgypt_New
        dataset_kwargs = {"img_load": img_load, "remove_unknown": True}
    elif dataset_name == "SeaTurtleID2022":
        matching_dataset_class = SeaTurtleID2022
        dataset_kwargs = {"img_load": img_load}
    elif dataset_name == "TurtlesOfSMSRC":
        with open("scripts/SMSRC/hidden/matched_encounters.txt") as file:
            replace_identity = ast.literal_eval(file.read())

        matching_dataset_class = TurtlesOfSMSRC
        dataset_kwargs = {"img_load": img_load, "replace_identity": replace_identity}
    else:
        raise ValueError("Dataset now known")

    return matching_dataset_class.from_paths(paths, config.database_query_matching, dataset_kwargs=dataset_kwargs)


dataset_names = ["TurtlewatchEgypt", "SeaTurtleID2022"]
devices = ["cuda", "cpu"]

B = 100
n_samples = 64

local_path = "config/paths.json"

rows = []
detector_rows = []
detector_model = turtle_detector()
for dataset_name in dataset_names:
    paths = load_paths(dataset_name, local_path=local_path)
    config = get_similarity_dataset_config(dataset_name, B=B)

    detector_dataset = get_subset(get_matching_dataset(dataset_name, paths, config, img_load="full"), range(n_samples), None)

    with Timer("cpu") as timer:
        detector_loaded = PreloadedDataset(detector_dataset)
    detector_image_load = timer.elapsed / len(detector_loaded)
    detector_images = [image for image, _ in detector_loaded.items]

    for device in devices:
        detector_model.predict(source=detector_images[0], device=device, verbose=False, save=False, show=False)  # warm up

        with Timer(device) as timer:
            detector_model.predict(source=detector_images, device=device, verbose=False, save=False, show=False)

        detector_rows.append(
            {
                "dataset": dataset_name,
                "device": device,
                "image_load": detector_image_load,
                "detect": timer.elapsed / len(detector_loaded),
            }
        )

    matching_dataset = get_matching_dataset(dataset_name, paths, config)

    for device in devices:
        for pipeline_spec in config.pipeline_specs:
            row = {"dataset": dataset_name, "device": device, "pipeline": pipeline_spec.name}

            with Timer(device) as timer:
                pipeline_spec.build(device, None)
            row["build"] = timer.elapsed

            if isinstance(pipeline_spec, SinglePipelineSpec):
                pipeline = pipeline_spec.pipeline
            elif isinstance(pipeline_spec, WildfusionPipelineSpec):
                pipeline = pipeline_spec.wildfusion_similarity.pipeline
            else:
                raise ValueError("Unknown pipeline_spec encountered")

            query = get_subset(matching_dataset, range(n_samples), pipeline.transform)
            database = get_subset(matching_dataset, range(n_samples, 2 * n_samples), pipeline.transform)
            warmup = get_subset(matching_dataset, range(1), pipeline.transform)
            pipeline.extractor(warmup)

            with Timer(device) as timer:
                query_loaded = PreloadedDataset(query)
                database_loaded = PreloadedDataset(database)
            row["image_load"] = timer.elapsed / (len(query) + len(database))

            # Data is already in memory, so a worker process would only add fork overhead.
            pipeline.extractor.num_workers = 0

            with Timer(device) as timer:
                query_features = pipeline.extractor(query_loaded)
                database_features = pipeline.extractor(database_loaded)
            row["extract"] = timer.elapsed / (len(query) + len(database))

            with Timer(device) as timer:
                similarity = pipeline.matcher(query_features, database_features)
            row["match"] = timer.elapsed / (len(query) * len(database))

            rows.append(row)

results = pd.DataFrame(rows).set_index(["dataset", "device", "pipeline"])
results["image_load"] = results.groupby("dataset")["image_load"].transform("mean")
results = (results[["image_load", "extract", "match"]] * 1000).round(0).astype(int)
print(results.to_latex())

detector_results = pd.DataFrame(detector_rows).set_index(["dataset", "device"])
detector_results = (detector_results * 1000).round(0).astype(int)
print(detector_results.to_latex())