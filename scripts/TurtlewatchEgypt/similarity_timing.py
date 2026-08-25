import ast
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
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


def images_per_hour(time_per_image_ms):
    return 3600 / (time_per_image_ms / 1000)


dataset_names = ["TurtlewatchEgypt", "SeaTurtleID2022"]
devices = ["cuda", "cpu"]
device_to_name = {"cuda": "GPU", "cuda:2": "GPU", "cpu": "CPU"}

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
        # Ultralytics setting for a proper timing
        torch_device = torch.device(device)
        detector_model.predictor = None
        detector_model.predict(source=detector_images[0], device=torch_device, verbose=False, save=False, show=False)  # warm up

        with Timer(device) as timer:
            detector_model.predict(source=detector_images, device=torch_device, verbose=False, save=False, show=False)

        detector_rows.append(
            {
                "dataset": dataset_name,
                "device": device_to_name[device],
                "image_load": detector_image_load,
                "detect": timer.elapsed / len(detector_loaded),
            }
        )

    matching_dataset = get_matching_dataset(dataset_name, paths, config)

    for device in devices:
        for pipeline_spec in config.pipeline_specs:
            row = {"dataset": dataset_name, "device": device_to_name, "pipeline": pipeline_spec.name}

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
results_print = (results[["image_load", "extract", "match"]] * 1000).round(0).astype(int)
print(results_print.to_latex())

detector_results = pd.DataFrame(detector_rows).set_index(["dataset", "device"])
detector_results_print = (detector_results * 1000).round(0).astype(int)
print(detector_results_print.to_latex())

raw_results = pd.DataFrame(rows).set_index(["dataset", "device", "pipeline"])[["image_load", "extract", "match"]] * 1000

Bs = np.arange(1, 101)

for method_name, dataset_name in zip(["miew", "dino"], dataset_names):
    plt.figure(figsize=(5, 3))
    for device in devices:
        device_name = device_to_name[device]
        dino_row = raw_results.loc[(dataset_name, device_name, method_name)]
        aliked_row = raw_results.loc[(dataset_name, device_name, f"{method_name}-aliked")]

        dino_only_time = dino_row["image_load"] + dino_row["extract"]
        wildfusion_time = dino_only_time + aliked_row["image_load"] + aliked_row["extract"] + Bs * aliked_row["match"]

        line, = plt.plot(Bs, images_per_hour(wildfusion_time), label=f"{method_name.upper()}-ALIKED ({device_name})")
        plt.axhline(images_per_hour(dino_only_time), color=line.get_color(), linestyle="--", label=f"{method_name.upper()} only ({device_name})")

    plt.legend(loc="upper right")
    plt.grid(alpha=0.7)
    plt.xlim(0, Bs.max())
    plt.xlabel("B")
    plt.ylabel("processed images per hour")
    plt.title(dataset_name)
    plt.savefig(f"{dataset_name}_time.png", dpi=300, bbox_inches="tight")