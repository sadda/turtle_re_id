from .datasets import MatchingDataset, SeaTurtleID2022, StripeSpotter, TurtlesOfSMSRC, TurtlewatchEgypt_New


def get_matching_dataset(dataset_name: str, paths: dict, database_query_matching: bool, img_load: str = "bbox") -> MatchingDataset:
    if dataset_name == "TurtlewatchEgypt":
        matching_dataset_class = TurtlewatchEgypt_New
        dataset_kwargs = {"img_load": img_load, "remove_unknown": True}
    elif dataset_name == "SeaTurtleID2022":
        matching_dataset_class = SeaTurtleID2022
        dataset_kwargs = {"img_load": img_load}
    elif dataset_name == "TurtlesOfSMSRC":
        matching_dataset_class = TurtlesOfSMSRC
        dataset_kwargs = {"img_load": img_load}
    elif dataset_name == "StripeSpotter":
        matching_dataset_class = StripeSpotter
        dataset_kwargs = {}
    else:
        raise ValueError(f"Dataset {dataset_name} not known")

    return matching_dataset_class.from_paths(paths, database_query_matching, dataset_kwargs=dataset_kwargs)
