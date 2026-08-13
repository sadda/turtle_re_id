import os
import pickle


class SimilarityCache:
    def __init__(self, root):
        self.root = root

    def key_for(self, pipeline_spec, ignore_policy, split, img_load, matching_part):
        return [
            pipeline_spec.name,
            pipeline_spec.B,
            img_load,
            matching_part,
            ignore_policy.cache_key,
            split.name,
        ]

    def path(self, key_parts):
        file_name = "_".join(str(part) for part in key_parts)
        return os.path.join(self.root, f"{file_name}.pkl")

    def exists(self, key_parts):
        return os.path.exists(self.path(key_parts))

    def save(self, key_parts, data):
        with open(self.path(key_parts), "wb") as file:
            pickle.dump(data, file)

    def load(self, key_parts):
        with open(self.path(key_parts), "rb") as file:
            return pickle.load(file)
