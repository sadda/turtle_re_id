import os
import pickle
from typing import Any


class SimilarityCache:
    def __init__(self, root: str) -> None:
        self.root = root

    def path(self, key_parts: list[Any]) -> str:
        file_name = "_".join(str(part) for part in key_parts)
        return os.path.join(self.root, f"{file_name}.pkl")

    def exists(self, key_parts: list[Any]) -> bool:
        return os.path.exists(self.path(key_parts))

    def save(self, key_parts: list[Any], data: dict[str, Any]) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.path(key_parts), "wb") as file:
            pickle.dump(data, file)

    def load(self, key_parts: list[Any]) -> dict[str, Any]:
        with open(self.path(key_parts), "rb") as file:
            return pickle.load(file)
