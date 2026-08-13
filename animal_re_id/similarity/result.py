from dataclasses import dataclass
from typing import Any

import numpy as np
from wildlife_datasets.datasets import WildlifeDataset


@dataclass
class SimilarityResult:
    similarity: np.ndarray
    query: WildlifeDataset
    database: WildlifeDataset
    name: str
    B: int | None
    img_load: str
    matching_part: str
    match_opposite_sides: Any
    split_year: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity": self.similarity,
            "query": self.query,
            "database": self.database,
            "name": self.name,
            "B": self.B,
            "img_load": self.img_load,
            "matching_part": self.matching_part,
            "match_opposite_sides": self.match_opposite_sides,
            "split_year": self.split_year,
        }
