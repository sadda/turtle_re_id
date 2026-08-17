from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import PairSelector, SimilarityPipeline

from ..models import Wildfusion_Similarity


class PipelineSpec(ABC):
    def __init__(self, name: str, B: int | None = None) -> None:
        self.name = name
        self.B = B

    @abstractmethod
    def build(self, device: str, cache_prefix: str | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
    ) -> np.ndarray:
        raise NotImplementedError


class SinglePipelineSpec(PipelineSpec):
    def __init__(self, name: str, pipeline_fun: Callable[..., SimilarityPipeline], B: int | None = None) -> None:
        super().__init__(name, B)
        self.pipeline_fun = pipeline_fun
        self.pipeline: SimilarityPipeline | None = None

    def build(self, device: str, cache_prefix: str | None) -> None:
        self.pipeline = self.pipeline_fun(device=device, cache_prefix=cache_prefix)

    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
    ) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: call build() before compute()")
        similarity = self.pipeline(query, database)

        # SinglePipelineSpec computes the full similarity matrix (no shortlisting), so the
        # ignore mask has to be applied here explicitly instead of via pair_selector.
        if hasattr(pair_selector, "get_ignore_mask"):
            ignore_mask = pair_selector.get_ignore_mask(query, database)
            similarity[ignore_mask] = -np.inf
        return similarity


class WildfusionPipelineSpec(PipelineSpec):
    def __init__(
        self,
        name: str,
        pipeline1_fun: Callable[..., SimilarityPipeline],
        pipeline2_fun: Callable[..., SimilarityPipeline],
        B: int | None = None,
    ) -> None:
        super().__init__(name, B)
        self.pipeline1_fun = pipeline1_fun
        self.pipeline2_fun = pipeline2_fun
        self.wildfusion_similarity: Wildfusion_Similarity | None = None

    def build(self, device: str, cache_prefix: str | None) -> None:
        pipeline1 = self.pipeline1_fun(device=device, cache_prefix=cache_prefix)
        pipeline2 = self.pipeline2_fun(device=device, cache_prefix=cache_prefix)
        self.wildfusion_similarity = Wildfusion_Similarity(pipeline1, pipeline2, self.B)

    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
    ) -> np.ndarray:
        if self.wildfusion_similarity is None:
            raise RuntimeError(f"{self.name}: call build() before compute()")
        self.wildfusion_similarity.set_pair_selector(pair_selector)
        return self.wildfusion_similarity(query, database)
