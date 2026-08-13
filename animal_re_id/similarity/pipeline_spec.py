from collections.abc import Callable

from wildlife_tools.similarity import SimilarityPipeline

from ..models import Wildfusion_Similarity
from .compute import SimilarityComputer, SinglePipelineComputer, WildfusionComputer


class PipelineSpec:
    def __init__(
        self,
        name: str,
        pipeline1_fun: Callable[..., SimilarityPipeline],
        pipeline2_fun: Callable[..., SimilarityPipeline] | None = None,
        B: int | None = None,
    ) -> None:
        self.name = name
        self.pipeline1_fun = pipeline1_fun
        self.pipeline2_fun = pipeline2_fun
        self.B = B

    def build(self, device: str, cache_prefix: str | None) -> SimilarityComputer:
        pipeline1 = self.pipeline1_fun(device=device, cache_prefix=cache_prefix)
        if self.pipeline2_fun is None:
            return SinglePipelineComputer(pipeline1)

        pipeline2 = self.pipeline2_fun(device=device, cache_prefix=cache_prefix)
        wildfusion_similarity = Wildfusion_Similarity(pipeline1, pipeline2, self.B)
        return WildfusionComputer(wildfusion_similarity)
