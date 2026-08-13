from ..models import Wildfusion_Similarity
from .compute import SinglePipelineComputer, WildfusionComputer


class PipelineSpec:
    def __init__(self, name, pipeline1_fun, pipeline2_fun=None, B=None):
        self.name = name
        self.pipeline1_fun = pipeline1_fun
        self.pipeline2_fun = pipeline2_fun
        self.B = B

    def build(self, device, cache_prefix):
        pipeline1 = self.pipeline1_fun(device=device, cache_prefix=cache_prefix)
        if self.pipeline2_fun is None:
            return SinglePipelineComputer(pipeline1)

        pipeline2 = self.pipeline2_fun(device=device, cache_prefix=cache_prefix)
        wildfusion_similarity = Wildfusion_Similarity(pipeline1, pipeline2, self.B)
        return WildfusionComputer(wildfusion_similarity)
