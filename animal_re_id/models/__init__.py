from .detector import turtle_detector
from .pipelines import (
    Wildfusion_Similarity,
    get_pipeline_aliked,
    get_pipeline_dedodeb,
    get_pipeline_dedodeg,
    get_pipeline_dino,
    get_pipeline_disk,
    get_pipeline_dog_affnet_hardnet,
    get_pipeline_doghardnet,
    get_pipeline_keynet_affnet_hardnet,
    get_pipeline_mega,
    get_pipeline_miew,
    get_pipeline_xfeat,
)
from .predictions import EncounterPredictions, merge_data
from .utils import plot_matches, visualise_match
