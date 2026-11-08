import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import timm
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import (
    AlikedExtractor,
    DeDoDeExtractor,
    DeepFeatures,
    DinoFeatures,
    DiskExtractor,
    DoGHardNetExtractor,
    KeyNetAffNetHardNetExtractor,
    XFeatExtractor,
)
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue, PairSelector
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion


class Wildfusion_Similarity:
    def __init__(self, priority: SimilarityPipeline, pipeline: SimilarityPipeline, B: int, pair_selector: PairSelector | None = None) -> None:
        self.priority = priority
        self.pipeline = pipeline
        self.B = B
        self.wildfusion = WildFusion(
            calibrated_pipelines=[self.pipeline],
            priority_pipeline=self.priority,
            pair_selector=pair_selector,
        )

    def __call__(self, query: WildlifeDataset, database: WildlifeDataset) -> np.ndarray:
        similarity = self.wildfusion(query, database, B=self.B)
        return similarity.astype(np.float64)

    def set_pair_selector(self, pair_selector: PairSelector) -> None:
        self.wildfusion.pair_selector = pair_selector


def get_calibration(use_calibration: bool) -> IsotonicCalibration | None:
    return IsotonicCalibration() if use_calibration else None


def get_pipeline_dino(
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    output_name: str = "dino-vitl16",
    device: str = "cpu",
    cache_prefix: str | None = None,
    use_calibration: bool = False,
) -> SimilarityPipeline:

    processor = AutoImageProcessor.from_pretrained(model_name)

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DinoFeatures.lazy_load(
            functools.partial(AutoModel.from_pretrained, model_name, device_map=device),
            processor=processor,
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_{output_name}.lmdb" if cache_prefix else None,
            config_tag="hf_processor",
        ),
        transform=None,
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_mega(device: str = "cpu", cache_prefix: str | None = None, use_calibration: bool = False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures.lazy_load(
            functools.partial(timm.create_model, "hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_md-l-384.lmdb" if cache_prefix else None,
            config_tag="resize384_imagenet-norm",
        ),
        transform=T.Compose(
            [
                T.Resize(size=(384, 384)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_miew(device: str = "cpu", cache_prefix: str | None = None, use_calibration: bool = False, img_size: int = 512) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures.lazy_load(
            functools.partial(AutoModel.from_pretrained, "conservationxlabs/miewid-msv3", trust_remote_code=True),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_miewid3-{img_size}.lmdb" if cache_prefix else None,
            config_tag=f"resize{img_size}_imagenet-norm",
        ),
        transform=T.Compose(
            [
                T.Resize(size=(img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_lightglue(
    features: str,
    extractor_class: Callable[..., Any],
    device: str = "cpu",
    cache_prefix: str | None = None,
    use_calibration: bool = False,
    img_size: int = 512,
    cache_scores_only: bool = True,
) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=MatchLightGlue(
            features=features,
            device=device,
            cache_path=f"{cache_prefix}_{features}_lightglue-{img_size}.lmdb" if cache_prefix else None,
            config_tag=f"{features}_resize{img_size}",
            cache_scores_only=cache_scores_only,
        ),
        extractor=extractor_class(
            cache_path=f"{cache_prefix}_{features}-{img_size}.lmdb" if cache_prefix else None,
            config_tag=f"resize{img_size}",
            device=device,
        ),
        transform=T.Compose([T.Resize([img_size, img_size]), T.ToTensor()]),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_aliked(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("aliked", AlikedExtractor, **kwargs)


def get_pipeline_disk(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("disk", DiskExtractor, **kwargs)


def get_pipeline_xfeat(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("xfeat", XFeatExtractor, **kwargs)


def get_pipeline_dedodeb(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("dedodeb", functools.partial(DeDoDeExtractor, descriptor="B"), **kwargs)


def get_pipeline_dedodeg(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("dedodeg", functools.partial(DeDoDeExtractor, descriptor="G"), **kwargs)


def get_pipeline_doghardnet(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("doghardnet", DoGHardNetExtractor, **kwargs)


def get_pipeline_dog_affnet_hardnet(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("dog_affnet_hardnet", functools.partial(DoGHardNetExtractor, affnet=True), **kwargs)


def get_pipeline_keynet_affnet_hardnet(**kwargs: Any) -> SimilarityPipeline:
    return get_pipeline_lightglue("keynet_affnet_hardnet", KeyNetAffNetHardNetExtractor, **kwargs)
