import numpy as np
import timm
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
from wildlife_tools.features import (
    AlikedExtractor,
    DeepFeatures,
    DinoFeatures,
    DiskExtractor,
    SuperPointExtractor,
)
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion


class Wildfusion_Similarity:
    def __init__(self, priority, pipeline, B):
        self.priority = priority
        self.pipeline = pipeline
        self.B = B
        self.wildfusion = WildFusion(calibrated_pipelines=[self.pipeline], priority_pipeline=self.priority)

    def __call__(self, query, database):
        similarity = self.wildfusion(query, database, B=self.B, ignore_pairs=self.ignore_pairs)
        return similarity.astype(np.float64)

    def set_ignore_pairs(self, ignore_pairs: list[tuple[int, int]]):
        self.ignore_pairs = ignore_pairs


def get_calibration(use_calibration) -> IsotonicCalibration | None:
    return IsotonicCalibration() if use_calibration else None


def get_pipeline_dino(
    model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    output_name="dino-vitl16",
    device="cpu",
    cache_prefix=None,
    use_calibration=False,
):

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map=device)

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DinoFeatures(
            model,
            processor,
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_{output_name}.lmdb" if cache_prefix else None,
        ),
        transform=None,
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_mega(device="cpu", cache_prefix=None, use_calibration=False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(
            model=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_md.lmdb" if cache_prefix else None,
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


def get_pipeline_miew(device="cpu", cache_prefix=None, use_calibration=False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(
            model=AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_miew.lmdb" if cache_prefix else None,
        ),
        transform=T.Compose(
            [
                T.Resize(size=(512, 512)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_aliked(device="cpu", cache_prefix=None, use_calibration=False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=MatchLightGlue(features="aliked", device=device),
        extractor=AlikedExtractor(
            cache_path=f"{cache_prefix}_aliked.lmdb" if cache_prefix else None,
            device=device,
        ),
        transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_disk(device="cpu", cache_prefix=None, use_calibration=False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=MatchLightGlue(features="disk"),
        extractor=DiskExtractor(cache_path=f"{cache_prefix}_disk.lmdb" if cache_prefix else None),
        transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
        calibration=get_calibration(use_calibration),
    )


def get_pipeline_superpoint(device="cpu", cache_prefix=None, use_calibration=False) -> SimilarityPipeline:
    return SimilarityPipeline(
        matcher=MatchLightGlue(features="superpoint", device=device),
        extractor=SuperPointExtractor(
            cache_path=f"{cache_prefix}_superpoint.lmdb" if cache_prefix else None,
            device=device,
        ),
        transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
        calibration=get_calibration(use_calibration),
    )


def get_pipelines(device="cpu", cache_prefix=None, use_calibration=False):
    pipeline_mega = get_pipeline_mega(device=device, cache_prefix=cache_prefix, use_calibration=use_calibration)
    pipeline_miew = get_pipeline_miew(device=device, cache_prefix=cache_prefix, use_calibration=use_calibration)
    pipeline_aliked = get_pipeline_aliked(device=device, cache_prefix=cache_prefix, use_calibration=use_calibration)
    pipeline_disk = get_pipeline_disk(device=device, cache_prefix=cache_prefix, use_calibration=use_calibration)
    pipeline_superpoint = get_pipeline_superpoint(
        device=device, cache_prefix=cache_prefix, use_calibration=use_calibration
    )

    return (
        pipeline_mega,
        pipeline_miew,
        pipeline_aliked,
        pipeline_disk,
        pipeline_superpoint,
    )
