import torchvision.transforms as T
import timm
from transformers import AutoModel
from wildlife_tools.features import SuperPointExtractor, AlikedExtractor, DiskExtractor, DeepFeatures
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.wildfusion import SimilarityPipeline

def get_calibration(use_calibration):
    return IsotonicCalibration() if use_calibration else None

def get_pipelines(device='cpu', cache_prefix=None, use_calibration=False):
    pipeline_md = SimilarityPipeline(
        matcher = CosineSimilarity(),
        extractor = DeepFeatures(
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_md.lmdb" if cache_prefix else None
        ),
        transform = T.Compose([
            T.Resize(size=(384, 384)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        calibration = get_calibration(use_calibration),
    )
    pipeline_miew = SimilarityPipeline(
        matcher = CosineSimilarity(),
        extractor = DeepFeatures(
            model = AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True),
            device=device,
            batch_size=8,
            cache_path=f"{cache_prefix}_miew.lmdb" if cache_prefix else None
        ),
        transform = T.Compose([
            T.Resize(size=(512, 512)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        calibration = get_calibration(use_calibration),
    )
    pipeline_aliked = SimilarityPipeline(
        matcher = MatchLightGlue(features='aliked'),
        extractor = AlikedExtractor(cache_path=f"{cache_prefix}_aliked.lmdb" if cache_prefix else None),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = get_calibration(use_calibration),
    )    
    pipeline_disk = SimilarityPipeline(
        matcher = MatchLightGlue(features='disk'),
        extractor = DiskExtractor(cache_path=f"{cache_prefix}_disk.lmdb" if cache_prefix else None),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = get_calibration(use_calibration),
    )
    pipeline_superpoint = SimilarityPipeline(
        matcher = MatchLightGlue(features='superpoint'),
        extractor = SuperPointExtractor(cache_path=f"{cache_prefix}_superpoint.lmdb" if cache_prefix else None),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = get_calibration(use_calibration),
    )

    return pipeline_md, pipeline_miew, pipeline_aliked, pipeline_disk, pipeline_superpoint