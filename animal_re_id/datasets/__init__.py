from .datasets import (
    SeaTurtleID2022,
    StripeSpotter,
    TurtlesFromMetadata,
    TurtlesOfSMSRC,
    TurtlewatchEgypt_Citizen,
    TurtlewatchEgypt_Combined,
    TurtlewatchEgypt_Master,
    TurtlewatchEgypt_New,
    combine_datasets,
    combine_datasets_twe,
    combine_datasets_twe_from_paths,
)
from .factory import get_matching_dataset
from .utils import add_date_exif
