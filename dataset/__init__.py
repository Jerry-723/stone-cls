from .utils import *
from .meteorite import *
from .stone import *

DATASET_REGISTRY = {
    "meteorite": Meteorite,
    "stone": Stone
}

def get_dataset(dataset_name, *args, **kwargs):
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name](*args, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is not registered.")