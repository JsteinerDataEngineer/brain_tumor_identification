from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIRECTORY = 'src\saved_models'
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


### insert models here



MODEL_FACTORY = {}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs
) -> torch.nn.Module:
    """
    Load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = DIRECTORY / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"

            ) from e
        

def save_model(model: torch.nn.Module) -> str:
    """
    Function used to save model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    
    output_path = DIRECTORY / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path