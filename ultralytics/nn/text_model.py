from abc import abstractmethod
from typing import Optional

import clip
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import smart_inference_mode

try:  # pragma: no cover - optional dependency
    import mobileclip  # type: ignore
except ImportError:  # pragma: no cover
    mobileclip = None

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(texts):
        pass
    
    @abstractmethod
    def encode_text(texts, dtype):
        pass

class CLIP(TextModel):
    def __init__(self, size, device):
        super().__init__()
        self.model, self.preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        return clip.tokenize(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats
        
class MobileCLIP(TextModel):
    
    config_size_map = {
        "s0": "s0",
        "s1": "s1",
        "s2": "s2",
        "b": "b",
        "blt": "b"
    }
    
    def __init__(self, size, device):
        super().__init__()
        if mobileclip is None:
            raise ImportError("mobileclip is required for MobileCLIP text model but is not installed.")
        config = self.config_size_map[size]
        model_and_transforms = mobileclip.create_model_and_transforms(
            f"mobileclip_{config}", pretrained=f"mobileclip_{size}.pt", device=device
        )
        self.model = model_and_transforms[0]
        self.preprocess = model_and_transforms[1] if len(model_and_transforms) > 1 else None
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        # max_len = text_tokens.argmax(dim=-1).max().item() + 1
        # text_tokens = text_tokens[..., :max_len]
        return text_tokens

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

def build_text_model(variant: str, device: Optional[torch.device] = None, allow_tuner: bool = True):
    LOGGER.info(f"Build text model {variant}")
    parts = variant.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid text model variant '{variant}'. Expected format '<model>:<size>'.")

    base = parts[0]
    size = ":".join(parts[1:])

    if base == "clip":
        return CLIP(size, device)
    if base == "mobileclip":
        return MobileCLIP(size, device)
    if base == "coop":
        if not allow_tuner:
            raise ValueError("Nested prompt tuners are not supported.")
        from ultralytics.nn.prompt_tuning import COOPPromptTuner

        return COOPPromptTuner(size, device=device)

    raise ValueError(f"Unknown text model variant '{variant}'.")
