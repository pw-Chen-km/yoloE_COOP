"""Prompt tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from ultralytics.nn.text_model import build_text_model, TextModel
from ultralytics.utils import LOGGER


@dataclass
class SupportExample:
    """Container for a single support example used during prompt tuning."""

    image: torch.Tensor
    label: int


class COOPPromptTuner(TextModel):
    """Prompt tuner that implements a simplified version of CoOp prompt learning."""

    def __init__(
        self,
        base_variant: str,
        device: Optional[torch.device] = None,
        ctx_len: int = 16,
        template: str = "a photo of a {}",
    ) -> None:
        super().__init__()
        self.base_variant = base_variant
        self.ctx_len = ctx_len
        self.template = template
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = build_text_model(base_variant, device=self.device, allow_tuner=False)
        self.model = self.base_model.model

        # Detect whether we are working with a standard CLIP model (which exposes the text encoder
        # directly) or a MobileCLIP style model which nests the encoder under ``text_encoder``.
        self._is_mobileclip = False

        if hasattr(self.model, "token_embedding"):
            self.text_encoder = self.model
            self._token_embedding_layer = self.text_encoder.token_embedding
        elif hasattr(self.model, "text_encoder"):
            self._is_mobileclip = True
            self.text_encoder = self.model.text_encoder
            if not hasattr(self.text_encoder, "embedding_layer"):
                raise AttributeError(
                    "MobileCLIP text encoder must expose an 'embedding_layer' for CoOp prompt tuning."
                )
            self._token_embedding_layer = self.text_encoder.embedding_layer
        else:
            raise AttributeError("Base text encoder must expose token embeddings for CoOp prompt tuning.")

        self._dropout = getattr(self.text_encoder, "dropout", None)
        self.context: Optional[nn.Parameter] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.class_names: Optional[List[str]] = None
        self.tokenized_prompts: Optional[torch.Tensor] = None
        self.prefix: Optional[torch.Tensor] = None
        self.suffix: Optional[torch.Tensor] = None
        self.eot_indices: Optional[torch.Tensor] = None
        self.last_lr: Optional[float] = None

    # ---------------------------------------------------------------------
    # TextModel API
    # ---------------------------------------------------------------------
    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return self.base_model.tokenize(texts)

    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.tokenized_prompts is not None and self.context is not None:
            if texts.shape == self.tokenized_prompts.shape and torch.equal(texts.cpu(), self.tokenized_prompts.cpu()):
                return self.get_text_features(dtype=dtype)
        return self.base_model.encode_text(texts, dtype=dtype)

    # ------------------------------------------------------------------
    # Prompt building helpers
    # ------------------------------------------------------------------
    def _ensure_buffers(self, name: str, value: torch.Tensor) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._token_embedding_layer is None:
            raise RuntimeError("Token embedding layer has not been initialized.")
        return self._token_embedding_layer(tokens)

    def _initialize_prompts(
        self,
        class_names: Sequence[str],
        tokenized_prompts: Optional[torch.Tensor] = None,
    ) -> None:
        self.class_names = list(class_names)
        prompts = [self.template.format(name) for name in class_names]
        tokens = tokenized_prompts if tokenized_prompts is not None else self.base_model.tokenize(prompts)
        tokens = tokens.to(self.device)

        token_embedding = self._embed_tokens(tokens).to(self.device)
        dtype = token_embedding.dtype

        ctx_init = token_embedding[:, 1 : 1 + self.ctx_len]
        if ctx_init.shape[1] < self.ctx_len:
            pad_len = self.ctx_len - ctx_init.shape[1]
            ctx_init = F.pad(ctx_init, (0, 0, 0, pad_len))
        elif ctx_init.shape[1] > self.ctx_len:
            ctx_init = ctx_init[:, : self.ctx_len]

        prefix = token_embedding[:, :1]
        suffix = token_embedding[:, 1 + self.ctx_len :]

        if "context" in self._parameters:
            del self._parameters["context"]
        self.register_parameter("context", nn.Parameter(ctx_init.clone().to(dtype)))

        self.tokenized_prompts = tokens
        self.prefix = prefix
        self.suffix = suffix
        self.eot_indices = tokens.argmax(dim=-1)
        self.to(self.device)

    def _build_prompts(self) -> torch.Tensor:
        assert self.prefix is not None and self.suffix is not None and self.context is not None
        prompts = torch.cat([self.prefix, self.context, self.suffix], dim=1)
        return prompts

    def get_text_features(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert self.tokenized_prompts is not None and self.context is not None
        prompts = self._build_prompts().to(self.device)
        if self._is_mobileclip:
            text_features = self._get_text_features_mobileclip(prompts, dtype)
        else:
            text_features = self._get_text_features_clip(prompts, dtype)
        text_features = text_features.to(dtype)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def _get_text_features_clip(self, prompts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        model_dtype = getattr(self.text_encoder, "dtype", prompts.dtype)
        prompts = prompts.to(model_dtype)
        positional = self.text_encoder.positional_embedding.to(model_dtype)
        x = prompts + positional
        x = x.permute(1, 0, 2)
        x = self.text_encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.text_encoder.ln_final(x)
        eot = self.eot_indices
        assert eot is not None
        batch_indices = torch.arange(x.shape[0], device=x.device)
        x = x[batch_indices, eot.to(x.device)] @ self.text_encoder.text_projection
        return x

    def _get_text_features_mobileclip(self, prompts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        model_dtype = getattr(self.text_encoder, "dtype", prompts.dtype)
        prompts = prompts.to(model_dtype)
        positional = self.text_encoder.positional_embedding.to(model_dtype)
        x = prompts + positional
        if self._dropout is not None:
            x = self._dropout(x)
        transformer_layers = getattr(self.text_encoder, "transformer_layers", None)
        if transformer_layers is None:
            raise AttributeError("MobileCLIP text encoder must define 'transformer_layers'.")
        for layer in transformer_layers:
            x = layer(x)
        final_layer_norm = getattr(self.text_encoder, "final_layer_norm", None)
        if final_layer_norm is None:
            raise AttributeError("MobileCLIP text encoder must define 'final_layer_norm'.")
        x = final_layer_norm(x)
        eot = self.eot_indices
        assert eot is not None
        batch_indices = torch.arange(x.shape[0], device=x.device)
        x = x[batch_indices, eot.to(x.device)]
        projection_layer = getattr(self.text_encoder, "projection_layer", None)
        if projection_layer is None:
            raise AttributeError("MobileCLIP text encoder must define 'projection_layer'.")
        x = projection_layer(x)
        return x

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def _as_examples(self, support_crops: Sequence[Tuple[torch.Tensor, int]]) -> List[SupportExample]:
        examples: List[SupportExample] = []
        for image, label in support_crops:
            if image.ndim == 3:
                tensor = image.unsqueeze(0)
            else:
                tensor = image
            examples.append(SupportExample(tensor.to(self.device), int(label)))
        return examples

    def fit(
        self,
        support_crops: Sequence[Tuple[torch.Tensor, int]],
        class_names: Sequence[str],
        steps: int = 200,
        lr: float = 1e-3,
    ) -> torch.Tensor:
        if len(support_crops) == 0:
            raise ValueError("support_crops must contain at least one example.")

        self._initialize_prompts(class_names)
        examples = self._as_examples(support_crops)

        self.train()
        self.base_model.eval()

        with torch.no_grad():
            image_batch = torch.cat([ex.image for ex in examples], dim=0).to(self.device)
            image_encoder = getattr(self.model, "encode_image", None)
            if image_encoder is None:
                raise AttributeError("Base model must expose an 'encode_image' method for CoOp prompt tuning.")
            image_features = image_encoder(image_batch)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            labels = torch.tensor([ex.label for ex in examples], device=self.device, dtype=torch.long)

        self.last_lr = lr
        self.optimizer = torch.optim.Adam([self.context], lr=lr)

        logit_scale_param = getattr(self.model, "logit_scale", None)
        if logit_scale_param is None:
            logit_scale_param = getattr(self.text_encoder, "logit_scale", None)
        if logit_scale_param is None:
            raise AttributeError("Base model must expose a 'logit_scale' parameter for CoOp prompt tuning.")
        logit_scale = logit_scale_param.exp().item()

        for _ in range(max(1, steps)):
            assert self.optimizer is not None
            self.optimizer.zero_grad()
            text_features = self.get_text_features(dtype=image_features.dtype)
            logits = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

        return self.get_text_features(dtype=torch.float32).detach()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Return an in-memory snapshot of the current prompt tuner state."""

        if self.context is None or self.tokenized_prompts is None:
            raise RuntimeError("Prompt tuner must be fitted before exporting state.")

        return {
            "base_variant": self.base_variant,
            "ctx_len": self.ctx_len,
            "template": self.template,
            "class_names": self.class_names,
            "tokenized_prompts": self.tokenized_prompts.detach().cpu(),
            "context": self.context.detach().cpu(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "lr": self.last_lr,
        }

    def save_state(self, path: str | Path) -> Path:
        state = self.get_state()

        path = Path(path)
        if path.suffix == "":
            path = path / "coop_prompt_state.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state, path)
        LOGGER.info(f"Saved CoOp prompt tuner state to {path}")
        return path

    def load_state(self, path: str | Path, map_location: Optional[str | torch.device] = None) -> None:
        path = Path(path)
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, map_location=map_location)
        LOGGER.info(f"Loaded CoOp prompt tuner state from {path}")

    def load_state_dict(self, state: dict, map_location: Optional[str | torch.device] = None) -> None:
        target_device = map_location or self.device

        if state.get("base_variant") not in {None, self.base_variant}:
            raise ValueError(
                f"State was trained for base variant {state.get('base_variant')} but current tuner uses {self.base_variant}."
            )

        self.ctx_len = state.get("ctx_len", self.ctx_len)
        self.template = state.get("template", self.template)

        tokenized_prompts = state.get("tokenized_prompts")
        class_names = state.get("class_names")
        if class_names is None or tokenized_prompts is None:
            raise ValueError("State dictionary is missing required prompt metadata.")

        tokenized_prompts = tokenized_prompts.to(target_device)
        self._initialize_prompts(class_names, tokenized_prompts)
        self.context.data.copy_(state["context"].to(target_device))

        self.last_lr = state.get("lr", self.last_lr)
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer = torch.optim.Adam([self.context], lr=self.last_lr or 1e-3)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self.optimizer = None
