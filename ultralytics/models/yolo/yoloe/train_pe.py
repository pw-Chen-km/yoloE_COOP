# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch, YOLOETrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from copy import deepcopy
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from copy import copy
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG, RANK, LOGGER
from ultralytics.utils.torch_utils import de_parallel

class YOLOEPETrainer(DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self._prompt_tuned = False
        self._head_prepared = False
        self._coop_tuner: Optional[Any] = None
        save_target = getattr(self.args, "coop_tuner_save_path", None)
        self._coop_tuner_save_path = Path(save_target) if save_target else self.wdir / "tuned_prompt.coop.pt"
        self.add_callback("on_train_end", lambda _: self._save_tuned_prompt())

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        if weights:
            model.load(weights)

        model.eval()
        self._head_prepared = False
        names = self._get_class_names()
        if names:
            model.names = names

        tuner_state = getattr(self.args, "coop_tuner_state", None)
        if tuner_state is not None:
            tuner_cfg = self._build_tuner_cfg()
            model.set_classes(names, tuner_state=tuner_state, tuner_cfg=tuner_cfg)
            self._prompt_tuned = True
            self._coop_tuner = getattr(model, "coop_tuner", None)
            self._prepare_head_for_finetune(model)
        elif self.args.train_pe_path:
            pe_state = torch.load(self.args.train_pe_path)
            model.set_classes(pe_state["names"], embeddings=pe_state["pe"])
            self._prompt_tuned = True
            self._coop_tuner = getattr(model, "coop_tuner", None)
            self._prepare_head_for_finetune(model)
        else:
            self._prompt_tuned = False
            self._head_prepared = False
            self._coop_tuner = None

        return model

    def _get_class_names(self) -> Sequence[str]:
        names = self.data.get("names") if isinstance(self.data, dict) else None
        if names is None:
            return []
        if isinstance(names, dict):
            return list(names.values())
        return list(names)

    def _build_tuner_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        text_model = getattr(self.args, "text_model", None)
        if text_model:
            cfg["base_variant"] = text_model
        ctx_len = getattr(self.args, "coop_ctx_len", None)
        if ctx_len:
            cfg["ctx_len"] = ctx_len
        template = getattr(self.args, "coop_template", None)
        if template:
            cfg["template"] = template
        steps = getattr(self.args, "coop_tuner_steps", None)
        if steps is not None:
            cfg["steps"] = steps
        lr = getattr(self.args, "coop_tuner_lr", None)
        if lr is not None:
            cfg["lr"] = lr
        tuner_state = getattr(self.args, "coop_tuner_state", None)
        if tuner_state is not None:
            if isinstance(tuner_state, (str, Path)):
                cfg["state_path"] = tuner_state
            else:
                cfg["state"] = tuner_state
        return cfg

    def _prepare_head_for_finetune(self, model: YOLOEModel) -> None:
        if self._head_prepared:
            return
        head = model.model[-1]
        if hasattr(model, "pe"):
            head.fuse(model.pe)
        head.cv3[0][2] = deepcopy(head.cv3[0][2]).requires_grad_(True)
        head.cv3[1][2] = deepcopy(head.cv3[1][2]).requires_grad_(True)
        head.cv3[2][2] = deepcopy(head.cv3[2][2]).requires_grad_(True)
        if hasattr(model, "pe"):
            del model.pe
        model.train()
        self._head_prepared = True

    def _ensure_prompt_initialized(self, batch: Dict[str, Any]) -> None:
        if self._prompt_tuned:
            model = de_parallel(self.model)
            if hasattr(model, "pe") and not self._head_prepared:
                self._prepare_head_for_finetune(model)
            return

        support = batch.get("vp_crops")
        if not isinstance(support, dict):
            return
        cls_tensor = support.get("cls")
        if not isinstance(cls_tensor, torch.Tensor) or not (cls_tensor >= 0).any():
            return

        model = de_parallel(self.model)
        tuner_cfg = self._build_tuner_cfg()
        names = self._get_class_names()
        try:
            _, tuner = model.tune_with_visual_support(support, names, tuner_cfg)
        except ValueError as err:
            LOGGER.warning(f"Skipping prompt tuning for current batch: {err}")
            return

        self._prompt_tuned = True
        self._coop_tuner = tuner
        self._prepare_head_for_finetune(model)

    def _save_tuned_prompt(self) -> None:
        model = de_parallel(self.model)
        tuner = self._coop_tuner or getattr(model, "coop_tuner", None)
        if tuner is None or getattr(tuner, "context", None) is None:
            return
        save_path = self._coop_tuner_save_path
        tuner.save_state(save_path)
        LOGGER.info(f"Saved tuned prompt state to {save_path}")

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._ensure_prompt_initialized(batch)
        return batch
    
class YOLOEPESegTrainer(YOLOEPETrainer, SegmentationTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        
        del model.model[-1].savpe
        
        if weights:
            model.load(weights)
        
        model.eval()
        self._head_prepared = False
        names = self._get_class_names()
        if names:
            model.names = names

        tuner_state = getattr(self.args, "coop_tuner_state", None)
        if tuner_state is not None:
            tuner_cfg = self._build_tuner_cfg()
            model.set_classes(names, tuner_state=tuner_state, tuner_cfg=tuner_cfg)
            self._prompt_tuned = True
            self._coop_tuner = getattr(model, "coop_tuner", None)
            self._prepare_head_for_finetune(model)
        elif self.args.train_pe_path:
            pe_state = torch.load(self.args.train_pe_path)
            model.set_classes(pe_state["names"], embeddings=pe_state["pe"])
            self._prompt_tuned = True
            self._coop_tuner = getattr(model, "coop_tuner", None)
            self._prepare_head_for_finetune(model)
        else:
            self._prompt_tuned = False
            self._head_prepared = False
            self._coop_tuner = None

        return model

class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOE training, adjusting formatting and dimensions as needed."""
        batch = super(YOLOETrainer, self).preprocess_batch(batch)
        return batch