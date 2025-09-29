# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import Optional

from ultralytics.data.visual_prompt_utils import extract_support_crops
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, YOLOEModel, YOLOESegModel
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOE if model filename contains 'yoloe'."""
        path = Path(model)
        if "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOE(Model):
    """YOLOE object detection and segmentation model."""

    def __init__(self, model="yoloe-v8s-seg.pt", task=None, verbose=False) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        assert(isinstance(self.model, YOLOEModel))
        return self.model.get_text_pe(texts)
    
    def get_visual_pe(self, img, visual):
        assert(isinstance(self.model, YOLOEModel))
        return self.model.get_visual_pe(img, visual)

    @staticmethod
    def _prepare_support_image(visual_prompt) -> torch.Tensor:
        if isinstance(visual_prompt, torch.Tensor):
            tensor = visual_prompt.clone()
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError("Expected a single visual prompt image when providing a batched tensor.")
                tensor = tensor.squeeze(0)
        elif isinstance(visual_prompt, np.ndarray):
            array = visual_prompt
            if array.ndim == 2:
                array = np.expand_dims(array, axis=-1)
            tensor = torch.from_numpy(array)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.permute(2, 0, 1)
        elif isinstance(visual_prompt, Image.Image):
            tensor = torch.from_numpy(np.array(visual_prompt))
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            tensor = tensor.permute(2, 0, 1)
        else:
            raise TypeError("visual_prompt must be a PIL image, numpy array, or torch tensor.")

        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def tune_text_prompt(
        self,
        visual_prompt,
        boxes,
        class_names,
        box_format: str = "xyxy",
        normalized: bool = False,
        support_size: Optional[int] = None,
        tuner_cfg: Optional[dict] = None,
    ):
        """Tune text prompts using a single visual prompt image and bounding boxes."""

        if not isinstance(self.model, YOLOEModel):
            raise TypeError("Text prompt tuning is only supported for YOLOE models.")

        image_tensor = self._prepare_support_image(visual_prompt)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4).")
        if len(class_names) != boxes_tensor.shape[0]:
            raise ValueError("Number of class names must match number of bounding boxes.")

        fmt = box_format.lower()
        if fmt not in {"xyxy", "xywh"}:
            raise ValueError("box_format must be either 'xyxy' or 'xywh'.")

        boxes_tensor = boxes_tensor.clone()
        if fmt == "xyxy":
            x1, y1, x2, y2 = boxes_tensor.unbind(dim=1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
        else:
            cx, cy, bw, bh = boxes_tensor.unbind(dim=1)

        if not normalized:
            height, width = image_tensor.shape[1:]
            cx = cx / width
            cy = cy / height
            bw = bw / width
            bh = bh / height

        normalized_boxes = torch.stack([cx, cy, bw, bh], dim=1)
        crop_size = support_size or getattr(self.model.args, "coop_support_size", 224)
        support_crops, _ = extract_support_crops(image_tensor, normalized_boxes, int(crop_size))
        labels = torch.arange(len(class_names), dtype=torch.long)
        support_payload = {"images": support_crops, "cls": labels}

        tuned_pe, tuner = self.model.tune_with_visual_support(
            support_payload,
            class_names,
            tuner_cfg or {},
        )

        if self.predictor:
            self.predictor.model.names = self.model.names

        return tuned_pe, tuner

    def set_vocab(self, vocab, names):
        assert(isinstance(self.model, YOLOEModel))
        self.model.set_vocab(vocab, names=names)
    
    def get_vocab(self, names):
        assert(isinstance(self.model, YOLOEModel))
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        assert(isinstance(self.model, YOLOEModel))
        self.model.set_classes(classes, embeddings)
        # Remove background if it's given
        assert(" " not in classes)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
