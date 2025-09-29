from pathlib import Path

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

os.environ["PYTHONHASHSEED"] = "0"

data = "ultralytics/cfg/datasets/coco.yaml"

model_path = "yoloe-v8l-seg.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8l-seg.pt")

# Prepare prompt tuning assets
names = list(yaml_load(data)["names"].values())
support_path = Path("support/coco_support.pt")
base_tuner_state = Path("coop_states/coco_base.coop.pt")
pe_path = Path("artifacts/coco-pe.pt")
tuned_prompt_path = Path("artifacts/coco-tuned.coop.pt")
pe_path.parent.mkdir(parents=True, exist_ok=True)
tuned_prompt_path.parent.mkdir(parents=True, exist_ok=True)

tuner_cfg = {}
if base_tuner_state.exists():
    tuner_cfg["state_path"] = str(base_tuner_state)

if support_path.exists():
    support_payload = torch.load(support_path)
    support_crops = support_payload.get("vp_crops", support_payload)
    class_names = support_payload.get("names", names)
    tuned_pe, tuner = model.tune_with_visual_support(support_crops, class_names, tuner_cfg)
    tuned_pe_cpu = tuned_pe.cpu()
    torch.save({"names": class_names, "pe": tuned_pe_cpu}, pe_path)
    tuner.save_state(tuned_prompt_path)
    coop_state_for_training = str(tuned_prompt_path)
    names = class_names
else:
    LOGGER.warning("Support crops not found; falling back to text prompt embeddings.")
    tpe = model.get_text_pe(names).cpu()
    torch.save({"names": names, "pe": tpe}, pe_path)
    coop_state_for_training = str(base_tuner_state) if base_tuner_state.exists() else None

head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if 'cv3' not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend([f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1", f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1", f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1"])

train_kwargs = dict(
    data=data,
    epochs=10,
    close_mosaic=5,
    batch=128,
    optimizer="AdamW",
    lr0=1e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    device="0,1,2,3,4,5,6,7",
    trainer=YOLOEPESegTrainer,
    freeze=freeze,
    train_pe_path=str(pe_path),
    coop_tuner_save_path=str(tuned_prompt_path),
    **extends,
)

if coop_state_for_training:
    train_kwargs["coop_tuner_state"] = coop_state_for_training

model.train(**train_kwargs)