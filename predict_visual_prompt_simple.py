import argparse
import os
from typing import List, Tuple

import torch
from PIL import Image
import supervision as sv

from ultralytics import YOLOE
from ultralytics.nn.text_model import build_text_model


def parse_boxes(box_args: List[str]) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for b in box_args:
        parts = [p.strip() for p in b.split(",")]
        assert len(parts) == 4, f"--bbox expects x1,y1,x2,y2; got: {b}"
        x1, y1, x2, y2 = map(float, parts)
        boxes.append((x1, y1, x2, y2))
    return boxes


def get_preprocess(variant: str):
    base, size = variant.split(":")
    if base == "clip":
        import clip as _clip
        _, preprocess = _clip.load(size, device="cpu")
        return preprocess
    elif base == "mobileclip":
        import mobileclip as _mclip
        size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}
        mapped = size_map.get(size, size)
        try:
            _, preprocess, _ = _mclip.create_model_and_transforms(
                f"mobileclip_{mapped}", pretrained=f"mobileclip_{size}.pt"
            )
        except Exception:
            try:
                _, preprocess, _ = _mclip.create_model_and_transforms(f"mobileclip_{mapped}")
            except Exception:
                preprocess = None
        if preprocess is not None:
            return preprocess
        # fallback
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        return Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unknown text model base: {base}")


def crop_rois(image: Image.Image, boxes: List[Tuple[float, float, float, float]]) -> List[Image.Image]:
    W, H = image.size
    crops: List[Image.Image] = []
    for x1, y1, x2, y2 in boxes:
        x1i = int(max(0, min(W - 1, int(x1))))
        y1i = int(max(0, min(H - 1, int(y1))))
        x2i = int(max(1, min(W, int(x2))))
        y2i = int(max(1, min(H, int(y2))))
        if x2i <= x1i:
            x2i = min(W, x1i + 1)
        if y2i <= y1i:
            y2i = min(H, y1i + 1)
        crops.append(image.crop((x1i, y1i, x2i, y2i)))
    return crops


class SimpleCoOpCLIP(torch.nn.Module):
    """簡化版 CoOp，類似原版設計"""
    
    def __init__(self, classnames, text_model, num_ctx=16, per_class=False):
        super().__init__()
        self.classnames = classnames
        self.text_model = text_model
        self.num_ctx = num_ctx
        self.per_class = per_class
        
        # 建立 PromptLearner（無條件化）
        from ultralytics.nn.prompt import PromptLearner
        self.prompt_learner = PromptLearner(
            classnames=classnames,
            text_model=text_model,
            num_ctx=num_ctx,
            per_class=per_class,
            cond_dim=0,  # 無條件化
        )
        
        # 預先計算 tokenized prompts
        self.tokenized_prompts = self.prompt_learner.build_name_token_ids()
        
    def forward(self, image=None):
        """類似原版 CoOp 的 forward"""
        # 直接產生文字特徵（無條件化）
        text_features = self.prompt_learner.forward_and_encode(normalize=True)
        return text_features


def main():
    # 固定使用者提供的影像與框作為預設
    PROMPT_IMAGE = "/Users/patrick/Desktop/yoloe/COOP_YOLO_img/WhatsApp Image 2025-09-30 at 18.26.11.jpeg"
    TARGET_IMAGE = "/Users/patrick/Desktop/yoloe/COOP_YOLO_img/WhatsApp Image 2025-09-30 at 18.26.12.jpeg"

    PROMPT_BOXES_XYXY: List[Tuple[float, float, float, float]] = [
        (726, 854, 885, 1302),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt", help="YOLOE checkpoint")
    parser.add_argument("--names", nargs="+", default=["object"], help="Class names")
    parser.add_argument("--device", type=str, default="cpu", help="cuda:x or cpu")
    parser.add_argument("--num_ctx", type=int, default=16)
    parser.add_argument("--per_class", action="store_true")
    parser.add_argument("--tune_steps", type=int, default=0, help=">0 to enable light prompt tuning")
    parser.add_argument("--tune_lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07, help="logit temperature for CE (multi-class)")
    parser.add_argument("--output", type=str, help="Path to save annotated image")
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(TARGET_IMAGE)
        args.output = f"{base}-output{ext}"

    # Load image
    prompt_pil = Image.open(PROMPT_IMAGE).convert("RGB")
    target_pil = Image.open(TARGET_IMAGE).convert("RGB")
    boxes = PROMPT_BOXES_XYXY
    roi_images = crop_rois(prompt_pil, boxes)

    # Build models
    device = torch.device(args.device)
    model = YOLOE(args.checkpoint)
    model.to(device)

    # 從 YOLOE checkpoint 讀出所用的文字模型設定
    variant = model.model.args.get("text_model")
    text_model = build_text_model(variant, device=device)
    preprocess = get_preprocess(variant)

    # 建立簡化版 CoOp
    coop_model = SimpleCoOpCLIP(
        classnames=args.names,
        text_model=text_model,
        num_ctx=args.num_ctx,
        per_class=args.per_class
    ).to(device)

    # 可選的輕量 tuning（類似原版 CoOp 的訓練）
    if args.tune_steps > 0:
        print(f"開始 {args.tune_steps} 步輕量 tuning...")
        
        # 準備 ROI 特徵作為監督信號
        crop_tensors = torch.stack([preprocess(roi) for roi in roi_images], dim=0).to(device)
        with torch.no_grad():
            img_feats = text_model.model.encode_image(crop_tensors)
            if isinstance(img_feats, (tuple, list)):
                img_feats = img_feats[0]
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        
        # 設定優化器（只優化 context tokens）
        optimizer = torch.optim.Adam([coop_model.prompt_learner.context_tokens], lr=args.tune_lr)
        
        # 簡單的對比學習目標
        for step in range(args.tune_steps):
            optimizer.zero_grad()
            
            # 前向傳播
            text_features = coop_model.forward()
            
            # 計算相似度損失
            if len(args.names) > 1:
                # 多類別：使用 cross entropy
                logits = (img_feats @ text_features.t()) / args.temperature
                target_idx = logits.argmax(dim=-1)
                loss = torch.nn.functional.cross_entropy(logits, target_idx)
            else:
                # 單類別：使用 cosine similarity
                sim = (img_feats @ text_features.t()).mean()
                loss = 1.0 - sim
            
            loss.backward()
            optimizer.step()
            
            print(f"Step {step+1}/{args.tune_steps} - Loss: {loss.item():.6f}")

    # 產生最終的文字特徵並注入 YOLOE
    text_features = coop_model.forward()  # [N, D]
    tpe = text_features.view(1, len(args.names), -1)  # [1, N, D]
    
    print(f"[debug] text_features.shape={tuple(text_features.shape)}")
    print(f"[debug] tpe.shape={tuple(tpe.shape)}")
    print(f"[debug] class names: {args.names}")
    
    model.set_classes(args.names, tpe)

    # 執行檢測
    results = model.predict(target_pil, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    
    print(f"[debug] Detection results:")
    print(f"[debug] - Number of detections: {len(detections)}")
    if len(detections) > 0:
        print(f"[debug] - Confidence scores: {detections.confidence}")
        print(f"[debug] - Class names: {detections['class_name']}")
    else:
        print(f"[debug] - No detections found")

    # 標註和儲存
    annotated_image = target_pil.copy()
    resolution_wh = target_pil.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    annotated_image.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
