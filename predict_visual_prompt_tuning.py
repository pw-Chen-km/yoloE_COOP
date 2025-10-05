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
        # load returns (model, preprocess). We only need preprocess; do not reuse model from here
        _, preprocess = _clip.load(size, device="cpu")
        return preprocess
    elif base == "mobileclip":
        import mobileclip as _mclip
        # map size alias consistent with our wrapper (blt -> b)
        size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}
        mapped = size_map.get(size, size)
        # try with pretrained to ensure transforms are populated
        try:
            _, preprocess, _ = _mclip.create_model_and_transforms(
                f"mobileclip_{mapped}", pretrained=f"mobileclip_{size}.pt"
            )
        except Exception:
            # fallback without pretrained
            try:
                _, preprocess, _ = _mclip.create_model_and_transforms(f"mobileclip_{mapped}")
            except Exception:
                preprocess = None
        if preprocess is not None:
            return preprocess
        # final fallback: build a standard CLIP-like preprocess
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


def main():
    # 固定使用者提供的影像與框作為預設
    PROMPT_IMAGE = \
        "/Users/patrick/Desktop/yoloe/COOP_YOLO_img/WhatsApp Image 2025-09-30 at 18.26.11.jpeg"
    TARGET_IMAGE = \
        "/Users/patrick/Desktop/yoloe/COOP_YOLO_img/WhatsApp Image 2025-09-30 at 18.26.12.jpeg"

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

    # 從 YOLOE checkpoint 讀出所用的文字模型設定（與 predict_text_prompt.py 一致的環境）
    variant = model.model.args.get("text_model")
    print(f"[debug] variant: {variant}")
    text_model = build_text_model(variant, device=device)
    print(f"[debug] text_model type: {type(text_model)}")
    print(f"[debug] has preprocess: {hasattr(text_model, 'preprocess')}")
    if hasattr(text_model, 'preprocess'):
        print(f"[debug] preprocess type: {type(text_model.preprocess)}")
    # 優先使用封裝內的 preprocess，必要時退回舊邏輯
    preprocess = getattr(text_model, "preprocess", get_preprocess(variant))
    print(f"[debug] final preprocess type: {type(preprocess)}")

    # Encode visual prompt crops -> cond vector (mean across rois)
    crop_tensors = torch.stack([preprocess(roi) for roi in roi_images], dim=0).to(device)
    
    # Debug: 檢查前處理結果
    print(f"[debug] preprocess type: {type(preprocess)}")
    print(f"[debug] crop_tensors shape: {crop_tensors.shape}")
    print(f"[debug] crop_tensors mean: {crop_tensors.mean():.4f}")
    print(f"[debug] crop_tensors std: {crop_tensors.std():.4f}")
    print(f"[debug] crop_tensors min/max: {crop_tensors.min():.4f}/{crop_tensors.max():.4f}")
    
    # 使用封裝統一的 encode_image（內部已做 L2 normalize）
    # 將 Inference tensor 轉成一般 tensor，讓 backward 可以保存
    img_feats = text_model.encode_image(crop_tensors).clone().detach()  # [M, D]
    cond_vec = img_feats.mean(dim=0, keepdim=True)  # [1, D]
    # debug: print inputs status
    print(f"[debug] roi_images={len(roi_images)} | img_feats.shape={tuple(img_feats.shape)} | cond_vec.shape={tuple(cond_vec.shape)}")
    print(f"[debug] img_feats norm after L2: {img_feats.norm(p=2, dim=-1)}")
    print(f"[debug] cond_vec norm: {cond_vec.norm(p=2, dim=-1)}")

    # Construct PromptLearner after knowing cond dim
    from ultralytics.nn.prompt import PromptLearner
    learner = PromptLearner(
        classnames=args.names,
        text_model=text_model,
        num_ctx=args.num_ctx,
        per_class=args.per_class,
    cond_dim=0,
    ).to(device)

    # 深入除錯：檢查 tokenizer 與 embedding 輸出是否為 0
    try:
        tok_ids = learner.build_name_token_ids()
        print(f"[debug] token_ids shape: {tuple(tok_ids.shape)}, dtype: {tok_ids.dtype}")
        print(f"[debug] token_ids min/max: {tok_ids.min().item()}/{tok_ids.max().item()}")
        # unique 可能很長，僅列出前 16 個
        uniq = torch.unique(tok_ids)
        print(f"[debug] token_ids unique[:16]: {uniq[:16].tolist()}")

        tok_emb = text_model.token_embedding
        print(f"[debug] token_embedding weight shape: {tuple(tok_emb.weight.shape)}, dtype: {tok_emb.weight.dtype}")
        print(f"[debug] token_embedding weight min/max/mean: {tok_emb.weight.min().item():.6f}/{tok_emb.weight.max().item():.6f}/{tok_emb.weight.mean().item():.6f}")

        name_emb = tok_emb(tok_ids)
        print(f"[debug] name_emb shape: {tuple(name_emb.shape)}, dtype: {name_emb.dtype}")
        print(f"[debug] name_emb min/max/mean: {name_emb.min().item():.6f}/{name_emb.max().item():.6f}/{name_emb.mean().item():.6f}")

        emb_seq, new_tok_ids = learner.assemble(tok_ids, cond=None)
        print(f"[debug] emb_seq shape: {tuple(emb_seq.shape)}, dtype: {emb_seq.dtype}")
        print(f"[debug] emb_seq min/max/mean: {emb_seq.min().item():.6f}/{emb_seq.max().item():.6f}/{emb_seq.mean().item():.6f}")
        print(f"[debug] new_token_ids shape: {tuple(new_tok_ids.shape)} | last idx nonzero? {(new_tok_ids[:, -1] != 0).tolist()}")

        tf_probe = learner.forward_and_encode(cond=None, normalize=True)
        print(f"[debug] probe text_feats shape: {tuple(tf_probe.shape)}")
        print(f"[debug] probe text_feats min/max/mean: {tf_probe.min().item():.6f}/{tf_probe.max().item():.6f}/{tf_probe.mean().item():.6f}")
    except Exception as e:
        print(f"[debug] probing text pipeline failed: {e}")

    # Optional light tuning on crops (optimize context only)
    if args.tune_steps > 0:
        params = [learner.context_tokens]
        optim = torch.optim.Adam(params, lr=args.tune_lr)
        # Pseudo labels for multi-class: use current text to get argmax as labels
        num_classes = len(args.names)
        if num_classes > 1:
            with torch.no_grad():
                tf_init = learner.forward_and_encode(cond=None, normalize=True)  # [N,D]
                logits0 = (img_feats @ tf_init.t()) / max(args.temperature, 1e-6)
                target_idx = logits0.argmax(dim=-1)
                print(f"[debug] init text_feats.shape={tuple(tf_init.shape)} | logits0.shape={tuple(logits0.shape)} | target_idx.shape={tuple(target_idx.shape)} | target_idx.unique={target_idx.unique().tolist()}")
        else:
            target_idx = None  # single-class path doesn't use CE
        for i in range(args.tune_steps):
            optim.zero_grad(set_to_none=True)
            text_feats = learner.forward_and_encode(cond=None, normalize=True)  # [N,D]
            if i == 0:
                print(f"[debug] step0 text_feats.shape={tuple(text_feats.shape)}")
            if num_classes > 1:
                logits = (img_feats @ text_feats.t()) / max(args.temperature, 1e-6)  # [M,N]
                # Debug: 檢查 multi-class 分支的輸入
                if i == 0:
                    print(f"[debug] multi-class branch:")
                    print(f"[debug] img_feats shape: {img_feats.shape}")
                    print(f"[debug] img_feats min/max/mean: {img_feats.min():.6f}/{img_feats.max():.6f}/{img_feats.mean():.6f}")
                    print(f"[debug] text_feats shape: {text_feats.shape}")
                    print(f"[debug] text_feats min/max/mean: {text_feats.min():.6f}/{text_feats.max():.6f}/{text_feats.mean():.6f}")
                    print(f"[debug] logits shape: {logits.shape}")
                    print(f"[debug] logits min/max/mean: {logits.min():.6f}/{logits.max():.6f}/{logits.mean():.6f}")
                    print(f"[debug] target_idx: {target_idx}")
                loss = torch.nn.functional.cross_entropy(logits, target_idx)
            else:
                # single-class: minimize (1 - cosine similarity)
                sim = img_feats @ text_feats.t()  # [M,1]
                # Debug: 檢查 single-class 分支的輸入
                if i == 0:
                    print(f"[debug] single-class branch:")
                    print(f"[debug] img_feats shape: {img_feats.shape}")
                    print(f"[debug] img_feats min/max/mean: {img_feats.min():.6f}/{img_feats.max():.6f}/{img_feats.mean():.6f}")
                    print(f"[debug] text_feats shape: {text_feats.shape}")
                    print(f"[debug] text_feats min/max/mean: {text_feats.min():.6f}/{text_feats.max():.6f}/{text_feats.mean():.6f}")
                    print(f"[debug] sim shape: {sim.shape}")
                    print(f"[debug] sim min/max/mean: {sim.min():.6f}/{sim.max():.6f}/{sim.mean():.6f}")
                loss = (1.0 - sim).mean()
            loss.backward()
            optim.step()
            print(f"tune step {i+1}/{args.tune_steps} - loss: {loss.item():.6f}")

    # Build final TPE and run detection
    text_feats = learner.forward_and_encode(cond=None, normalize=True)  # [N,D]
    tpe = text_feats.view(1, len(args.names), -1)
    
    # Debug: 檢查文字特徵和 TPE
    print(f"[debug] text_feats.shape={tuple(text_feats.shape)}")
    print(f"[debug] text_feats norm: {text_feats.norm(p=2, dim=-1)}")
    print(f"[debug] tpe.shape={tuple(tpe.shape)}")
    print(f"[debug] class names: {args.names}")
    
    model.set_classes(args.names, tpe)

    results = model.predict(target_pil, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Debug: 檢查檢測結果
    print(f"[debug] Detection results:")
    print(f"[debug] - Number of detections: {len(detections)}")
    if len(detections) > 0:
        print(f"[debug] - Confidence scores: {detections.confidence}")
        print(f"[debug] - Class names: {detections['class_name']}")
        print(f"[debug] - Bounding boxes shape: {detections.xyxy.shape}")
    else:
        print(f"[debug] - No detections found")

    # Annotate and save
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


